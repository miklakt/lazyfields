"""Optional JSON reader with memory and SQLite caches of byte ranges."""

import hashlib
import json
import mmap
import sqlite3
import threading
from collections import OrderedDict
from contextlib import closing
from pathlib import Path

import pandas as pd

from ..readers import Reader, _path_get, json_load


_SPACE = b" \t\r\n"
_DELIMITER = b",]} \t\r\n"
_QUOTE = ord('"')
_BACKSLASH = ord("\\")
_OBJECT_START, _OBJECT_END = ord("{"), ord("}")
_ARRAY_START, _ARRAY_END = ord("["), ord("]")
_COLON, _COMMA = ord(":"), ord(",")
_VALUE_KINDS = {
    _OBJECT_START: "object", _ARRAY_START: "array", _QUOTE: "string",
    ord("t"): "boolean", ord("f"): "boolean", ord("n"): "null",
}
_MIB = 1024 * 1024
_DEFAULT_SIZE_CUTOFF = _MIB
_DEFAULT_MAX_CACHED_FILES = 32
_DEFAULT_MAX_CACHE_BYTES = 64 * _MIB
_SQLITE_TIMEOUT_SECONDS = 30
# Approximate Python dict entry, tuple, and integer overhead per indexed key.
_OFFSET_ENTRY_ESTIMATE_BYTES = 96


class _Scanner:
    def __init__(self, data):
        self.data = data
        self.end = len(data)

    def space(self, pos):
        while pos < self.end and self.data[pos] in _SPACE:
            pos += 1
        return pos

    def string(self, pos):
        start = pos
        pos += 1
        while pos < self.end:
            mark = self.data.find(b'"', pos)
            if mark < 0:
                break
            slash = mark - 1
            while slash > start and self.data[slash] == _BACKSLASH:
                slash -= 1
            if (mark - 1 - slash) % 2 == 0:
                return mark + 1
            pos = mark + 1
        raise ValueError("Unterminated JSON string")

    def value(self, pos):
        pos = self.space(pos)
        if pos >= self.end:
            raise ValueError("Unexpected end of JSON")
        kind = self.data[pos]
        if kind == _QUOTE:
            return self.string(pos)
        if kind == _OBJECT_START:
            pos = self.space(pos + 1)
            if self.data[pos] == _OBJECT_END:
                return pos + 1
            while True:
                if self.data[pos] != _QUOTE:
                    raise ValueError("Expected JSON object key")
                finish = self.string(pos)
                key = json.loads(self.data[pos:finish])
                if "/" in key:
                    raise ValueError(f"Key {key!r} contains '/', which is reserved for nested paths.")
                pos = self.space(finish)
                if self.data[pos] != _COLON:
                    raise ValueError("Expected ':' after JSON key")
                pos = self.space(self.value(pos + 1))
                if self.data[pos] == _OBJECT_END:
                    return pos + 1
                if self.data[pos] != _COMMA:
                    raise ValueError("Expected ',' in JSON object")
                pos = self.space(pos + 1)
        if kind == _ARRAY_START:
            pos = self.space(pos + 1)
            if self.data[pos] == _ARRAY_END:
                return pos + 1
            while True:
                pos = self.space(self.value(pos))
                if self.data[pos] == _ARRAY_END:
                    return pos + 1
                if self.data[pos] != _COMMA:
                    raise ValueError("Expected ',' in JSON array")
                pos = self.space(pos + 1)
        finish = pos
        while finish < self.end and self.data[finish] not in _DELIMITER:
            finish += 1
        if finish == pos:
            raise ValueError("Expected JSON value")
        return finish

    def top_level(self):
        pos = self.space(0)
        if pos >= self.end or self.data[pos] != _OBJECT_START:
            raise TypeError("expected a mapping row dictionary.")
        pos = self.space(pos + 1)
        offsets = {}
        if self.data[pos] != _OBJECT_END:
            while True:
                if self.data[pos] != _QUOTE:
                    raise ValueError("Expected JSON object key")
                finish = self.string(pos)
                key = json.loads(self.data[pos:finish])
                if "/" in key:
                    raise ValueError(f"Key {key!r} contains '/', which is reserved for nested paths.")
                pos = self.space(finish)
                if self.data[pos] != _COLON:
                    raise ValueError("Expected ':' after JSON key")
                start = self.space(pos + 1)
                pos = self.space(self.value(start))
                offsets[key] = (start, pos)
                if self.data[pos] == _OBJECT_END:
                    break
                if self.data[pos] != _COMMA:
                    raise ValueError("Expected ',' in JSON object")
                pos = self.space(pos + 1)
        if self.space(pos + 1) != self.end:
            raise ValueError("Extra data after JSON object")
        return offsets

    def locate(self, start, parts):
        pos = start
        for part in parts:
            pos = self.space(pos)
            if self.data[pos] == _OBJECT_START:
                quoted = len(part) >= 2 and part[0] == part[-1] and part[0] in "\"'"
                wanted = part[1:-1] if quoted else part
                cursor = self.space(pos + 1)
                while self.data[cursor] != _OBJECT_END:
                    finish = self.string(cursor)
                    key = json.loads(self.data[cursor:finish])
                    cursor = self.space(finish)
                    begin = self.space(cursor + 1)
                    finish = self.value(begin)
                    if key == wanted:
                        pos = begin
                        break
                    cursor = self.space(finish)
                    if self.data[cursor] == _COMMA:
                        cursor = self.space(cursor + 1)
                else:
                    raise KeyError(wanted)
            elif self.data[pos] == _ARRAY_START:
                if len(part) >= 2 and part[0] == part[-1] and part[0] in "\"'":
                    raise TypeError("Array indices must be integers")
                index = int(part)
                if index < 0:
                    count = 0
                    cursor = self.space(pos + 1)
                    while self.data[cursor] != _ARRAY_END:
                        count += 1
                        cursor = self.space(self.value(cursor))
                        if self.data[cursor] == _COMMA:
                            cursor = self.space(cursor + 1)
                    index += count
                cursor = self.space(pos + 1)
                for _ in range(index):
                    if self.data[cursor] == _ARRAY_END:
                        raise IndexError(index)
                    cursor = self.space(self.value(cursor))
                    if self.data[cursor] == _COMMA:
                        cursor = self.space(cursor + 1)
                if index < 0 or self.data[cursor] == _ARRAY_END:
                    raise IndexError(index)
                pos = cursor
            else:
                raise TypeError("Cannot traverse a JSON scalar")
        return pos, self.value(pos)

    def array_slice(self, start, selection):
        if (not isinstance(selection, slice) or selection.step not in (None, 1) or
                (selection.start is not None and selection.start < 0) or
                (selection.stop is not None and selection.stop < 0)):
            return None
        if self.data[start] != _ARRAY_START:
            return None
        first = selection.start or 0
        stop = selection.stop
        values = []
        cursor = self.space(start + 1)
        index = 0
        while self.data[cursor] != _ARRAY_END and (stop is None or index < stop):
            end = self.value(cursor)
            if index >= first:
                values.append(json.loads(self.data[cursor:end]))
            cursor = self.space(end)
            if self.data[cursor] == _COMMA:
                cursor = self.space(cursor + 1)
            index += 1
        return values


def JSON_FAST(size_cutoff: int = _DEFAULT_SIZE_CUTOFF,
              max_cached_files: int = _DEFAULT_MAX_CACHED_FILES,
              max_cache_bytes: int = _DEFAULT_MAX_CACHE_BYTES,
              cache_dir: str | Path | None = None,
              use_disk_cache: bool = True) -> Reader:
    """Build a JSON reader with offsets above size_cutoff.

    SQLite files default to .lazyfields_cache/json_fast in the current project.
    Pass cache_dir to choose another folder.
    Set use_disk_cache=False for an in-process cache only. Missing or stale
    entries are rebuilt from the source JSON file.
    """
    if size_cutoff < 0 or max_cached_files < 1 or max_cache_bytes < 1:
        raise ValueError("size_cutoff must be nonnegative and cache limits positive")
    cache = OrderedDict()
    lock = threading.RLock()
    if cache_dir is None and use_disk_cache:
        cache_dir = Path.cwd() / ".lazyfields_cache" / "json_fast"
    cache_dir = Path(cache_dir) if cache_dir is not None else None

    def fingerprint(path):
        stat = Path(path).stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)

    def database_path(path):
        digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()
        return cache_dir / f"{digest}.sqlite3"

    def open_database(path):
        cache_dir.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(database_path(path), timeout=_SQLITE_TIMEOUT_SECONDS)
        connection.execute("CREATE TABLE IF NOT EXISTS file_meta "
                           "(id INTEGER PRIMARY KEY CHECK (id = 1), path TEXT NOT NULL, "
                           "device INTEGER NOT NULL, inode INTEGER NOT NULL, "
                           "size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL)")
        connection.execute("CREATE TABLE IF NOT EXISTS fields "
                           "(key TEXT PRIMARY KEY, start INTEGER NOT NULL, "
                           "end INTEGER NOT NULL, kind TEXT NOT NULL)")
        return connection

    def disk_index(path, stamp):
        try:
            with closing(open_database(path)) as connection:
                connection.execute("BEGIN")
                row = connection.execute("SELECT path, device, inode, size, mtime_ns "
                                         "FROM file_meta WHERE id = 1").fetchone()
                if row != (str(path), *stamp):
                    return None
                rows = connection.execute("SELECT key, start, end, kind FROM fields").fetchall()
                if any(start < 0 or end <= start or end > stamp[2] or
                       kind not in ("object", "array", "string", "number", "boolean", "null")
                       for _, start, end, kind in rows):
                    return None
                return {key: (start, end) for key, start, end, _ in rows}
        except (OSError, sqlite3.DatabaseError):
            return None

    def write_disk_index(path, stamp, offsets, kinds):
        try:
            with closing(open_database(path)) as connection:
                with connection:
                    connection.execute("DELETE FROM fields")
                    connection.execute("DELETE FROM file_meta")
                    connection.executemany(
                        "INSERT INTO fields (key, start, end, kind) VALUES (?, ?, ?, ?)",
                        ((key, *offsets[key], kinds[key]) for key in offsets),
                    )
                    connection.execute("INSERT INTO file_meta VALUES (1, ?, ?, ?, ?, ?)",
                                       (str(path), *stamp))
        except (OSError, sqlite3.DatabaseError):
            # The cache is optional; source JSON remains authoritative.
            pass

    def index(path):
        path = Path(path).resolve()
        stamp = fingerprint(path)
        with lock:
            cached = cache.get(path)
            if cached is not None and cached[0] == stamp:
                cache.move_to_end(path)
                return cached[1]
            offsets = disk_index(path, stamp) if use_disk_cache else None
            rebuild = offsets is None
            if rebuild:
                with path.open("rb") as fh:
                    with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as data:
                        offsets = _Scanner(data).top_level()
                        if use_disk_cache:
                            kinds = {
                                key: _VALUE_KINDS.get(data[start], "number")
                                for key, (start, _) in offsets.items()
                            }
            if fingerprint(path) != stamp:
                return index(path)
            if rebuild and use_disk_cache:
                write_disk_index(path, stamp, offsets, kinds)
            estimate = sum(_OFFSET_ENTRY_ESTIMATE_BYTES + len(key.encode("utf-8")) for key in offsets)
            if estimate <= max_cache_bytes:
                cache[path] = (stamp, offsets, estimate)
                cache.move_to_end(path)
                while (len(cache) > max_cached_files or
                       sum(item[2] for item in cache.values()) > max_cache_bytes):
                    cache.popitem(last=False)
            return offsets

    def load(path):
        return json_load(path)

    def select(value, selection):
        if not isinstance(selection, tuple):
            return value[selection]
        def apply(current, parts):
            if not parts:
                return current
            head, *tail = parts
            selected = current[head]
            if isinstance(head, slice) and tail:
                return [apply(item, tail) for item in selected]
            return apply(selected, tail)
        return apply(value, selection)

    def read(path, key, selection=None):
        if Path(path).stat().st_size < size_cutoff:
            value = _path_get(load(path), key)
        else:
            offsets = index(path)
            parts = list(filter(None, key.split("/")))
            if not parts:
                value = load(path)
            else:
                top = parts.pop(0)
                if len(top) >= 2 and top[0] == top[-1] and top[0] in "\"'":
                    top = top[1:-1]
                start, end = offsets[top]
                with Path(path).open("rb") as fh:
                    with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as data:
                        if parts:
                            start, end = _Scanner(data).locate(start, parts)
                        scanner = _Scanner(data)
                        if selection is not None:
                            outer_slice = selection
                            if isinstance(selection, tuple) and selection and all(
                                part == slice(None) for part in selection[1:]
                            ):
                                outer_slice = selection[0]
                            sliced = scanner.array_slice(start, outer_slice)
                            if sliced is not None:
                                return sliced
                        value = json.loads(data[start:end])
        return value if selection is None else select(value, selection)

    def inspect(path):
        if Path(path).stat().st_size < size_cutoff:
            values = load(path)
            if not isinstance(values, dict):
                raise TypeError("expected a mapping row dictionary.")
            yield from ((key, pd.api.types.is_scalar(value), value) for key, value in values.items())
            return
        offsets = index(path)
        with Path(path).open("rb") as fh:
            for key, (start, end) in offsets.items():
                fh.seek(start)
                first = fh.read(1)
                if first in (b"{", b"["):
                    yield key, False, None
                else:
                    fh.seek(start)
                    value = json.loads(fh.read(end - start))
                    yield key, pd.api.types.is_scalar(value), value

    return Reader((".json",), load, read, inspect)
