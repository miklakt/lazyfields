"""Read-only OS mount of a filesystem through FUSE.

Groups appear as directories. Scalars and datasets appear as files under their own
names holding raw contents: text for scalars and lists, buffer bytes for arrays.
Presenters add further files beside a leaf, such as ``preview``. Listing reads no
dataset values; content is produced when a file is opened. Requires the optional
``mfusepy`` package and a FUSE implementation (libfuse, macFUSE, WinFsp).

Command line: ``lf-mount SPEC MOUNTPOINT [--relative-to DIR]``, where SPEC
is a specification file or an inline JSON string.
"""

import argparse
import errno
import shutil
import stat
import subprocess
import time
from itertools import count
from pathlib import Path

from .filesystem import FileSystem
from .specification import add_spec_arguments, filesystem_from_arguments
from ..readers import require_reader

PREVIEW_ROWS = 5
# Sources without direct reads load whole files; skip previews above this size.
PREVIEW_LOAD_LIMIT = 50 * 1024 * 1024


def raw(value) -> bytes:
    """Bytes as stored, buffer bytes for numeric arrays, otherwise text."""
    if isinstance(value, bytes):
        return value
    dtype = getattr(value, "dtype", None)
    if dtype is not None and not dtype.hasobject and hasattr(value, "tobytes"):
        return value.tobytes()
    return f"{value}\n".encode()


def preview(filesystem: FileSystem, path: str, kind: str):
    """Presenter adding ``<name>.preview.txt`` with a dataset's leading rows."""
    if kind != "dataset":
        return None

    def render():
        node = filesystem.node(path)
        origin = node.origin
        if origin is None or require_reader(origin.file_path).read is None:
            total = sum(file.stat().st_size for file in node.files())
            if total > PREVIEW_LOAD_LIMIT:
                return f"{path}: preview skipped, it would load {total / 2**20:.0f} MiB\n"
        value = filesystem.read(path, slice(0, PREVIEW_ROWS))
        try:
            import numpy as np
            text = np.array2string(np.asarray(value), threshold=50, edgeitems=3, max_line_width=100)
        except (ImportError, ValueError):
            text = repr(value)
        return f"{path}: first {PREVIEW_ROWS} rows\n{text}\n"

    return ".preview.txt", render


def mount(filesystem: FileSystem, mountpoint: str | Path, *, presenters=(), foreground: bool = True) -> None:
    """Mount read-only until the process ends or ``fusermount -u mountpoint``.

    A presenter ``(filesystem, path, kind)`` returns ``(suffix, render)`` to add
    ``<name><suffix>`` beside a scalar or dataset, or None. It runs while listing,
    so it must not read values; ``render()`` runs on open and returns bytes or str.
    """
    import mfusepy as fuse

    class Operations(fuse.Operations):
        use_ns = True

        def __init__(self):
            self.mounted = time.time_ns()
            self.handles = count(1)
            self.contents = {}
            # Sizes are known once rendered; until then files report zero.
            self.sizes = {}
            # Whole-file readers reload per lookup, so keep each group's entries
            # {os name: (path, kind, render)} together with the group's digest.
            self.groups = {}

        def entries(self, path):
            digest = filesystem.digest(path)
            cached = self.groups.get(path)
            if cached is None or cached[0] != digest:
                entries = {}
                for name, kind, value in filesystem.items(path):
                    child = f"{path}/{name}" if path else name
                    if kind == "group":
                        entries[name] = (child, kind, None)
                        continue
                    entries[name] = (child, kind, (lambda value=value: f"{value}\n") if kind == "scalar"
                                     else (lambda child=child: raw(filesystem.read(child))))
                    for presenter in presenters:
                        presented = presenter(filesystem, child, kind)
                        if presented is not None:
                            suffix, render = presented
                            entries[name + suffix] = (child, kind, render)
                cached = self.groups[path] = (digest, entries)
            return cached[1]

        def resolve(self, path):
            path = path.strip("/")
            if not path:
                return "", "group", None
            parent, _, name = path.rpartition("/")
            try:
                return self.entries(parent)[name]
            except (KeyError, IndexError, TypeError, ValueError, NotADirectoryError, FileNotFoundError):
                raise fuse.FuseOSError(errno.ENOENT) from None

        def getattr(self, path, fh=None):
            size = self.sizes.get(path, 0)
            path, kind, _ = self.resolve(path)
            try:  # The latest of its files; a node without files reports the mount time.
                mtime = max(file.stat().st_mtime_ns for file in filesystem.node(path).files())
            except (OSError, ValueError):
                mtime = self.mounted
            group = kind == "group"
            # Direct I/O reads unopened files to their end despite the zero size.
            return {"st_mode": stat.S_IFDIR | 0o555 if group else stat.S_IFREG | 0o444,
                    "st_nlink": 2 if group else 1, "st_size": size,
                    "st_mtime": mtime, "st_ctime": mtime, "st_atime": mtime}

        def readdir(self, path, fh):
            path, kind, _ = self.resolve(path)
            if kind != "group":
                raise fuse.FuseOSError(errno.ENOTDIR)
            return [".", "..", *self.entries(path)]

        def open(self, path, flags):
            _, kind, render = self.resolve(path)
            if kind == "group":
                raise fuse.FuseOSError(errno.EISDIR)
            try:
                content = render()
            except Exception as exc:
                content = f"Error: {type(exc).__name__}: {exc}\n"
            handle = next(self.handles)
            self.contents[handle] = content.encode() if isinstance(content, str) else content
            self.sizes[path] = len(self.contents[handle])
            return handle

        def read(self, path, size, offset, fh):
            return self.contents[fh][offset:offset + size]

        def release(self, path, fh):
            self.contents.pop(fh, None)
            return 0

    # Keep attributes uncached so stat reflects the size after rendering.
    fuse.FUSE(Operations(), str(mountpoint), foreground=foreground, nothreads=True,
              ro=True, direct_io=True, attr_timeout=0, entry_timeout=60, negative_timeout=60)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="lf-mount",
        description="Mount a filesystem specification read-only in the OS. "
                    "The mount ends with the process: Ctrl+C, closing the terminal, or kill.")
    add_spec_arguments(parser)
    parser.add_argument("mountpoint", type=Path, help="empty directory to mount at; created if missing")
    args = parser.parse_args(argv)

    filesystem = filesystem_from_arguments(args)
    try:
        args.mountpoint.stat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        # A killed mount leaves a dead endpoint at the mount point; release it.
        if exc.errno != errno.ENOTCONN:
            raise
        subprocess.run([shutil.which("fusermount3") or "fusermount", "-uz", str(args.mountpoint)], check=True)
    created = not args.mountpoint.exists()
    args.mountpoint.mkdir(parents=True, exist_ok=True)
    print(f"Mounted at {args.mountpoint}; press Ctrl+C or close the terminal to unmount.", flush=True)
    try:
        mount(filesystem, args.mountpoint)
    finally:
        # Remove only a mount point this command created; it is empty once unmounted.
        if created:
            try:
                args.mountpoint.rmdir()
            except OSError:
                pass


if __name__ == "__main__":
    main()
