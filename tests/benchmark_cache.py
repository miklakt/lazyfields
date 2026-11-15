#!/usr/bin/env python3
"""Reads versus RAM of metadata caching, over many JSON and pickle files.

Many files are spawned "virtually": one template file per format is symlinked N times, so every
path is a distinct file the readers parse, while the disk holds one copy. Each workload runs
in a fresh process, at two file counts, and is checked against two targets that pull opposite ways:

1. reads: every file is opened once to fill in its structure (index, browse, or both together), and
   once per value read.
2. RAM: nothing is materialized process-wide. Retained memory grows by metadata only per file,
   no whole file stays loaded after the work, and the transient peak stays within a couple of files
   regardless of how many files there are.

usage: benchmark_cache.py [--root DIR] [--sizes N1 N2] [--formats ...] [--workloads ...]
  --root  the directory containing the lazyfields package to measure (default: this checkout)
"""

import argparse
import ctypes
import gc
import json
import os
import pickle
import resource
import subprocess
import sys
import time
import tracemalloc
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "benchmark_cache_data"
MiB, KiB = 1024 * 1024, 1024

SIGNAL = 100_000  # floats in the large array; the template parses to about 4 MiB as Python lists
FORMATS = {"json": ".json", "pkl_list": ".pkl", "pkl_numpy": ".pkl"}
WORKLOADS = ("index", "browse", "index+browse", "read")
SIZES = (100, 500)

# Targets
READS_PER_FILE = 1  # opens of each file per workload
RETAINED_KIB_PER_FILE = 32  # memory kept per indexed file: metadata, never data
# Memory kept regardless of file count, in parsed files: no whole file stays loaded. The allocator alone
# keeps up to 2 here for list-heavy formats, holding memory freed after parsing.
KEPT_FILES = 2.5
TRANSIENT_FILES = 2.0  # peak above what is retained, in parsed files: one file at a time, plus parsing
MEMORY_LIMIT = 8 * 1024 * MiB  # address space of each run, so a leaking version fails instead of swapping


def template(fmt, size=SIGNAL):
    """One row file: scalars, a nested group, a list of mappings, and large arrays."""
    if fmt == "pkl_numpy":
        import numpy as np
        signal, trace = np.arange(size, dtype=float), np.arange(size // 3, dtype=float)
    else:
        signal, trace = [float(i) for i in range(size)], [float(i) for i in range(size // 3)]
    return {"index": 0, "label": "row", "x": 0.5, "signal": signal,
            "meta": {"a": 1, "unit": "s", "trace": trace}, "rows": [{"t": 0, "v": 1.5}, {"t": 1, "v": 2.5}]}


def _write(path, fmt, size=SIGNAL):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        value = template(fmt, size)
        path.write_bytes(json.dumps(value).encode() if fmt == "json" else pickle.dumps(value))
    return path


def prepare(data_dir, fmt, n):
    """A directory of n symlinks to the format's template; the template is written once.

    A tiny file of the same shape goes in a warmup directory beside it, for the imports and setup a
    first run does once, so they are not counted as retained.
    """
    suffix = FORMATS[fmt]
    source = _write(data_dir / "templates" / f"{fmt}{suffix}", fmt)
    _write(data_dir / "warmup" / fmt / f"w{suffix}", fmt, size=30)
    directory = data_dir / fmt / f"n{n}"
    if len(list(directory.glob(f"*{suffix}"))) != n:
        directory.mkdir(parents=True, exist_ok=True)
        for file in directory.iterdir():
            file.unlink()
        for i in range(n):
            (directory / f"r{i:05d}{suffix}").symlink_to(os.path.relpath(source, directory))
    return directory, source


def parsed_size(source, fmt):
    """Python heap of one fully parsed template file."""
    raw = source.read_bytes()
    tracemalloc.start()
    value = json.loads(raw) if fmt == "json" else pickle.loads(raw)
    size = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    del value
    return size


# Child process: one workload, one directory.

def _status(field):
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1]) * KiB
    raise RuntimeError(field)


def _rchar():
    return int(Path("/proc/self/io").read_text().split("rchar:")[1].split()[0])


def _settle():
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    return _status("VmRSS")


def _browse(fs, names):
    """What the browser asks on expanding a file and its groups: listings, then each child's full info."""
    for name in names:
        for child, kind, _ in fs.items(name):
            path = f"{name}/{child}"
            fs.info(path)
            if kind == "group":
                for grandchild, _, _ in fs.items(path):
                    fs.info(f"{path}/{grandchild}")


def run_case(root, directory, workload):
    sys.path.insert(0, str(root))
    import lazyfields
    from lazyfields.filesystem.filesystem import FileSystem

    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT, MEMORY_LIMIT))
    prefix = str(directory)
    opens = Counter()

    def audit(event, args):
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = os.fsdecode(args[0])
            if path.startswith(prefix):
                opens[Path(path).name] += 1

    def work(directory):
        """The workload's results, kept alive as a user would keep them."""
        table = fs = None
        if workload in ("index", "index+browse"):
            table = lazyfields.create_reference_table(str(directory))
        if workload in ("browse", "index+browse", "read"):
            handle = table is not None and table.attrs.get("filesystem")
            fs = handle.filesystem if hasattr(handle, "filesystem") else FileSystem.from_directory(directory, "*")
            names = fs.listdir()
            if workload == "read":
                for name in names:
                    fs[f"{name}/signal"]  # Read, then dropped.
            else:
                _browse(fs, names)
        return table, fs

    files = sorted(directory.iterdir())
    warmup = work(directory.parent.parent / "warmup" / directory.parent.name)  # noqa: F841  Kept, as results are.
    base = _settle()
    Path("/proc/self/clear_refs").write_text("5")  # The peak counts from here.
    sys.addaudithook(audit)
    read_before, start = _rchar(), time.perf_counter()
    try:
        results = work(directory)  # noqa: F841  Kept alive until retained memory is measured.
    except MemoryError:
        return {"error": f"MemoryError above {MEMORY_LIMIT // MiB} MiB"}
    seconds, read = time.perf_counter() - start, _rchar() - read_before
    peak = _status("VmHWM") - base
    retained = _settle() - base
    counts = [opens[file.name] for file in files]
    return {"n": len(files), "seconds": seconds, "opens_mean": sum(counts) / len(files), "opens_max": max(counts),
            "bytes_read": read, "data_bytes": sum(file.stat().st_size for file in files),
            "peak": peak, "retained": retained}


# Parent process: prepare, run each case in its own process, judge.

def run_subprocess(root, directory, workload):
    completed = subprocess.run([sys.executable, __file__, "--case", str(root), str(directory), workload],
                               capture_output=True, text=True, check=False)
    if completed.returncode:
        return {"error": completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "failed"}
    return json.loads(completed.stdout.strip().splitlines()[-1])


def judge(small, large, file_size):
    """Retained growth per file, retained floor, and transient peak, against the targets."""
    n1, n2 = small["n"], large["n"]
    per_file = (large["retained"] - small["retained"]) / (n2 - n1)
    kept = small["retained"] - per_file * n1
    transient = max(run["peak"] - run["retained"] for run in (small, large))
    failures = []  # Of the four checks: reads, growth per file, kept, peak.
    if large["opens_max"] > READS_PER_FILE:
        failures.append(f"reads {large['opens_max']}/file")
    if per_file > RETAINED_KIB_PER_FILE * KiB:
        failures.append(f"keeps {per_file / file_size:.2f} file/file")
    if kept > KEPT_FILES * file_size:
        failures.append(f"keeps {kept / file_size:.1f} files")
    if transient > TRANSIENT_FILES * file_size:
        failures.append(f"peak {transient / file_size:.1f} files")
    return per_file, kept, transient, failures


def main():
    if len(sys.argv) == 5 and sys.argv[1] == "--case":
        print(json.dumps(run_case(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])))
        return
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=ROOT_DIR.parent.parent)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--sizes", type=int, nargs=2, default=SIZES)
    parser.add_argument("--formats", nargs="+", choices=FORMATS, default=list(FORMATS))
    parser.add_argument("--workloads", nargs="+", choices=WORKLOADS, default=list(WORKLOADS))
    args = parser.parse_args()

    print(f"lazyfields from {args.root / 'lazyfields'}; files per run: {args.sizes[0]} and {args.sizes[1]}")
    print(f"targets: {READS_PER_FILE} read/file; retained <= {RETAINED_KIB_PER_FILE} KiB/file and "
          f"<= {KEPT_FILES:g} file kept; peak <= {TRANSIENT_FILES:g} files above retained\n")
    header = (f"{'format':10} {'workload':13} {'reads/file':>10} {'read x':>7} {'seconds':>8} "
              f"{'KiB/file':>9} {'kept':>6} {'peak':>6}  verdict")
    print(header + "\n" + "-" * len(header), flush=True)
    failed, checks, passed = False, 0, 0
    for fmt in args.formats:
        directories = [prepare(args.data, fmt, n) for n in args.sizes]
        file_size = parsed_size(directories[0][1], fmt)
        for workload in args.workloads:
            small, large = (run_subprocess(args.root, directory, workload) for directory, _ in directories)
            if "error" in small or "error" in large:
                print(f"{fmt:10} {workload:13} {(large.get('error') or small.get('error'))}", flush=True)
                failed, checks = True, checks + 4
                continue
            per_file, kept, transient, failures = judge(small, large, file_size)
            failed |= bool(failures)
            checks, passed = checks + 4, passed + 4 - len(failures)
            reads = f"{large['opens_mean']:.2f}/{large['opens_max']}"
            print(f"{fmt:10} {workload:13} {reads:>10} {large['bytes_read'] / large['data_bytes']:7.2f} "
                  f"{large['seconds']:8.2f} {per_file / KiB:9.1f} {kept / file_size:6.1f} "
                  f"{transient / file_size:6.1f}  {'ok' if not failures else 'FAIL: ' + ', '.join(failures)}",
                  flush=True)
    print("\nreads/file: mean/max opens of each file; read x: bytes read / bytes on disk; KiB/file: retained growth "
          "per file; kept: retained regardless of file count, and peak: transient above retained, both in parsed "
          "files (" + ", ".join(f"{fmt} {parsed_size(prepare(args.data, fmt, args.sizes[0])[1], fmt) / MiB:.1f} MiB"
                                 for fmt in args.formats) + ")")
    print(f"score: {passed}/{checks} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
