#!/usr/bin/env python3
"""Speed, reads and RAM of indexing and reading, over the data layouts lazyfields is used with.

Layouts follow the projects that use lazyfields, with synthetic data of the same shape:
  many_json      many mid-size JSON results (config, input, output, metadata), one row each
  many_pickle    many pickled results holding numpy arrays, one row each
  large_json     few large JSON results with wide groups of many datasets, one row each
  one_file_rows  one large JSON file whose top-level groups are the rows
  run_dirs       a directory per run, config.json + input.json + output.json forming one row

Workloads follow how those projects use it:
  index          create_reference_table, no pipe
  index_pipe     create_reference_table with a pipe reading a group, a nested scalar and a whole subtree
  update         update_reference_table over an unchanged index
  browse         listing every row and each child's full info, as the browser does
  store_read     reference_df.store[<array>] for every row

Each run is a fresh process; files are symlinks to one template, so many files cost one on disk.
--save writes the results as a baseline; --compare checks a run against one and fails on any
degradation: more reads per file, or time or RAM above the tolerances below.

usage: benchmark_layouts.py [--root DIR] [--scale X] [--layouts ...] [--workloads ...]
                            [--save FILE] [--compare FILE]
"""

import argparse
import json
import os
import pickle
import resource
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from benchmark_cache import MEMORY_LIMIT, _rchar, _settle, _status

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "benchmark_layouts_data"
MiB = 1024 * 1024

# Layout: rows at scale 1, run shape (points per array, monomers, molecules, an extra long array)
LAYOUTS = {
    "many_json": dict(rows=600, points=450, monomers=3, molecules=2, extra=45_000),
    "many_pickle": dict(rows=600, points=450, monomers=3, molecules=2, extra=45_000),
    "large_json": dict(rows=6, points=24_000, monomers=64, molecules=64, extra=0),
    "one_file_rows": dict(rows=400, points=450, monomers=3, molecules=2, extra=0),
    "run_dirs": dict(rows=300, points=450, monomers=3, molecules=2, extra=45_000),
}
WORKLOADS = ("index", "index_pipe", "update", "browse", "store_read")

# Tolerances for --compare: a run degrades when it exceeds the baseline by more than these.
TIME_RATIO, TIME_SLACK = 1.3, 0.5  # seconds
RAM_RATIO, RAM_SLACK = 1.15, 16 * MiB


def simulation(points, monomers, molecules, extra, seed=0):
    """One result shaped like the SCF outputs these projects index."""
    def array(n, offset=0.0):
        return [round(offset + i / n, 6) for i in range(n)]

    monomer_names = [f"M{i}" for i in range(monomers)]
    output = {
        "x": array(points), "y": array(points, 1.0),
        "mon": {name: {"phi": array(points, i), "u": array(points, -i)} for i, name in enumerate(monomer_names)},
        "mol": {f"pol{j}": {"phi": array(points, j), "n": 10.0 + j, "phibulk": 0.01} for j in range(molecules)},
        "sys": {"NN": {"free_energy": -1.5 - seed, "q": array(points)}},
    }
    if extra:
        output["mol"]["pol0"]["phi_ranked"] = array(extra)
    return {
        "config": {f"param_{i}": float(i) + seed for i in range(16)} | {"name": f"run{seed}", "geometry": "cylindrical"},
        "metadata": {"result_id": f"id{seed}", "labels": ["A", "B"], "polymer_name": "pol0",
                     "source": "def step_fraction(x):\n    return x\n"} | {f"meta_{i}": i for i in range(20)},
        "input": {"problems": [{"mon": {name: {"chi_S": 0.5, "freedom": "free"} for name in monomer_names},
                                "mol": {f"pol{j}": {"composition": "(A)10(B)10"} for j in range(molecules)}}]},
        "output": {"problems": [output]},
    }


def _as_numpy(value):
    """Lists of floats as numpy arrays, as a pickled result holds them."""
    import numpy as np
    if isinstance(value, dict):
        return {key: _as_numpy(item) for key, item in value.items()}
    if isinstance(value, list) and value and isinstance(value[0], float):
        return np.asarray(value)
    return [_as_numpy(item) for item in value] if isinstance(value, list) else value


def write_layout(directory, layout, rows, shape):
    """The layout's files under directory, symlinking one template per kind of file."""
    directory.mkdir(parents=True, exist_ok=True)
    templates = directory.parent / f"{directory.name}.templates"
    templates.mkdir(exist_ok=True)
    result = simulation(**shape)

    def template(name, value, as_pickle=False):
        path = templates / name
        if not path.exists():
            path.write_bytes(pickle.dumps(_as_numpy(value)) if as_pickle else json.dumps(value).encode())
        return path

    def link(source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.symlink_to(os.path.relpath(source, target.parent))

    if layout == "one_file_rows":
        path = directory / "results.json"
        if not path.exists():
            path.write_text(json.dumps({f"run_{i:05d}": simulation(**shape, seed=i) for i in range(rows)}))
    elif layout == "run_dirs":
        sources = {part: template(f"{part}.json", result[part]) for part in ("config", "input", "output")}
        for i in range(rows):
            for part, source in sources.items():
                link(source, directory / f"run_{i:05d}" / f"{part}.json")
    else:
        source = template("result.pkl" if layout == "many_pickle" else "result.json", result,
                          as_pickle=layout == "many_pickle")
        for i in range(rows):
            link(source, directory / f"r{i:05d}{source.suffix}")
    return directory


def prepare(data_dir, layout, scale):
    spec = LAYOUTS[layout]
    rows = max(2, int(spec["rows"] * scale))
    shape = {key: spec[key] for key in ("points", "monomers", "molecules", "extra")}
    small = dict(points=30, monomers=2, molecules=1, extra=0)
    return (write_layout(data_dir / f"{layout}_{rows}", layout, rows, shape),
            write_layout(data_dir / f"{layout}_warmup", layout, 2, small))


def source_of(layout, directory):
    """What create_reference_table indexes, and the prefix of a row's results inside a row."""
    if layout == "one_file_rows":
        return {"path": "results.json"}, str(directory), ""
    if layout == "run_dirs":
        return {"mount": [{"glob": "*/*.json"}]}, str(directory), ".json"
    return str(directory), None, ""


# Child process: one layout, one workload.

def run_case(root, layout, directory, warmup, workload):
    sys.path.insert(0, str(root))
    import inspect
    import lazyfields

    # A version indexing only a directory's files has no specifications and no browser.
    directory_only = "directory" in inspect.signature(lazyfields.create_reference_table).parameters
    if directory_only and (layout in ("one_file_rows", "run_dirs") or workload == "browse"):
        return {"skipped": "not in this version"}
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT, MEMORY_LIMIT))

    def table(directory, **kwargs):
        if directory_only:
            return lazyfields.create_reference_table(str(directory), **kwargs)
        source, relative_to, _ = source_of(layout, directory)
        options = {"relative_to": relative_to} if relative_to else {}
        return lazyfields.create_reference_table(source, **options, **kwargs)

    suffix = source_of(layout, directory)[2]
    config, output = f"config{suffix}", f"output{suffix}"

    def pipe(row):
        for key, value in row[config].items():  # A whole small group, as charged_pore's extract_config.
            row[key] = value
        row["free_energy"] = row[f"{output}/problems/0/sys/NN/free_energy"]  # One nested scalar.
        row["n"] = row[output]["problems"][0]["mol"]["pol0"]["n"]  # A whole subtree, as diblock_micelle's.

    def work(directory, measured):
        """The workload; what it keeps stays alive until memory is measured. Reads count from measured()."""
        if workload == "index":
            measured()
            return table(directory)
        if workload == "index_pipe":
            measured()
            return table(directory, pipe=[pipe])
        reference = table(directory)
        if workload == "update":
            measured()
            return lazyfields.update_reference_table(reference, *([str(directory)] if directory_only else []))
        measured()
        if workload == "browse":
            fs = reference.attrs["filesystem"].filesystem
            for path in reference["lf_path"]:
                for child, kind, _ in fs.items(path):
                    fs.info(f"{path}/{child}")
                    if kind == "group":
                        for grandchild, _, _ in fs.items(f"{path}/{child}"):
                            fs.info(f"{path}/{child}/{grandchild}")
        else:
            reference.store[f"{output}/problems/0/x"]
        return reference

    work(warmup, lambda: None)  # Imports and first-use setup, not counted.
    prefix, opens = str(directory), Counter()

    def audit(event, args):
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = os.fsdecode(args[0])
            if path.startswith(prefix):
                opens[path] += 1

    state = {}

    def measured():
        state["base"] = _settle()
        Path("/proc/self/clear_refs").write_text("5")
        sys.addaudithook(audit)
        state["read"], state["start"] = _rchar(), time.perf_counter()

    try:
        kept = work(directory, measured)  # noqa: F841  Kept alive until retained memory is measured.
    except MemoryError:
        return {"error": f"MemoryError above {MEMORY_LIMIT // MiB} MiB"}
    seconds, read = time.perf_counter() - state["start"], _rchar() - state["read"]
    peak = _status("VmHWM") - state["base"]
    retained = _settle() - state["base"]
    files = [path for path in directory.rglob("*") if path.is_file()]
    return {"files": len(files), "rows": len(kept) if hasattr(kept, "__len__") else None, "seconds": seconds,
            "reads_mean": sum(opens.values()) / len(files), "reads_max": max(opens.values(), default=0),
            "bytes_read": read / sum(path.stat().st_size for path in files),
            "peak": peak, "retained": retained}


# Parent process.

def run_subprocess(root, layout, directory, warmup, workload):
    completed = subprocess.run([sys.executable, __file__, "--case", str(root), layout, str(directory), str(warmup),
                                workload], capture_output=True, text=True, check=False, cwd=ROOT_DIR)
    if completed.returncode:
        return {"error": (completed.stderr.strip().splitlines() or ["failed"])[-1]}
    return json.loads(completed.stdout.strip().splitlines()[-1])


def degradations(result, base):
    """How result is worse than base, beyond the tolerances."""
    found = []
    if result["reads_max"] > base["reads_max"]:
        found.append(f"reads {base['reads_max']}->{result['reads_max']}/file")
    if result["seconds"] > base["seconds"] * TIME_RATIO + TIME_SLACK:
        found.append(f"time {base['seconds']:.1f}->{result['seconds']:.1f}s")
    for name in ("peak", "retained"):
        if result[name] > base[name] * RAM_RATIO + RAM_SLACK:
            found.append(f"{name} {base[name] / MiB:.0f}->{result[name] / MiB:.0f} MiB")
    return found


def main():
    if len(sys.argv) == 7 and sys.argv[1] == "--case":
        _, _, root, layout, directory, warmup, workload = sys.argv
        print(json.dumps(run_case(Path(root), layout, Path(directory), Path(warmup), workload)))
        return
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=ROOT_DIR.parent.parent)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--scale", type=float, default=1.0, help="multiplies the number of rows of each layout")
    parser.add_argument("--layouts", nargs="+", choices=LAYOUTS, default=list(LAYOUTS))
    parser.add_argument("--workloads", nargs="+", choices=WORKLOADS, default=list(WORKLOADS))
    parser.add_argument("--save", type=Path, help="write the results as a baseline")
    parser.add_argument("--compare", type=Path, help="fail on degradation against a saved baseline")
    args = parser.parse_args()

    baseline = json.loads(args.compare.read_text()) if args.compare else {}
    print(f"lazyfields from {args.root / 'lazyfields'}; scale {args.scale:g}"
          + (f"; compared with {args.compare}" if args.compare else ""))
    header = (f"{'layout':14} {'workload':11} {'files':>6} {'rows':>5} {'reads/file':>10} {'read x':>7} "
              f"{'seconds':>8} {'peak MiB':>9} {'kept MiB':>9}  verdict")
    print(header + "\n" + "-" * len(header), flush=True)
    results, failed = {}, False
    for layout in args.layouts:
        directory, warmup = prepare(args.data, layout, args.scale)
        for workload in args.workloads:
            name = f"{layout}/{workload}"
            result = results[name] = run_subprocess(args.root, layout, directory, warmup, workload)
            if "skipped" in result:
                print(f"{layout:14} {workload:11} n/a: {result['skipped']}", flush=True)
                continue
            if "error" in result:
                print(f"{layout:14} {workload:11} {result['error']}", flush=True)
                failed = True
                continue
            base = baseline.get(name)
            base = base if base and "error" not in base and "skipped" not in base else None
            found = degradations(result, base) if base else []
            failed |= bool(found)
            verdict = "DEGRADED: " + ", ".join(found) if found else ("ok" if base else "-")
            reads = f"{result['reads_mean']:.2f}/{result['reads_max']}"
            print(f"{layout:14} {workload:11} {result['files']:6} {result['rows'] or '':>5} {reads:>10} "
                  f"{result['bytes_read']:7.2f} {result['seconds']:8.2f} {result['peak'] / MiB:9.1f} "
                  f"{result['retained'] / MiB:9.1f}  {verdict}", flush=True)
    print("\nreads/file: mean/max opens of each file; read x: bytes read / bytes on disk; peak: RSS growth during "
          "the workload; kept: RSS growth still held after it, results included")
    if args.save:
        args.save.write_text(json.dumps(results, indent=1))
        print(f"saved {args.save}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
