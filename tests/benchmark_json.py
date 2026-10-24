#!/usr/bin/env python3
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR.parent.parent))

import lazyfields as lf
from lazyfields.example.large_arrays.generate_dataset import generate_dataset

MiB = 1024 * 1024
ROW_COUNT = 3
TARGET_BYTES = 100 * MiB
SLICE_BYTES = 25 * MiB
SMALL_BYTES = MiB
SMALL_ELEMENT_COUNT = SMALL_BYTES // np.dtype(np.float64).itemsize
ALL_ROWS_BYTES = ROW_COUNT * TARGET_BYTES
DATA_DIR = ROOT_DIR / "benchmark_json_data"
REFERENCE_TABLE_PATH = ROOT_DIR / "benchmark_json_reference.pkl"

def current_peak_rss_bytes() -> int:
    for line in (Path("/proc/self/status").read_text()).splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Could not read VmHWM from /proc/self/status.")


def prepare_data() -> None:
    generate_dataset(DATA_DIR, "json", overwrite=False)


def build_reference() -> pd.DataFrame:
    table = lf.create_reference_table(
        {"paths": sorted(path.name for path in DATA_DIR.glob("*.json"))}, relative_to=DATA_DIR.resolve(),
    )
    table.to_pickle(REFERENCE_TABLE_PATH)
    return table


def measure_case(case_name: str, expected_bytes: int, loader) -> dict[str, object]:
    gc.collect()
    rss_before = current_peak_rss_bytes()
    start = time.perf_counter()
    loaded = loader()
    seconds = time.perf_counter() - start
    rss_after = current_peak_rss_bytes()
    del loaded
    gc.collect()
    return {
        "case": case_name,
        "expected_bytes": expected_bytes,
        "seconds": seconds,
        "peak_rss_delta_bytes": max(0, rss_after - rss_before),
    }


def run_case(case_name: str) -> dict[str, object]:
    gc.collect()
    if case_name == "reference-inspect":
        return measure_case(case_name, 0, build_reference)
    reference_df = pd.read_pickle(REFERENCE_TABLE_PATH)
    row = reference_df.iloc[0]

    if case_name == "row-scalar":
        return measure_case(case_name, 0, lambda: row.store["x_int"])
    if case_name == "row-full-array1d":
        return measure_case(case_name, TARGET_BYTES, lambda: row.store["array1d"])
    if case_name == "row-small-array1d":
        return measure_case(case_name, SMALL_BYTES, lambda: row.store["array1d", :SMALL_ELEMENT_COUNT])
    if case_name == "row-slice-array1d":
        return measure_case(case_name, SLICE_BYTES, lambda: row.store["array1d", :3_276_800])
    if case_name == "row-full-array2d":
        return measure_case(case_name, TARGET_BYTES, lambda: row.store["array2d"])
    if case_name == "row-slice-array2d":
        return measure_case(case_name, SLICE_BYTES, lambda: row.store["array2d", :800, :])
    if case_name == "row-full-array3d":
        return measure_case(case_name, TARGET_BYTES, lambda: row.store["array3d"])
    if case_name == "row-slice-array3d":
        return measure_case(case_name, SLICE_BYTES, lambda: row.store["array3d", :50, :, :])
    if case_name == "row-full-array1d_objects":
        return measure_case(case_name, TARGET_BYTES, lambda: row.store["array1d_objects"])
    if case_name == "row-slice-array1d_objects":
        return measure_case(case_name, SLICE_BYTES, lambda: row.store["array1d_objects", :800])
    if case_name == "frame-one-row-array1d":
        return measure_case(case_name, TARGET_BYTES, lambda: reference_df.iloc[0:1].store["array1d"])
    if case_name == "frame-one-row-small-array1d":
        return measure_case(case_name, SMALL_BYTES, lambda: reference_df.iloc[0:1].store["array1d", :SMALL_ELEMENT_COUNT])
    if case_name == "frame-one-row-array2d":
        return measure_case(case_name, TARGET_BYTES, lambda: reference_df.iloc[0:1].store["array2d"])
    if case_name == "frame-all-rows-array1d":
        return measure_case(case_name, ALL_ROWS_BYTES, lambda: reference_df.store["array1d"])
    if case_name == "frame-all-rows-array2d":
        return measure_case(case_name, ALL_ROWS_BYTES, lambda: reference_df.store["array2d"])
    raise ValueError(f"Unknown case: {case_name}")


def run_case_subprocess(case_name: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--case", case_name],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def print_results(cases: tuple[str, ...]) -> None:
    print(f"Data directory: {DATA_DIR}")
    print("Each row file stores the same 100.00 MiB logical arrays as the HDF5 and pickle benchmarks.")
    print("JSON file byte sizes differ from the logical array sizes shown below.")
    print(f"Small array accesses are expected to load {SMALL_BYTES} bytes, about 1 MiB.")
    print()
    print(f"{'case':28} {'expected MiB':>12} {'seconds':>10} {'peak MiB':>10}")
    print("-" * 65, flush=True)
    for case in cases:
        result = run_case_subprocess(case)
        expected = int(result["expected_bytes"])
        rss_delta = int(result["peak_rss_delta_bytes"])
        print(f"{result['case']:28} {expected / MiB:9.2f} {result['seconds']:10.3f} {rss_delta / MiB:9.2f}", flush=True)


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        print(json.dumps(run_case(sys.argv[2])))
        return

    prepare_data()
    cases = (
        "reference-inspect", "row-scalar",
        "row-full-array1d", "row-small-array1d", "row-slice-array1d",
        "row-full-array2d", "row-slice-array2d", "row-full-array3d",
        "row-slice-array3d", "row-full-array1d_objects",
        "row-slice-array1d_objects", "frame-one-row-array1d",
        "frame-one-row-small-array1d", "frame-one-row-array2d",
        "frame-all-rows-array1d", "frame-all-rows-array2d",
    )
    print_results(cases)


if __name__ == "__main__":
    main()
