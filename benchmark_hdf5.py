#!/usr/bin/env python3
import gc
import json
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR.parent))

import lazyfields as lf


MiB = 1024 * 1024
ROW_COUNT = 3
TARGET_BYTES = 100 * MiB
SLICE_BYTES = 25 * MiB
SMALL_BYTES = MiB
SMALL_ELEMENT_COUNT = SMALL_BYTES // np.dtype(np.float64).itemsize
ALL_ROWS_BYTES = ROW_COUNT * TARGET_BYTES
DATA_DIR = ROOT_DIR / "benchmark_ram_data"
REFERENCE_TABLE_PATH = ROOT_DIR / "benchmark_ram_reference.pkl"

FLOAT_COUNT = 13_107_200
OBJECT_ITEM_BYTES = 32 * 1024
OBJECT_COUNT = 3_200
OBJECT_TEMPLATE = "x" * OBJECT_ITEM_BYTES
STRING_DTYPE = h5py.string_dtype(encoding="utf-8")

ARRAY1D_SHAPE = (13_107_200,)
ARRAY2D_SHAPE = (3_200, 4_096)
ARRAY3D_SHAPE = (200, 256, 256)


def current_peak_rss_bytes() -> int:
    for line in (Path("/proc/self/status").read_text()).splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Could not read VmHWM from /proc/self/status.")

def prepare_data() -> None:
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    base = np.arange(FLOAT_COUNT, dtype=np.float64)
    arrays = {
        "array1d": base,
        "array2d": base.reshape(ARRAY2D_SHAPE),
        "array3d": base.reshape(ARRAY3D_SHAPE),
        "array1d_objects": np.full(OBJECT_COUNT, OBJECT_TEMPLATE, dtype=object),
    }
    for file_index in range(ROW_COUNT):
        with h5py.File(DATA_DIR / f"row_{file_index:02d}.h5", "w") as fh:
            for key, value in {
                "x_int": 100 + file_index,
                "x_float": 10.5 + file_index / 10.0,
                "x_bool": bool(file_index % 2),
                "x_str": f"row-{file_index:02d}",
                "variant_id": file_index,
            }.items():
                fh.create_dataset(key, data=value)
            for field, array in arrays.items():
                if field == "array1d_objects":
                    dataset = fh.create_dataset(field, data=array, dtype=STRING_DTYPE)
                else:
                    dataset = fh.create_dataset(field, data=array)
                if field == "array1d_objects":
                    # Make each file's first object-string unique while keeping the fixed 32 KiB content size.
                    prefix = f"file-{file_index:02d}:"
                    dataset[0] = prefix + ("y" * (OBJECT_ITEM_BYTES - len(prefix)))

    gc.collect()
    lf.create_reference_table(DATA_DIR, file_pattern="*.h5").to_pickle(REFERENCE_TABLE_PATH)


def measure_case(case_name: str, expected_bytes: int, loader) -> dict[str, object]:
    gc.collect()
    rss_before = current_peak_rss_bytes()
    loaded = loader()
    rss_after = current_peak_rss_bytes()
    del loaded
    gc.collect()
    return {
        "case": case_name,
        "expected_bytes": expected_bytes,
        "peak_rss_delta_bytes": max(0, rss_after - rss_before),
    }


def run_case(case_name: str) -> dict[str, object]:
    gc.collect()
    reference_df = pd.read_pickle(REFERENCE_TABLE_PATH)
    row = reference_df.iloc[0]

    ##########################
    # Access patterns under test
    ##########################
    if case_name == "row-full-array1d":
        return measure_case(case_name, TARGET_BYTES, lambda: row.store["array1d"])
    if case_name == "row-small-array1d":
        return measure_case(case_name, SMALL_BYTES, lambda: row.store["array1d", :SMALL_ELEMENT_COUNT])
    if case_name == "row-slice-array1d":
        return measure_case(case_name, SLICE_BYTES, lambda: row.store["array1d", :3_276_800])
    if case_name == "row-full-array2d":
        return measure_case(case_name, TARGET_BYTES, lambda: row.store["array2d"])
    if case_name == "row-slice-array2d":
        return measure_case(case_name, SLICE_BYTES, lambda: row.store["array2d", (slice(None, 800), slice(None))])
    if case_name == "row-full-array3d":
        return measure_case(case_name, TARGET_BYTES, lambda: row.store["array3d"])
    if case_name == "row-slice-array3d":
        return measure_case(case_name, SLICE_BYTES, lambda: row.store["array3d", (slice(None, 50), slice(None), slice(None))])
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
    ##########################

    raise ValueError(f"Unknown case: {case_name}")

def run_case_subprocess(case_name: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--case", case_name],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def print_results(results: list[dict[str, object]]) -> None:
    print(f"Data directory: {DATA_DIR}")
    print("Each non-scalar dataset is 100.00 MiB.")
    print("Single-row hyperslabs are expected to load 25.00 MiB.")
    print(f"Small array accesses are expected to load {SMALL_BYTES} bytes, about 1 MiB.")
    print()
    print(f"{'case':28} {'expected MiB':>12} {'peak MiB':>10}")
    print("-" * 54)
    for result in results:
        expected = int(result["expected_bytes"])
        rss_delta = int(result["peak_rss_delta_bytes"])
        print(f"{result['case']:28} {expected / MiB:9.2f} {rss_delta / MiB:9.2f}")


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        print(json.dumps(run_case(sys.argv[2])))
        return

    prepare_data()
    results = [
        run_case_subprocess("row-full-array1d"),
        run_case_subprocess("row-small-array1d"),
        run_case_subprocess("row-slice-array1d"),
        run_case_subprocess("row-full-array2d"),
        run_case_subprocess("row-slice-array2d"),
        run_case_subprocess("row-full-array3d"),
        run_case_subprocess("row-slice-array3d"),
        run_case_subprocess("row-full-array1d_objects"),
        run_case_subprocess("row-slice-array1d_objects"),
        run_case_subprocess("frame-one-row-array1d"),
        run_case_subprocess("frame-one-row-small-array1d"),
        run_case_subprocess("frame-one-row-array2d"),
        run_case_subprocess("frame-all-rows-array1d"),
        run_case_subprocess("frame-all-rows-array2d"),
    ]
    print_results(results)


if __name__ == "__main__":
    main()
