# Large arrays

With the checkout installed, NumPy and h5py available, and Tk for the explorer:

```bash
python3 example/large_arrays/generate_dataset.py
python3 example/large_arrays/mount_by_folder.py
python3 example/large_arrays/mount_flat.py
```

The generator writes the same three rows, scalar values, array shapes, and strings
as the benchmarks. Each row has three numeric arrays and one string array, each
with 100 MiB of logical content. Generated files take several GiB in total;
running the generator again overwrites them.

```text
large_arrays/
├── h5/row_{00,01,02}.h5
├── json/row_{00,01,02}.json
└── pickle/row_{00,01,02}.pkl
```

`mount_by_folder.py` mounts each file at exactly this relative path, so
`h5/row_00.h5/array2d` accesses the first row's HDF5 matrix. `mount_flat.py`
mounts only the HDF5 and pickle rows directly at the root, such as `row_00.h5`
and `row_00.pkl`. The explorer is shared with the composed_filesystem example
through `lazyfields.filesystem.explorer`.

The first run of each script saves its mount next to it (`mount_by_folder.json`,
`mount_flat.json`) and later runs restore it from there; pass `--rebuild` to
rediscover and save again.
