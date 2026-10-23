# Simple filesystem

With the checkout installed, NumPy and h5py available, and Tk for the explorer:

```bash
python3 example/simple_filesystem/generate_dataset.py
python3 example/simple_filesystem/minimal_filesystem.py
```

The generator writes the same three rows, scalar values, array shapes, and strings
as the benchmarks. Each row has three numeric arrays and one string array, each
with 100 MiB of logical content. Generated files take several GiB in total;
running the generator again overwrites them.

```text
simple_filesystem/
├── h5/row_{00,01,02}.h5
├── json/row_{00,01,02}.json
└── pickle/row_{00,01,02}.pkl
```

The filesystem mounts each file at exactly this relative path. For example,
`h5/row_00.h5/array2d` accesses the first row's HDF5 matrix. The explorer is
shared with the existing filesystem example through
`lazyfields.filesystem.explorer`.
