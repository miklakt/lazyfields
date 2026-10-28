# Large arrays

With the checkout installed, NumPy and h5py available, and Tk for the browser:

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
and `row_00.pkl`. Both open the filesystem browser shared with the composed_filesystem example
through `lazyfields.browser`.

The first run of each script saves its mount next to it (`mount_by_folder.json`,
`mount_flat.json`) and later runs restore it from there; pass `--rebuild` to
rediscover and save again.

## Browse in the OS file manager

With `pip install lazyfields[fuse]` and FUSE available (libfuse on Linux, macFUSE
or FUSE-T on macOS, WinFsp on Windows), mount a specification file or inline JSON:

```bash
lf-mount example/large_arrays/mount_by_folder.json ~/lazyfields_mnt
lf-mount example/composed_filesystem/resolve_and_merge.json ~/lazyfields_mnt \
    --relative-to example/composed_filesystem/data
lf-mount '{"mount": [{"glob": "h5/*"}]}' ~/lazyfields_mnt --relative-to example/large_arrays
```

`python3 -m lazyfields.filesystem.os_mount` accepts the same arguments.
Specification paths are resolved against `--relative-to`, by default the
specification file's directory, or the working directory for inline JSON. The mount ends with the process: Ctrl+C, closing
the terminal, or kill all release the mount point.

The mount is read-only. Groups are directories and scalars are small text files.
Datasets appear under their own names with raw contents, loaded only when opened:
buffer bytes for numeric arrays (`np.fromfile(path, dtype)`), text otherwise.

Presenters add further files beside a leaf without reading anything while listing:

```python
from lazyfields.filesystem.os_mount import mount, preview

def npy(filesystem, path, kind):
    if kind == "dataset":
        import io, numpy as np
        def render():
            buffer = io.BytesIO()
            np.save(buffer, filesystem.read(path))
            return buffer.getvalue()
        return ".npy", render

mount(fs, mountpoint, presenters=[preview, npy])
```

`lf-mount` injects none. The optional `preview` presenter adds
`<name>.preview.txt` with the first rows, read through `selection`, so HDF5
previews never load whole arrays. Pickle and JSON readers must load a whole file to read any part of it, so their previews
are skipped for sources above 50 MiB, and listing inside such a file loads it once.
