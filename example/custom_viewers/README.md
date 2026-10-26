# Custom viewers

A viewer shows any node it suits, wherever it appears: in the filesystem
browser, below a table, or as a method's result. Unlike a store method, it
knows nothing of rows or parents. Register one with
`register_viewer(name, accepts, view)`:

- `accepts(kind, attrs)` decides from the node's metadata alone, such as a
  dataset's shape, dtype, or loaded type, without reading it.
- `view(value, attrs)` gets the value and the same attrs. It either shows the
  value itself and returns `None`, as `grid.plot()` does in PyVista's window,
  or returns a matplotlib `Figure` to embed in the browser's window. A function
  that fills a Tk frame is also accepted.

The browser opens and titles its window, reports errors, and adds a button that
toggles to the value as text.

From the repository root, with Tk available:

```bash
pip install -e '.[plot]' pyvista
python3 example/custom_viewers/generate_dataset.py
python3 example/custom_viewers/browse.py
```

Both files are mounted together in one filesystem:

| Dataset path | Shape | Dtype | Default viewer |
| --- | --- | --- | --- |
| `volume.npz/voxels` | `(76, 507, 512)` | int32 | PyVista volume |
| `image.npz/rgb` | `(3, 512, 512)` | uint8 | Matplotlib RGB image |

Double-click a dataset to open its viewer; right-click any node for every
viewer that suits it, built-in ones such as Slices and Text included. The
browser waits while PyVista's window is open.

`browse.py` registers Image for numeric arrays with shape `(3, height, width)`
and Volume for other numeric 3D arrays. A three-slice volume therefore matches
the image convention in this example. The predicates use no filenames or
dataset-specific checks. The image viewer moves the channel axis for `imshow`;
the volume viewer interprets axes and `attrs["spacing"]` as `(z, y, x)`, converting
both to PyVista's `(x, y, z)` order. Arrays without spacing use unit voxels.
Values up to the median, typically background such as air, render transparent.

The generator writes `data/mount.json`, which mounts both NPZ files and attaches
the original CT spacing to `volume.npz/voxels`: approximately
`(2.528, 0.429688, 0.429688)` millimetres in `(z, y, x)` order. This preserves the
head's proportions without resampling or changing the generic viewer. These
attributes are also visible in the browser and reach viewers opened through
either double-click or the context menu.

`generate_dataset.py` keeps data preparation separate. The RGB image is the
standard Lena (woman in a hat) sample from [OpenCV](https://github.com/opencv/opencv/blob/4.12.0/samples/data/lena.jpg).
It downloads about 90 KB, verifies its SHA-256 checksum, and caches the JPEG in
`data/lena.jpg`. Pillow (installed with Matplotlib) decodes it at its original
512 by 512 resolution; only the channel axis is moved for storage.

The volume comes from [3D Slicer's CT-MR Brain sample](https://github.com/Slicer/Slicer/blob/main/Modules/Scripted/SampleData/SampleData.py),
donated for use without restrictions; see the [Slicer license](https://github.com/Slicer/Slicer/blob/main/License.txt).
The first run downloads about 16 MB, verifies the published SHA-256 checksum,
and caches the source in `data/CT-brain.nrrd`. Later runs work offline. The CT's
original shape, int32 dtype, and voxel values are preserved without resampling.
