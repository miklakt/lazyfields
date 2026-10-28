# Custom viewers

A viewer shows any node it suits, wherever it appears: in the filesystem
browser, below a table, or as a method's result. Unlike a store method, it
knows nothing of rows or parents. Register one with
`register_viewer(name, accepts, view)`:

- `accepts(node)` decides without reading the node's value, from `node.kind`,
  a dataset's `node.shape`, `node.dtype`, or loaded `node.type`, its
  `node.attrs`, and `node.children`, the names of its immediate children.
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

The generated files are mounted together in one filesystem:

| Dataset path | Shape | Dtype | Default viewer |
| --- | --- | --- | --- |
| `volume.npz/voxels` | `(76, 507, 512)` | int32 | PyVista volume |
| `image.npz/rgb` | `(3, 400, 600)` | uint8 | Matplotlib RGB image |

`1UBQ.pdb`, ubiquitin with 660 atoms, and `4HHB.pdb`, human deoxyhaemoglobin
with 4,779 atoms, are read by the PDB reader `browse.py` registers. Each is a
group of header fields, such as `title`, one dataset per atom column, such as
`x`, `y`, `z`, and `element`, and `bonds`, the pairs of atom serials its
`CONECT` records list; 1UBQ has none, 4HHB those of its four haem groups.

Double-click a dataset to open its viewer; right-click any node for every
viewer that suits it, built-in ones such as Slices and Text included. The
browser waits while PyVista's window is open.

`browse.py` registers Image for numeric arrays with shape `(3, height, width)`
and Volume for other numeric 3D arrays. A three-slice volume therefore matches
the image convention in this example. The image viewer moves the channel axis
for `imshow`; the volume viewer interprets axes and `attrs["spacing"]` as
`(z, y, x)`, converting both to PyVista's `(x, y, z)` order. Arrays without
spacing use unit voxels. Values up to the median, typically background such as
air, render transparent.

It registers Molecule for any group with `x`, `y`, `z`, and `element` children,
so it also recognizes those columns pickled or stored in HDF5: right-click a PDB
file and choose Open with Molecule, and drag to rotate. Atoms are drawn as balls
and sticks in CPK colours, sized by their van der Waals radii. Bonds join atoms
closer than their covalent radii and 0.4 Å together, as most molecular viewers
infer them, together with those a `bonds` dataset gives by atom serial, such as
a PDB file's `CONECT` records. None of the predicates use filenames or
dataset-specific checks.

The generator writes `data/mount.json`, which mounts all four files and attaches
the original CT spacing to `volume.npz/voxels`: approximately
`(2.528, 0.429688, 0.429688)` millimetres in `(z, y, x)` order. This preserves the
head's proportions without resampling or changing the generic viewer. These
attributes are also visible in the browser and reach viewers opened through
either double-click or the context menu.

`generate_dataset.py` keeps data preparation separate. The RGB image is the
coffee cup sample from [scikit-image](https://github.com/scikit-image/scikit-image/blob/v0.24.0/skimage/data/coffee.png),
released under CC0 by the photographer, Rachel Michetti. It downloads about
470 KB, verifies its SHA-256 checksum, and caches the PNG in `data/coffee.png`.
Pillow (installed with Matplotlib) decodes it at its original 600 by 400
resolution; only the channel axis is moved for storage.

The volume comes from [3D Slicer's CT-MR Brain sample](https://github.com/Slicer/Slicer/blob/main/Modules/Scripted/SampleData/SampleData.py),
donated for use without restrictions; see the [Slicer license](https://github.com/Slicer/Slicer/blob/main/License.txt).
The first run downloads about 16 MB, verifies the published SHA-256 checksum,
and caches the source in `data/CT-brain.nrrd`. Later runs work offline. The CT's
original shape, int32 dtype, and voxel values are preserved without resampling.

The proteins are [1UBQ](https://www.rcsb.org/structure/1UBQ) and
[4HHB](https://www.rcsb.org/structure/4HHB) from the RCSB Protein Data Bank,
whose archive data is CC0. They download about 80 KB and 470 KB with pinned
SHA-256 checksums. RCSB occasionally re-issues entries with updated headers; a
mismatch after such an update needs the checksum refreshed.
