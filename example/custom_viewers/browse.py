"""Inject volume, RGB image and molecule viewers into one mounted filesystem browser, with NPZ and PDB readers."""

import json
from pathlib import Path

import numpy as np

from lazyfields.browser import browse_filesystem, register_viewer
from lazyfields.filesystem import FileSystem
from lazyfields.readers import register_reader
from lazyfields.readers.npz import NUMPY_ARCHIVE
from lazyfields.readers.pdb import PROTEIN_DATA_BANK


DATA_DIR = Path(__file__).resolve().parent / "data"


def is_numeric_3d(node):
    """Numeric 3D datasets, by their shape, which only datasets have."""
    shape = node.shape or ()
    return len(shape) == 3 and all(shape) and np.dtype(node.dtype).kind in "biuf"


def is_image(node):
    """Recognize channel-first RGB arrays using metadata alone."""
    return is_numeric_3d(node) and node.shape[0] == 3


def is_volume(node):
    """Offer volume rendering for numeric 3D arrays other than RGB images."""
    return is_numeric_3d(node) and not is_image(node)


def show_image(value, attrs):
    """Matplotlib expects RGB channels last; the stored array stays unchanged."""
    from matplotlib.figure import Figure

    figure = Figure(layout="constrained")
    axes = figure.add_subplot()
    axes.imshow(np.moveaxis(np.asarray(value), 0, -1))
    axes.set_axis_off()
    return figure


def show_volume(value, attrs):
    """Render the volume in PyVista's window, with voxel spacing from attrs; unit voxels without it."""
    import pyvista as pv

    array = np.asarray(value)
    # Interpret array axes as z, y, x; the grid's dimensions are x, y, z.
    grid = pv.ImageData(dimensions=array.shape[::-1], spacing=attrs.get("spacing", (1, 1, 1))[::-1])
    grid.point_data["value"] = array.ravel()
    # Values up to the median, typically background such as air, stay clear.
    grid.plot(volume=True, cmap="viridis", opacity="sigmoid", clim=np.percentile(array, (50, 99.9)))


def is_molecule(node):
    """Recognize atoms by their children: a PDB file, or the same columns pickled or in HDF5."""
    return {"x", "y", "z", "element"} <= set(node.children)


def show_molecule(value, attrs):
    """Atoms in CPK colours and van der Waals sizes, with bonds from distances and any given; drag to rotate."""
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    # CPK colour, covalent radius and van der Waals radius in ångström, by element; others pink, sized as carbon.
    elements = {"H": ("white", 0.31, 1.20), "C": ("dimgrey", 0.76, 1.70), "N": ("blue", 0.71, 1.55),
                "O": ("red", 0.66, 1.52), "P": ("orange", 1.07, 1.80), "S": ("gold", 1.05, 1.80),
                "FE": ("darkorange", 1.32, 2.00)}
    xyz = np.column_stack([value["x"], value["y"], value["z"]]).astype(float)
    colours, covalent, van_der_waals = zip(*(elements.get((element or "").upper(), ("pink", 0.76, 1.70))
                                             for element in value["element"]))
    covalent, blocks = np.array(covalent), []
    # Infer bonds within the summed covalent radii plus 0.4 Å, processing distances in blocks.
    for start in range(0, len(xyz), 512):
        rows = np.arange(start, min(start + 512, len(xyz)))
        squared = (xyz[rows] ** 2).sum(1)[:, None] + (xyz ** 2).sum(1) - 2 * xyz[rows] @ xyz.T
        close = (squared < (covalent[rows, None] + covalent + 0.4) ** 2) & (rows[:, None] < np.arange(len(xyz)))
        first, second = np.nonzero(close)
        blocks.append(np.column_stack([rows[first], second]))
    if "bonds" in value and "atom_serial" in value:  # Given ones too, such as a PDB file's CONECT records.
        position = {serial: index for index, serial in enumerate(value["atom_serial"])}
        given = [(position[a], position[b]) for a, b in value["bonds"] if a in position and b in position]
        blocks.append(np.array(given, dtype=int).reshape(-1, 2))
    pairs = np.unique(np.sort(np.vstack(blocks), axis=1), axis=0)
    figure = Figure(layout="constrained")
    axes = figure.add_subplot(projection="3d")
    axes.add_collection3d(Line3DCollection(xyz[pairs], colors="dimgrey", linewidths=1.2))
    # Balls and sticks: areas grow with the radius squared; matplotlib sizes markers in points, so only ratios hold.
    axes.scatter(*xyz.T, s=(1.5 * np.array(van_der_waals)) ** 2, c=colours, depthshade=False)
    axes.set_aspect("equal")
    return figure


def main():
    if not (DATA_DIR / "mount.json").exists():  # Written last.
        raise FileNotFoundError("Run example/custom_viewers/generate_dataset.py first.")
    register_reader(NUMPY_ARCHIVE)
    register_reader(PROTEIN_DATA_BANK)
    register_viewer("Volume", is_volume, show_volume)
    register_viewer("Image", is_image, show_image)
    register_viewer("Molecule", is_molecule, show_molecule)
    filesystem = FileSystem.from_spec(json.loads((DATA_DIR / "mount.json").read_text()),
                                      relative_to=DATA_DIR)
    browse_filesystem(filesystem)


if __name__ == "__main__":
    main()
