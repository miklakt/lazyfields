"""Inject volume, RGB image, molecule, channel-splitting and point-cloud methods, plus the built-in
slices, plot and table methods, into one mounted filesystem browser, with NPZ and PDB readers.

Split Channels returns plain named arrays, no Tk or matplotlib of its own: open_value shows its
result with whichever method fits, here XYZ Scatter, exactly as a registered store method's result
would be shown.
"""

import json
from pathlib import Path

import numpy as np

from lazyfields.browser import browse_filesystem, register_method
from lazyfields.browser.extra_methods import PLOT, SLICES, TABLE
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


def split_channels(value, attrs, *, order="rgb"):
    """Each channel-first array's channel flattened to its own named array.

    order maps each array position to a channel letter, for images stored in another channel order.
    The result is shown by whichever method fits it, here XYZ Scatter — no Tk or matplotlib here.
    """
    names = {"r": "red", "g": "green", "b": "blue"}
    return {names[letter]: channel.ravel() for letter, channel in zip(order, np.asarray(value))}


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


def is_xyz_group(node):
    """Any group with at least three children, offered as a point cloud with axes chosen in a popup.

    accepts only sees child names here, not their shapes, so this is a loose heuristic: groups meant
    for this method should hold only 1D arrays of the same length, such as Split Channels' result or
    a molecule's x/y/z/element columns, which also happens to qualify.
    """
    return node.kind == "group" and len(node.children) >= 3


def show_xyz_group(value, attrs, *, x="red", y="green", z="blue"):
    """Point cloud from three named 1D arrays in the group; axes are picked in the popup above."""
    from matplotlib.figure import Figure

    figure = Figure(layout="constrained")
    axes = figure.add_subplot(projection="3d")
    axes.scatter(np.asarray(value[x]), np.asarray(value[y]), np.asarray(value[z]), s=2)
    axes.set(xlabel=x, ylabel=y, zlabel=z)
    return figure


def main():
    if not (DATA_DIR / "mount.json").exists():  # Written last.
        raise FileNotFoundError("Run example/custom_methods/generate_dataset.py first.")
    register_reader(NUMPY_ARCHIVE)
    register_reader(PROTEIN_DATA_BANK)
    register_method("Volume", is_volume, show_volume, category="Display/Volumes")
    # Registered first: wins the default glance at a photo over Split Channels, which comes next.
    register_method("Image", is_image, show_image, category="Display/Images")
    register_method("Split Channels", is_image, split_channels, category="Transform/Images")
    # Marked specific: its exact x/y/z/element shape should win over Browser, which also matches
    # any group. Left uncategorized, it appears directly in the context menu.
    register_method("Molecule", is_molecule, show_molecule, specific=True)
    # Left unmarked: this loose "any group of three or more children" heuristic loses the tie against
    # Browser, which registered first, so a nested result browses before it scatters.
    register_method("XYZ Scatter", is_xyz_group, show_xyz_group, category="Display/Plots")
    register_method(SLICES, category="Display/Volumes")
    register_method(PLOT, category="Display/Plots")
    register_method(TABLE, category="Display/Tables")
    filesystem = FileSystem.from_spec(json.loads((DATA_DIR / "mount.json").read_text()),
                                      relative_to=DATA_DIR)
    browse_filesystem(filesystem)


if __name__ == "__main__":
    main()
