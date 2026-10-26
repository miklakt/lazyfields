"""Inject volume and RGB image viewers into one mounted filesystem browser."""

import json
from pathlib import Path

import numpy as np

from lazyfields.browser import browse_filesystem, register_viewer
from lazyfields.filesystem import FileSystem
from lazyfields.readers import register_reader
from lazyfields.readers.npz import NUMPY_ARCHIVE


DATA_DIR = Path(__file__).resolve().parent / "data"


def is_numeric_3d(kind, attrs):
    shape = attrs.get("shape", ())
    return (kind == "dataset" and len(shape) == 3 and all(shape)
            and np.dtype(attrs["dtype"]).kind in "biuf")


def is_image(kind, attrs):
    """Recognize channel-first RGB arrays using metadata alone."""
    return is_numeric_3d(kind, attrs) and attrs["shape"][0] == 3


def is_volume(kind, attrs):
    """Offer volume rendering for numeric 3D arrays other than RGB images."""
    return is_numeric_3d(kind, attrs) and not is_image(kind, attrs)


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


def main():
    if not all((DATA_DIR / name).exists() for name in ("volume.npz", "image.npz", "mount.json")):
        raise FileNotFoundError("Run example/custom_viewers/generate_dataset.py first.")
    register_reader(NUMPY_ARCHIVE)
    register_viewer("Volume", is_volume, show_volume)
    register_viewer("Image", is_image, show_image)
    filesystem = FileSystem.from_spec(json.loads((DATA_DIR / "mount.json").read_text()),
                                      relative_to=DATA_DIR)
    browse_filesystem(filesystem)


if __name__ == "__main__":
    main()
