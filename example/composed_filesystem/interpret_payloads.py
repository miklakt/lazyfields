"""Browse inside a base64-encoded NPZ archive stored in a JSON field, through a registered view."""

import base64
import io
from pathlib import Path

import numpy as np

from lazyfields.browser import browse_filesystem
from lazyfields.filesystem import FileSystem
from lazyfields.readers import View, register_view


def npz_base64(text):
    """The arrays of a base64-encoded NPZ archive."""
    with np.load(io.BytesIO(base64.b64decode(text))) as archive:
        return {name: archive[name] for name in archive.files}


def main() -> None:
    register_view("npz-base64", View(npz_base64))
    root = Path(__file__).resolve().parent / "data"
    # The "view" attribute marks the field to interpret; other strings stay strings.
    fs = FileSystem.from_spec(
        {"path": "payloads.json", "children": {"frames": {"attrs": {"view": "npz-base64"}}}},
        relative_to=root,
    )
    print("frames:", fs.listdir("frames"), "first positions:", fs.read("frames/positions", (0, slice(2))).tolist())
    browse_filesystem(fs)


if __name__ == "__main__":
    main()
