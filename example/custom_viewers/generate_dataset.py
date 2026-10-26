"""Prepare a native CT volume and a channel-first RGB image for custom viewers.

CT source: https://github.com/Slicer/Slicer/blob/main/Modules/Scripted/SampleData/SampleData.py
The CT-MR Brain sample was donated for use without restrictions.
License: https://github.com/Slicer/Slicer/blob/main/License.txt
"""

import gzip
import hashlib
import json
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import numpy as np


CT_SHA256 = "6a5b6caccb76576a863beb095e3bfb910c50ca78f4c9bf043aa42f976cfa53d1"
CT_URL = f"https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/{CT_SHA256}"
CT_SHAPE = (76, 507, 512)  # NumPy order: z, y, x.
DATA_DIR = Path(__file__).resolve().parent / "data"
CACHE_PATH = DATA_DIR / "CT-brain.nrrd"
IMAGE_URL = "https://raw.githubusercontent.com/opencv/opencv/4.12.0/samples/data/lena.jpg"
IMAGE_SHA256 = "7de7ed51a1594fff247f4cae2301eceacf5313d6011e37b4a4c8733f7bb72c07"
IMAGE_CACHE_PATH = DATA_DIR / "lena.jpg"


def fetch(url, sha256, path, label):
    """path's bytes, downloaded from url on first use; both checked against sha256."""
    if path.exists():
        data = path.read_bytes()
    else:
        print(f"Downloading {label}...", flush=True)
        with urlopen(url, timeout=60) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(f"Checksum mismatch; remove {path} and retry.")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".tmp").write_bytes(data)
        path.with_suffix(".tmp").replace(path)
    return data


def load_head_ct():
    """Return the original (76, 507, 512) int32 CT, without resampling.

    This reads the fixed, checksum-pinned NRRD sample (gzip, little-endian int32),
    rather than requiring a medical-image reader.
    """
    data = fetch(CT_URL, CT_SHA256, CACHE_PATH, "3D Slicer head CT (about 16 MB)")
    _, payload = data.split(b"\n\n", 1)
    return np.frombuffer(gzip.decompress(payload), dtype="<i4").reshape(CT_SHAPE)


def load_rgb_image():
    """Load OpenCV's Lena sample as (3, 512, 512) uint8, without resizing."""
    from PIL import Image

    data = fetch(IMAGE_URL, IMAGE_SHA256, IMAGE_CACHE_PATH, "Lena RGB image (about 90 KB)")
    with Image.open(BytesIO(data)) as image:
        return np.moveaxis(np.asarray(image.convert("RGB")), -1, 0)


def volume_metadata():
    """Read native voxel spacing from the cached NRRD header, in NumPy axis order."""
    with CACHE_PATH.open("rb") as source:
        for line in source:
            if line.startswith(b"space directions:"):
                vectors = line.decode().split(":", 1)[1].split()
                directions = [np.fromstring(vector.strip("()"), sep=",") for vector in vectors]
                return {"spacing": np.linalg.norm(directions, axis=1)[::-1].tolist(), "units": "mm"}
            if line == b"\n":
                break
    raise ValueError("CT header has no voxel spacing.")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(DATA_DIR / "volume.npz", voxels=load_head_ct())
    np.savez(DATA_DIR / "image.npz", rgb=load_rgb_image())
    spec = {"children": {
        "volume.npz": {"path": "volume.npz", "overrides": {
            "voxels": {"attrs": volume_metadata()}}},
        "image.npz": {"path": "image.npz"},
    }}
    (DATA_DIR / "mount.json").write_text(json.dumps(spec, indent=2) + "\n")
    print(f"Created volume.npz, image.npz, and mount.json in {DATA_DIR}")


if __name__ == "__main__":
    main()
