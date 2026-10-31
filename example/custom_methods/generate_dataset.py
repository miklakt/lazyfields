"""Prepare a native CT volume, a channel-first RGB image and two protein structures for custom methods.

CT source: https://github.com/Slicer/Slicer/blob/main/Modules/Scripted/SampleData/SampleData.py
The CT-MR Brain sample was donated for use without restrictions.
License: https://github.com/Slicer/Slicer/blob/main/License.txt
Image source: scikit-image's coffee sample, CC0 by the photographer, Rachel Michetti.
Protein source: RCSB PDB entries 1UBQ and 4HHB; PDB archive data is CC0.
"""

import gzip
import hashlib
import json
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import numpy as np


DATA_DIR = Path(__file__).resolve().parent / "data"
CT_SHA256 = "6a5b6caccb76576a863beb095e3bfb910c50ca78f4c9bf043aa42f976cfa53d1"
CT_URL = f"https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/{CT_SHA256}"
CT_SHAPE = (76, 507, 512)  # NumPy order: z, y, x.
CT_CACHE_PATH = DATA_DIR / "CT-brain.nrrd"
IMAGE_URL = "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.24.0/skimage/data/coffee.png"
IMAGE_SHA256 = "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7"
IMAGE_CACHE_PATH = DATA_DIR / "coffee.png"
# Ubiquitin and human deoxyhaemoglobin by file name, as RCSB serves them.
PDB_SHA256 = {"1UBQ.pdb": "d4a6812d8951cf6594e6a0763f089e35f5a80b62acb3c117b2c5565228a7b161",
              "4HHB.pdb": "abf382e0fd84b53bd4c08b373d37c0d7fe9fbe1a9a9c3851b2f5f2fbbe627101"}


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
    """Load the head CT sample as a (76, 507, 512) int32 array."""
    data = fetch(CT_URL, CT_SHA256, CT_CACHE_PATH, "3D Slicer head CT (about 16 MB)")
    _, payload = data.split(b"\n\n", 1)
    return np.frombuffer(gzip.decompress(payload), dtype="<i4").reshape(CT_SHAPE)


def load_rgb_image():
    """Load scikit-image's coffee sample as a (3, 400, 600) uint8 array."""
    from PIL import Image

    data = fetch(IMAGE_URL, IMAGE_SHA256, IMAGE_CACHE_PATH, "coffee RGB image (about 470 KB)")
    with Image.open(BytesIO(data)) as image:
        return np.moveaxis(np.asarray(image.convert("RGB")), -1, 0)


def volume_metadata():
    """Read native voxel spacing from the cached NRRD header, in NumPy axis order."""
    with CT_CACHE_PATH.open("rb") as source:
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
    for name, sha256 in PDB_SHA256.items():
        fetch(f"https://files.rcsb.org/download/{name}", sha256, DATA_DIR / name, f"PDB entry {name}")
    spec = {"children": {
        "volume.npz": {"path": "volume.npz", "children": {
            "voxels": {"attrs": volume_metadata()}}},
        "image.npz": {"path": "image.npz"},
        **{name: {"path": name} for name in PDB_SHA256},
    }}
    (DATA_DIR / "mount.json").write_text(json.dumps(spec, indent=2) + "\n")
    print(f"Created volume.npz, image.npz, {', '.join(PDB_SHA256)}, and mount.json in {DATA_DIR}")


if __name__ == "__main__":
    main()
