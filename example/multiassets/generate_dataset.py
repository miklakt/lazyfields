"""Generate the experimental many-to-many physical/logical file use case.

Run from the repository root:
    python3 example/multiassets/generate_dataset.py

Requires NumPy. Does not require or implement the proposed logical-file API.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def generate_dataset(destination: Path) -> Path:
    """Write deterministic assets and an expected mapping outside the asset set."""
    assets = destination / "assets"
    for name in ("batches", "observables", "structures"):
        (assets / name).mkdir(parents=True, exist_ok=True)

    np.savez(
        assets / "structures" / "small.npz",
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        atomic_numbers=np.array([1, 8]),
    )
    np.savez(
        assets / "structures" / "large.npz",
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        atomic_numbers=np.array([1, 6, 8]),
    )

    structure_assignments = {
        "batch_a": ("small", "small"),
        "batch_b": ("small", "large"),
        "batch_c": ("large", "small"),
    }
    logical_files = []
    for batch_index, (batch_key, structures) in enumerate(structure_assignments.items()):
        batch_asset = f"batches/{batch_key}.json"
        batch = {
            "metadata": {
                "operator": "example",
                "comment": "Physical-file metadata; excluded from logical files.",
            },
            "runs": {},
        }
        for run_index, structure in enumerate(structures, start=1):
            number = batch_index * 2 + run_index
            run_key = f"run{run_index}"
            observables_asset = f"observables/{batch_key}_{run_key}.npz"
            structure_asset = f"structures/{structure}.npz"
            temperature = 280 + 10 * number
            energy = -10.0 + 0.5 * (number - 1)
            signal = np.arange(4, dtype=float) + number
            np.savez(
                assets / observables_asset,
                time=np.array([0.0, 0.5, 1.0, 1.5]),
                signal=signal,
            )
            batch["runs"][run_key] = {
                "run_key": run_key,
                "temperature": temperature,
                "energy": energy,
                # The user strategy interprets these paths relative to this JSON.
                "observables_file": f"../{observables_asset}",
                "structure_file": f"../{structure_asset}",
            }
            fields = {
                key: {"asset": batch_asset, "path": f"runs/{run_key}/{key}"}
                for key in ("run_key", "temperature", "energy")
            }
            fields.update({
                key: {"asset": observables_asset, "path": key}
                for key in ("time", "signal")
            })
            fields.update({
                key: {"asset": structure_asset, "path": key}
                for key in ("positions", "atomic_numbers")
            })
            logical_files.append({
                "logical_id": f"{batch_key}:{run_key}",
                "locator": {"asset": batch_asset, "run_key": run_key},
                "fields": fields,
                "expected": {
                    "run_key": run_key,
                    "temperature": temperature,
                    "energy": energy,
                    "signal_first_two": signal[:2].tolist(),
                    "signal_sum": float(signal.sum()),
                    "atom_count": 2 if structure == "small" else 3,
                },
            })
        (assets / batch_asset).write_text(json.dumps(batch, indent=2) + "\n")

    expected_path = destination / "expected_logical_files.json"
    expected_path.write_text(json.dumps({
        "description": "Expected mappings for the experiment; not a serialization API.",
        "assets_root": "assets",
        "logical_files": logical_files,
    }, indent=2) + "\n")
    return expected_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "data",
        help="Output directory; matching fixture filenames are overwritten.",
    )
    args = parser.parse_args()
    expected_path = generate_dataset(args.output)
    print(f"Created 3 JSON batches, 6 run NPZ files, and 2 shared structure NPZ files.")
    print(f"Expected mappings for 6 logical files: {expected_path}")


if __name__ == "__main__":
    main()
