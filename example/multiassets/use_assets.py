"""Access the multiasset fixture by relative path and alias."""

from pathlib import Path

from lazyfields import AssetsCollection
from lazyfields.extra_readers import NUMPY_ARCHIVE


def main() -> None:
    # Locate the fixture beside this script, regardless of working directory.
    root = Path(__file__).resolve().parent / "data" / "assets"
    if not root.is_dir():
        raise FileNotFoundError(
            "Generate the example data first: "
            "python3 example/multiassets/generate_dataset.py"
        )

    batch_path = Path("batches/batch_a.json")
    access = AssetsCollection(
        filepaths=sorted(path.relative_to(root) for path in root.rglob("*") if path.is_file()),
        readers=[NUMPY_ARCHIVE],  # Built-in readers are already available.
        relative_to=root,
        aliases={
            "batch": batch_path,
            "signals": "observables/batch_a_run2.npz",
            "structure": "structures/small.npz",
        },
    )
    print(f"Collection: {len(access)} physical assets")

    # Relative paths and aliases resolve the same file.
    print("By relative path:", access(batch_path)["runs/run2/energy"])
    print("By alias:", access("batch")["runs/run2/energy"])

    # Use slash paths for nested data, and tuple indices or slices for selections.
    print("First signal value:", access("signals")["signal", 0])
    print("First two signal values:", access("signals")["signal", 0:2])
    print("First two x coordinates:", access("structure")["positions", 0:2, 0])
    print("Selected fields:", access("batch")[["runs/run2/temperature", "runs/run2/energy"]])

    # Reader operations are also available on each bound asset.
    print("Physical JSON keys:", list(access("batch")[:]))
    print("Structure inspection:", list(access("structure").items()))
    print("Signal slice through get:", access("signals").get("signal", selection=slice(0, 2)))

    # Follow the JSON's relative links to run-specific and shared NPZ files.
    # All physical access goes through the same collection.
    for path in sorted((root / "batches").glob("*.json")):
        batch_path = path.relative_to(root)
        batch = access(batch_path)
        for run_key, run in batch["runs"].items():
            observables = access(batch_path.parent / run["observables_file"])
            structure_path = batch_path.parent / run["structure_file"]
            structure = access(structure_path)
            print(
                f"{path.stem}:{run_key}: "
                f"energy={run['energy']}, "
                f"signal[:2]={observables['signal', 0:2]}, "
                f"structure={structure_path.name}, "
                f"positions.shape={structure['positions'].shape}"
            )


if __name__ == "__main__":
    main()
