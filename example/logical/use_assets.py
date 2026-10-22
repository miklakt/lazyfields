"""Explore the logical-design fixtures through their physical filesystem."""

from pathlib import Path

from lazyfields.assets import AssetsCollection


def main() -> None:
    root = Path(__file__).resolve().parent / "data"
    assets = AssetsCollection(
        sorted(path.relative_to(root) for path in root.rglob("*.json")),
        relative_to=root,
    )
    print("Root:", assets.listdir())
    print("Partitions:", assets.listdir("partitions.json/partitions"))
    vector = assets.node("partitions.json/partitions/alpha/vector")
    print("Locator:", vector.locator)
    print("Vector[:2]:", vector.read(slice(0, 2)))
    print("Kind:", vector.kind)
    print("Second block:", assets.read("blocks/alpha_second.json/values"))
    print("Stored reference:", assets.read("partitions.json/partitions/alpha/file_ref"))


if __name__ == "__main__":
    main()
