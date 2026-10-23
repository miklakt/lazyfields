"""Replace fixture reference fields with mounts of their referenced files."""

from pathlib import Path

from lazyfields.filesystem import FileSystem
from explorer import explore


def main() -> None:
    root = Path(__file__).resolve().parent / "data"
    source = Path("partitions.json")
    fs = FileSystem(source, relative_to=root)
    view = fs.clone()
    # Explicit discovery: read the stored references once, then bind lazy targets.
    for name, partition in fs.read(f"{source}/partitions").items():
        prefix = f"{source}/partitions/{name}"
        view.unmount(f"{prefix}/file_ref")
        target = FileSystem.from_file(root / source.parent / partition["file_ref"])
        view.mount(f"{prefix}/file_ref", target.node())
        blocks = FileSystem()
        for index, reference in enumerate(partition["file_refs"]):
            target = FileSystem.from_file(root / source.parent / reference)
            blocks.mount(str(index), target.node())
        view.unmount(f"{prefix}/file_refs")
        view.mount(f"{prefix}/file_refs", blocks.node())
    explore(view)


if __name__ == "__main__":
    main()
