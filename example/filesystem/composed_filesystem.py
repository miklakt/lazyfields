"""Expose partitions at root with referenced metadata and merged block files."""

from pathlib import Path

from lazyfields.readers import debug_readers
from lazyfields.filesystem import FileSystem
from lazyfields.filesystem.merge import collect, prefer, prefix, stack
from lazyfields.filesystem.explorer import explore


def main() -> None:
    debug_readers()
    root = Path(__file__).resolve().parent / "data"
    source = Path("partitions.json")
    fs = FileSystem(source, relative_to=root)
    view = fs.clone(f"{source}/partitions")
    # Explicit discovery: read the stored references once, then bind lazy targets.
    for name, partition in fs.read(f"{source}/partitions").items():
        view.unmount(f"{name}/file_ref")
        target = FileSystem.from_file(root / source.parent / partition["file_ref"])
        view.mount(f"{name}/file_ref", target.node())
        sources = {
            Path(reference).stem: FileSystem.from_file(root / source.parent / reference).node()
            for reference in partition["file_refs"]
        }
        blocks = FileSystem.merge(sources, rules={
            "description": prefix(),
            "block_index": collect(),
            "coordinates": prefer(*sources),
            "values": stack(axis=0),
        })
        view.unmount(f"{name}/file_refs")
        view.mount(f"{name}/file_refs", blocks.node())
    explore(view)


if __name__ == "__main__":
    main()
