# Editable filesystem example

`FileSystem.from_file(path, reader=None)` wraps a reader as a filesystem rooted
at a file's contents. Mount these trees to compose a larger filesystem, then edit
or clone any subtree. Values remain read-only in their source files. File-backed
roots, composed filesystems, and clones use the same implementation.

The filesystem does not interpret strings as links. The example chooses
which stored references to resolve and which nodes to replace.

## Run the explorers

With the checkout installed in the active Python environment and Tk available:

```bash
python3 example/filesystem/minimal_filesystem.py
python3 example/filesystem/composed_filesystem.py
```

The first explorer shows the physical filesystem. The second clones it, then
replaces the partition reference fields with the referenced files. Expand groups
on the left or double-click children on the right. Leaf values get short previews
as they appear in the right pane; group contents wait until opened. Double-click
a dataset in the right pane to open its formatted text popup. OS directories show `📁`, file roots `📄`, other groups `⊞`, datasets
`▦`, and scalars `·` in both panes; only groups have expansion arrows. Children are inspected one level at a time. Returning to a parent
shows remembered short text previews; full values are not cached. Relocated mount boundaries also show `↪`, so relocated file roots show `↪ 📄`.

## Fixtures

All paths are relative to `example/filesystem/data/`:

```text
partitions.json
metadata/{alpha,beta,gamma}.json
blocks/{alpha,beta,gamma}_{first,second}.json
```

`partitions.json` contains three entries under `partitions`. Each has `scalar_a`,
`scalar_b`, a length-3 `vector`, a `(2, 3)` `matrix`, a `file_ref` string, and an
ordered `file_refs` list. Metadata files contain `description`, `label`, `weight`,
and `coefficients`. Block files contain `description`, `block_index`, `coordinates`,
and `values`. The distinct fixture values help verify source order and selection.

## Files as filesystem roots

```python
from lazyfields.filesystem import FileSystem

contents = FileSystem.from_file("example/filesystem/data/metadata/alpha.json")
contents.read("coefficients", 0)  # 1.01

fs = FileSystem()
fs.mount("metadata/alpha.json", contents.node())
fs.read("metadata/alpha.json/coefficients", 0)  # 1.01
```

The reader is inferred from the filename unless supplied explicitly. Neither
`from_file` nor mounting reads file contents. `FileSystem(paths, relative_to=root)`
is shorthand for wrapping and mounting each relative filename this way. Repeated
filenames are registered once; overlapping file paths are rejected before mounting.

`FileSystem.from_directory(path, pattern="*")` mounts matching files supported by
registered readers, preserving their physical relative paths. Unsupported files
and directories are skipped. Use `pattern="*.json"` to select JSON files, or
`pattern="**/*.json"` to include nested folders. Construction does not read file
contents.

## Clone, unmount, mount

```python
from pathlib import Path
from lazyfields.filesystem import FileSystem

root = Path("example/filesystem/data").resolve()
fs = FileSystem(
    sorted(path.relative_to(root) for path in root.rglob("*.json")),
    relative_to=root,
)
view = fs.clone()

path = "partitions.json/partitions/alpha/file_ref"
reference = view.read(path)  # 'metadata/alpha.json'
view.unmount(path)
view.mount(path, fs.node(reference))

view.read(f"{path}/coefficients", 0)  # 1.01
fs.read(path)  # 'metadata/alpha.json'
```

`mount` requires a vacant path. Mounting over an existing node raises
`FileExistsError`; explicitly unmount it first. Missing parent groups are created,
but mounting a child beneath a scalar or dataset raises `NotADirectoryError`.
Unmounting a group removes its whole logical subtree. Physical files are unchanged.
Mounting or unmounting the root is rejected; use `clone(path)` to choose a subtree root.

Use `FileSystem()` for an empty namespace. `clone(path)` returns another
`FileSystem` rooted at that subtree. A mount target must be a node obtained
with `node(path)` from any filesystem:

```python
view = fs.clone("partitions.json/partitions/alpha")
view.unmount("file_ref")
view.mount("file_ref", fs.node("metadata/alpha.json"))

combined = FileSystem()
combined.mount("alpha", view.node())
combined.mount("metadata", fs.node("metadata"))
```

`mount` and `unmount` work on the original filesystem too. Cloning or mounting a
node copies its current namespace structure, including edits. Subsequent namespace
changes are independent. Immutable physical source bindings remain shared and values
are read afresh; dataset values are never copied. Rooted scalar and dataset clones
support the same native reads and selections as their sources.

## Reference resolution in the demo

`composed_filesystem.py` registers only `partitions.json` and clones its `partitions` subtree,
so alpha, beta, and gamma appear directly at the view root. It reads the partition
mapping, resolves references relative to the parent of `partitions.json`, and calls `unmount` followed by
`mount`. Referenced files are wrapped with `FileSystem.from_file`. It mounts the
single `file_ref` target directly. It replaces `file_refs`
with a merged group for each partition: descriptions get filename prefixes,
block indices are collected into a list, coordinates come from the first block,
and values are stacked along a new first axis. The values shape is `(2, 2, 3)`.

```text
view/
├── alpha/
│   ├── scalar_a
│   ├── scalar_b
│   ├── vector
│   ├── matrix
│   ├── file_ref/                  metadata/alpha.json
│   │   └── coefficients
│   └── file_refs/
│       ├── alpha_first_description
│       ├── alpha_second_description
│       ├── block_index            [0, 1]
│       ├── coordinates
│       └── values                 stacked from both block files
├── beta/                         same structure
└── gamma/                        same structure
```

The tree is abbreviated; mounted files retain all their fields. Raw references
remain accessible through the unedited original filesystem and its explorer.

## Access and laziness

`fs.stat(path)` and `node.stat()` expose the physical file's `os.stat_result`.
File-backed descendants inherit their source file's stats; groups composed from
multiple sources have no single file stat and return `None`. Stats are read on
request and are not cached. `locate(path)` supplies source paths when aggregate
information is needed, such as the latest source creation time for a table row.

Filesystems and their `Node` handles expose `node`, `listdir`,
`read`, and `locate`. Handles also expose `path` and `kind`. An empty path or `/`
names the namespace root. Repeated separators and `.` normalize; `..` is rejected.
Handle-relative `node(path)` binds a descendant. Selections are separate from paths.

File registration and handle creation do not read data. The merge factory inspects
immediate source keys once; array computation waits until reading. Untouched subtrees delegate
reads, listings, and metadata inspection directly to their source. Editing a parent
records its child handles once along the edited path. Readers may need data reads
to discover those names: JSON generally loads the file, while HDF5 can inspect
structure and NPZ can list arrays from their headers without reading dataset values.
Expanded groups keep their logical child
names until explicitly edited; there is no automatic rediscovery or refresh.

`read(path, selection=None)` materializes the current logical subtree. Edited
groups traverse their recorded children, so whole-group reads agree with individual
paths and unmounted nodes stay absent. Numeric values keep their reader's native
types and selection behavior: JSON supports list slicing; HDF5 supports direct
multidimensional selections. No implicit NumPy conversion or value cache is added.

`locate(path, selection=None)` returns `list[SourceLocator]` without reads or
structure discovery. Each record holds an absolute `file_path`, an optional
`infile_path`, and `selection`. Namespace aliases preserve those source coordinates,
even when filesystems use the same relative filename under different roots.
An untouched subtree can remain one locator; edited groups collect dependencies
from their current children. Repeated source uses remain repeated. Use `listdir`
to inspect the current namespace.

## Scope

Reference discovery belongs to user code. Per-key merge rules provide collection,
prefixing, source preference, stacking, and concatenation. Deferred resolvers,
automatic refresh, and read coalescing are not implemented.
