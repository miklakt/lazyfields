# Editable filesystem example

`FileSystem.from_file(path, reader=None)` wraps a reader as a filesystem rooted
at a file's contents. Mount these trees to compose a larger filesystem, then edit
or clone any subtree. Values remain read-only in their source files. File-backed
roots, composed filesystems, and clones use the same implementation.

The filesystem does not interpret strings as links. The example chooses
which stored references to resolve and which nodes to replace.

## Run the examples

With the checkout installed in the active Python environment and Tk available:

```bash
python3 example/composed_filesystem/mount_all.py
python3 example/composed_filesystem/resolve_and_merge.py
```

Each script saves its composition next to itself as a specification
(`mount_all.json`, `resolve_and_merge.json`) and reuses it on later
runs instead of composing again. Pass `--rebuild` to compose and save it anew.

`mount_all.py` browses the physical filesystem. `resolve_and_merge.py` clones it,
replaces the partition reference fields with the referenced files, and browses a
reference table with one row per partition; selecting a row opens its group below. Expand groups
on the left or double-click children on the right. Leaf values get short previews
as they appear in the right pane; group contents wait until opened. Double-click
a dataset in the right pane to open its formatted text popup. OS directories show `📁`, file roots `📄`, other groups `⊞`, datasets
`▦`, and scalars `·` in both panes; only groups have expansion arrows. Children are inspected one level at a time. Returning to a parent
shows remembered short text previews; full values are not cached. Relocated mount boundaries also show `↪`, so relocated file roots show `↪ 📄`.

## Fixtures

All paths are relative to `example/composed_filesystem/data/`:

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

contents = FileSystem.from_file("example/composed_filesystem/data/metadata/alpha.json")
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

An unedited directory filesystem retains its glob in `to_spec()`, so rebuilding
the specification discovers the currently matching files.

## Clone, mount, replace, unmount

```python
from pathlib import Path
from lazyfields.filesystem import FileSystem

root = Path("example/composed_filesystem/data").resolve()
fs = FileSystem(
    sorted(path.relative_to(root) for path in root.rglob("*.json")),
    relative_to=root,
)
view = fs.clone()

path = "partitions.json/partitions/alpha/file_ref"
reference = view.read(path)  # 'metadata/alpha.json'
view.replace(path, fs.node(reference))

view.read(f"{path}/coefficients", 0)  # 1.01
fs.read(path)  # 'metadata/alpha.json'
```

`mount` requires a vacant path. Mounting over an existing node raises
`FileExistsError`; use `replace` to replace an existing entry. Missing parent groups are created,
but mounting a child beneath a scalar or dataset raises `NotADirectoryError`.
Unmounting a group removes its whole logical subtree. Physical files are unchanged.
Mounting, replacing, or unmounting the root is rejected; use `clone(path)` to choose a subtree root.

`replace(path, source)` prepares an independent mounted copy and swaps it into an
existing path in one namespace update. Preparation failures leave the old attachment
intact. It raises `KeyError` for a missing destination and does not create parents.
Like unmount followed by mount, it moves the entry to the end of its parent's
listing. The source can be inside the subtree being replaced.

Use `FileSystem()` for an empty namespace. `clone(path)` returns another
`FileSystem` rooted at that subtree. A mount target must be a node obtained
with `node(path)` from any filesystem:

```python
view = fs.clone("partitions.json/partitions/alpha")
view.replace("file_ref", fs.node("metadata/alpha.json"))

combined = FileSystem()
combined.mount("alpha", view.node())
combined.mount("metadata", fs.node("metadata"))
```

`mount`, `replace`, and `unmount` work on the original filesystem too. Cloning or mounting a
node copies its current namespace structure, including edits. Subsequent namespace
changes are independent. Immutable physical source bindings remain shared and values
are read afresh; dataset values are never copied. Rooted scalar and dataset clones
support the same native reads and selections as their sources.

## Reference resolution in the demo

`resolve_and_merge.py` registers only `partitions.json` and clones its `partitions` subtree,
so alpha, beta, and gamma appear directly at the view root. It reads the partition
mapping, resolves references relative to the parent of `partitions.json`, and calls
`replace`. Referenced files are wrapped with `FileSystem.from_file`. It mounts the
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
remain accessible through the unedited original filesystem and its browser.

## Access and laziness

`fs.stat(path)` and `node.stat()` expose the physical file's `os.stat_result`.
`fs.digest(path)` and `node.digest()` return a hex string that changes when a file
beneath the node changes, is deleted, or the namespace below it is edited. Nodes
inside a file share that file's size and modification time, so any change to the
file changes all of them. Only file stats are read, and only below the node.
File-backed descendants inherit their source file's stats; groups composed from
multiple sources have no single file stat and return `None`. Stats are read on
request and are not cached. `locate(path)` supplies source paths when aggregate
information is needed, such as the latest source modification time for a table row.

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

## Composition specifications

Every filesystem can export its current composition as a plain, JSON-compatible
dictionary. No operation history, parent pointers, Python source, or pickle is
required. Existing imperative construction works unchanged:

```python
import json

# After constructing view in resolve_and_merge.py:
spec = view.to_spec(relative_to=root)
Path("composed.json").write_text(json.dumps(spec, indent=2))

restored = FileSystem.from_spec(
    json.loads(Path("composed.json").read_text()),
    relative_to=root,
)
restored.read("alpha/file_refs/values")
```

All exported paths are relative. `to_spec()` uses the filesystem's `relative_to`
directory by default; pass `relative_to` to choose a different export base.
`from_spec()` resolves paths against the supplied `relative_to` directory, or
the current working directory when omitted. The base itself is not saved, and
absolute paths in specifications are rejected. Sources outside the chosen base
use `..`; preserve that relative layout when moving them, or choose a common base.
Export performs no source reads, metadata inspection, or directory discovery.
Restoring explicit file lists and exact child snapshots is likewise read-free.
Patterns scan directories without reading file contents. Restoring source overlays
discovers immediate child names in each edited source group; readers without
metadata support may load file contents for this discovery. Array computations
remain lazy.
Export returns a detached dictionary; later edits to it or the filesystem do not
affect one another. Call `to_spec()` again to capture later edits.

Node types are inferred from fields and parent context:

- `path` identifies a file, with optional `infile_path` and a named `reader`.
- `paths` registers relative filenames or glob patterns, such as `blocks/*.json`
  or `**/*.json`. Patterns select supported files, skip directories and unsupported
  formats, and combine matches in sorted order without duplicates.
  Parent groups and mount boundaries are reconstructed from the filenames, so
  `{"paths": ["blocks/*.json"]}` places the block files under `blocks`. Patterns
  are exported only relative to the registration's own base; exported relative
  to another base, as with `from_directory("data/blocks")` relative to `data`,
  the files are written as explicit `path` children.
- `op` and ordered `sources` identify a computed dataset, with `axis` for
  `stack` or `concatenate`. Other supported operations are `collect`, `vstack`,
  and `hstack`. Axis zero is the default and is omitted on export.
- Otherwise, a child of a file-backed parent inherits its file binding. At the
  root or under a virtual group, the default is a virtual group.

An explicit `type` overrides inference. Export uses `"type": "group"` only
when a virtual group beneath a file-backed parent needs to prevent inheritance.
Conflicting or unsupported fields are rejected.

A namespace containing ordinary file registrations is compacted automatically:

```json
{
  "paths": ["blocks/alpha_first.json", "metadata/alpha.json", "partitions.json"]
}
```

Literal filenames select a fixed set. To discover membership again on each
reconstruction, use patterns, optionally mixed with literal filenames:

```json
{
  "paths": ["blocks/*.json", "metadata/*.json", "partitions.json"]
}
```

`{"paths": ["**/*.json"]}` includes JSON files at all depths, including the root.
`{"paths": ["*/*.json"]}` selects exactly one directory level and excludes
`partitions.json` at the root. Patterns with no matches produce an empty namespace.
`from_directory()` retains its original pattern for an unmodified registration
tree, including an empty one. Edits fall back to an explicit namespace or file
list so reconstruction preserves those edits. Empty groups, aliases, and custom
reader bindings likewise use the general hierarchical format when needed.

For example, a file subtree alone needs only:

```json
{
  "path": "partitions.json",
  "infile_path": "partitions"
}
```

A file or group can have `children`, a mapping from child names to specifications.
For a file, absent `children` delegates to the original source.
Present `children` records the exact current
namespace, including removals; `children: {}` represents an empty group. Mapping
insertion order determines child order; merge inputs always use ordered arrays.
The file binding remains available for origin and stat information. A natural
file child needs only `{}`: its filename, infile path, and reader are inherited
from the parent and its child name. For example:

```json
{
  "path": "partitions.json",
  "infile_path": "partitions/alpha",
  "children": {
    "scalar_a": {},
    "vector": {},
    "file_ref": {"path": "metadata/alpha.json"}
  }
}
```

This snapshot retains exactly those three child names. An empty file-backed group
has a file binding and `children: {}`; an empty virtual group beneath a file has
`"type": "group"` and `children: {}`, and has no physical origin.

`to_spec()` automatically omits unchanged children when the source group's
original child names are known, recording only overrides and unmounts. Otherwise
it retains exact `children` listings. No export mode is required. The loader
accepts both forms, including a mixture at different nodes:

```json
{
  "path": "partitions.json",
  "infile_path": "partitions/alpha",
  "overrides": {
    "file_ref": {"path": "metadata/alpha.json"}
  },
  "unmount": ["vector"]
}
```

This inherits all current source children, replaces `file_ref`, and hides `vector`.
`overrides` also supports nested edits to an inherited group. `unmount` contains
immediate child names, matching the existing method's namespace-only behavior;
it never deletes source data. A missing name is still hidden if it reappears in
the source on a later reconstruction. A name cannot be both overridden and
unmounted at the same node. `children` cannot be combined with either edit field
at that node. Overlays require a file-backed source; virtual groups and resolved
merge groups retain explicit children.

Overlay discovery happens once per reconstruction. Newly added source fields are
visible on reconstruction; there is no automatic refresh afterward. Unchanged
children retain source order, while explicitly mounted replacements are appended
in override order, matching `replace`. Nested edits to an inherited
group keep that group's position.

Any node can carry `attrs`, such as `{"units": "nm", "viewer": "Slices"}`;
`node.attrs` adds them over the `dtype` and `shape` that datasets have by
default. HDF5 and NPZ give those from metadata; other readers load the dataset,
unless `attrs` already give both. For a field inside a file, use an override:

```json
{"path": "row_00.h5", "overrides": {"array3d": {"attrs": {"units": "nm"}}}}
```

Viewers in the browsers decide whether they fit from a node's kind and `attrs`;
its `viewer` attribute names the one opened by default when it fits.

When first expanding a source group, the runtime retains its original child names
as an immutable tuple. Export derives overrides and hidden names from that tuple
and the current tree, without a command log or source reread. If a node came from
an exact specification and its original names are unknown, export keeps
that node's exact `children` listing rather than guessing which names were removed.

Namespace positions follow the hierarchy. Physical base directories are never
repeated per node. A standalone root with an aliased namespace position can retain
an optional relative `namespace_path`.

Mounts are inferred for foreign file bindings and merged subtrees. Natural file
children, structural groups, and children produced by a merge are not implicitly
mounts. `is_mount` is emitted only when the actual boundary differs from that
inference. Relocation follows the mount's position and sources; `is_relocated`
is likewise only an override. Computed datasets imply `is_merged`; merged groups
retain `is_merged: true`, since resolved preference/prefix outputs alone cannot
identify their merge provenance.

Merges are saved after key discovery: prefixing has already produced named
children, and preference has selected a source subtree. Computed datasets retain
their ordered input specifications and operation. Restoration therefore does not
repeat discovery or validate input shapes until they are read.

The composed example exports an overlay with concrete references for alpha, beta,
and gamma. New manifest partitions can appear when it is reconstructed, but their
reference fields remain ordinary source values. Existing resolved references keep
their selected filenames. Rebuild the view with the Python construction code to
resolve new or changed references. Values in selected files remain live and lazy.
An exact snapshot instead retains its discovered partition and field names.
Directory patterns discover file membership again on reconstruction. Literal
file lists continue to select only the explicitly named files.

Custom readers need explicit names on both sides:

```python
readers = {"custom": custom_reader}
spec = filesystem.to_spec(readers=readers)
restored = FileSystem.from_spec(spec, readers=readers)
```

Readers inferred from the current suffix registry need no name; restoration uses
that registry again. An unregistered reader override without a supplied name
raises an error instead of silently changing behavior. No reader callbacks are
serialized. Specifications capture the supported filesystem state, not arbitrary
Python attributes, monkeypatched methods, or subclass-specific behavior.

## Scope

Reference discovery belongs to user code. Per-key merge rules provide collection,
prefixing, source preference, stacking, and concatenation. Deferred resolvers,
automatic refresh, and read coalescing are not implemented.
