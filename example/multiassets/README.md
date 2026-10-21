# Multiasset, multirow experiment

This is a concrete use case and an implementation proposal. The generator and
physical asset access work now; the logical-file API described below is not
implemented yet.

## Generate the use case

With NumPy installed and your virtual environment active, run from the repository
root. Install this checkout in editable mode so examples use the current source:

```bash
python3 -m pip install -e .
python3 example/multiassets/generate_dataset.py
```

The generator overwrites its own fixture filenames when rerun. It does not remove
other files. Use `--output /tmp/lazyfields-multiasset` for a separate destination.
The default generated `data/` directory is already ignored by Git.

```text
data/
  assets/
    batches/
      batch_a.json             # run1 and run2, plus ignored batch metadata
      batch_b.json             # run1 and run2, plus ignored batch metadata
      batch_c.json             # run1 and run2, plus ignored batch metadata
    observables/
      batch_a_run1.npz          # time and signal arrays
      batch_a_run2.npz
      batch_b_run1.npz
      batch_b_run2.npz
      batch_c_run1.npz
      batch_c_run2.npz
    structures/
      small.npz                # positions and atomic_numbers, shared by four runs
      large.npz                # positions and atomic_numbers, shared by two runs
  expected_logical_files.json   # Expected mappings and values; outside the assets
```

All asset paths below are relative to `data/assets/`.

| Logical ID | JSON asset and location | Run NPZ asset | Shared NPZ asset |
|---|---|---|---|
| batch_a:run1 | batches/batch_a.json, runs/run1 | observables/batch_a_run1.npz | structures/small.npz |
| batch_a:run2 | batches/batch_a.json, runs/run2 | observables/batch_a_run2.npz | structures/small.npz |
| batch_b:run1 | batches/batch_b.json, runs/run1 | observables/batch_b_run1.npz | structures/small.npz |
| batch_b:run2 | batches/batch_b.json, runs/run2 | observables/batch_b_run2.npz | structures/large.npz |
| batch_c:run1 | batches/batch_c.json, runs/run1 | observables/batch_c_run1.npz | structures/large.npz |
| batch_c:run2 | batches/batch_c.json, runs/run2 | observables/batch_c_run2.npz | structures/small.npz |

Each logical file depends on three physical files. Every JSON supplies two logical
files. Both structure files are shared across JSON batches. Run names repeat
across batches, so `run_key` alone is not a unique logical-file identifier.

A JSON run contains `run_key`, `temperature`, `energy`, `observables_file`, and
`structure_file`. The last two are relative to the containing JSON file. The user
strategy interprets them to establish relationships; they are not exposed as
logical data fields in this use case. Top-level JSON `metadata` is also ignored.

## Physical access available now

Run [`use_assets.py`](use_assets.py) from the repository root:

```bash
python3 example/multiassets/use_assets.py
```

The script demonstrates lookup by relative path and alias; nested fields;
indexed and sliced arrays; full-file loading; reader
inspection; and following JSON links to run-specific and shared NPZ assets.
The NPZ reader is supplied locally to the collection.

After generating the dataset, this runs from the repository root:

```python
from pathlib import Path

from lazyfields import AssetsCollection
from lazyfields.extra_readers import NUMPY_ARCHIVE

root = Path("example/multiassets/data/assets").resolve()
access = AssetsCollection(
    filepaths=[
        "batches/batch_a.json",
        "observables/batch_a_run2.npz",
        "structures/small.npz",
    ],
    readers=[NUMPY_ARCHIVE],
    aliases={
        "batch": "batches/batch_a.json",
        "run": "observables/batch_a_run2.npz",
        "structure": "structures/small.npz",
    },
    relative_to=root,
)

access("batch")["runs/run2/energy"]        # -9.5
access("run")["signal", 0:2]              # array([2., 3.])
access("structure")["positions"].shape    # (2, 3)
access("batches/batch_a.json")[:]          # Complete physical JSON
```

The collection accepts a single path or an iterable of paths.
`access("batches/batch_a.json")` uses the registered relative path. A reader
accessor provides `load()`, `get(key, selection=None)`, and `items()` bound to the
selected physical file. Iterating the collection yields relative paths.
Registering the NPZ reader globally is not required.

## Logical access

The logical file `batch_a:run2` exposes these fields:

| Logical field | Physical asset | Physical path |
|---|---|---|
| run_key | batches/batch_a.json | runs/run2/run_key |
| temperature | batches/batch_a.json | runs/run2/temperature |
| energy | batches/batch_a.json | runs/run2/energy |
| time | observables/batch_a_run2.npz | time |
| signal | observables/batch_a_run2.npz | signal |
| positions | structures/small.npz | positions |
| atomic_numbers | structures/small.npz | atomic_numbers |

Target behavior, once the API is implemented:

```python
row = reference.loc[reference["logical_id"] == "batch_a:run2"].iloc[0]
row.store["energy"]                  # -9.5
row.store["signal", 0:2]             # array([2., 3.])
row.store["positions"].shape         # (2, 3)
row.store[:]                        # All seven logical fields
```

The reference copies scalar data and leaves arrays in the physical assets:

| logical_id | run_key | temperature | energy |
|---|---|---:|---:|
| batch_a:run1 | run1 | 290 | -10.0 |
| batch_a:run2 | run2 | 300 | -9.5 |
| batch_b:run1 | run1 | 310 | -9.0 |
| batch_b:run2 | run2 | 320 | -8.5 |
| batch_c:run1 | run1 | 330 | -8.0 |
| batch_c:run2 | run2 | 340 | -7.5 |

Logical non-scalar fields are `time`, `signal`, `positions`, and `atomic_numbers`.
A single-row pipe can compute `signal_sum` or filter by `energy` after regrouping.
Pipe steps continue to accept and return one row; they never enumerate, split,
or merge logical files.

## Proposed implementation boundary

```text
asset collection -> injected grouping strategy -> logical files
                                                    |
                                              single-row pipe
                                                    |
                                               reference table
```

1. **Physical assets and readers.** An asset collection provides access to each
   physical file through the existing `Reader.load`, `Reader.get`, and
   `Reader.items` interface. Reader selection remains based on physical format.
   `Reader` itself stays unchanged. The example needs the existing JSON reader
   and the optional `NUMPY_ARCHIVE` reader.
2. **Injected grouping strategy.** Enumerate logical locators, resolve logical
   fields into `(asset, physical_path)` pairs, and enumerate logical field names.
   This strategy reads batch documents and follows their run-level asset
   references through the physical-reader interface. It has no JSON parsing or
   NumPy loading code. In a generic implementation, the caller supplies the seed
   assets (here, `batches/*.json`) rather than hardcoding a file format in the
   grouping layer. Inspection must reflect the logical fields, including scalar
   fields from JSON and array fields from NPZ.
3. **LogicalFileReader.** Compose the collection and strategy; provide logical
   `load`, `get`, and `items` for a locator. Pass selections to the physical
   reader after resolving the source. An identity strategy maps one physical
   file to one logical file with unchanged field paths.
4. **Store and pipe.** Both use the logical reader and locator. A mapping returned
   by a pipe still only changes indexed data, as today. Regrouping happens before
   the pipe. Explicit physical access can take an asset path and physical key;
   physical access needs to identify which of a row's three assets is intended.

The generated expected-mapping document is a fixture oracle, not a decision to
persist expanded per-field mappings in the reference table. The strategy remains
user-supplied runtime code. Persist a compact logical locator such as
`{"asset": "batches/batch_a.json", "run_key": "run2"}` and reattach the configured
logical reader after loading. A runtime binding must remain usable by filtered
DataFrames and extracted rows.

Physical dependencies are distinct from logical identity. The grouping strategy
must report all assets used by a logical file. They can be persisted as relative
paths for incremental updates, without serializing the strategy. Keep file
fingerprints once per physical asset. A changed shared NPZ invalidates every
dependent logical row; a changed JSON triggers regrouping its runs and refreshing
their dependencies. Deleting or changing referenced assets must not silently
reuse stale rows. Changing grouping configuration requires rebuilding, too.

The collection should share physical reads during a reference-building operation.
In particular, multiple runs from one JSON should not cause repeated JSON loads.
Share source access, but keep pipe mutations private to each logical row.

## Acceptance cases for the implementation

- Eleven physical assets produce exactly six logical rows.
- Each row reads its own JSON run and its own signal, and resolves its shared
  structure correctly across batches.
- JSON metadata and file-link fields do not enter the logical view or index.
- Selecting `run2` from different batches does not confuse their logical locators.
- Pipes receive one complete logical row, including fields from all three assets.
- Filtering rows does not change their source mappings.
- Saving and reloading, then supplying the reader, preserves logical access.
- An update to `structures/small.npz` invalidates its four dependent rows.
- An update to `observables/batch_b_run2.npz` invalidates only `batch_b:run2`.
- Adding or removing a JSON run updates the logical-file inventory.
- The identity strategy preserves the existing one-file/one-row behavior.

The current NPZ reader selects after loading an array member. Forwarding a slice
to it preserves its existing behavior; it does not make NPZ slicing equivalent
to direct HDF5 dataset slicing.
