# Pickled objects

A pickle can hold objects of any class, such as a pandas DataFrame or a class
only the writing script defines. lazyfields reads pickles as data: it never
imports those classes or runs their code. Each such object becomes a record of
what it is made of, a group with:

- `type`: the class or function the pickle names, such as `pandas.DataFrame`;
- `args`: the arguments it is built from, by position (`0`, `1`, ...);
- `state`, and `items` or `kwargs`, when the pickle gives the object those.

Data inside records stays readable: NumPy arrays, numbers, strings and dates
are built as usual, so a DataFrame's column arrays can be read and sliced.

From the repository root, with Tk available:

```bash
python3 example/pickled_objects/generate_dataset.py
python3 example/pickled_objects/browse.py
```

`generate_dataset.py` writes `data/results.pkl`: a DataFrame `table`, a `run`
of the script's own `Run` class, a `trajectory` array, and a `label`.
`browse.py` browses the data directory, where `results.pkl` is a file node
like any other, and mounts the file a second time as `readable`.

`results.pkl` shows it as recorded. `results.pkl/run` is a record of type
`__main__.Run`, with its fields under `state`. `results.pkl/table` is pandas'
internal layout: `state/_mgr` is the DataFrame's block manager, whose `args/0`
are blocks of column values and `args/1` the column and row labels:

```text
table/state/_mgr/args/0/0/args/0    float64 (2, 3)   temperature and energy
table/state/_mgr/args/0/1/args/0    int64 (1, 3)     steps
table/state/_mgr/args/1/0/...       column labels
```

`readable` is the same file under another name, so the browser marks it
relocated (`↪`), with `table` read through a view.
`browse.py` registers `columns`, which rearranges a DataFrame record into its
`index` and one array per column, from the record's parts only, and marks
`table` with its name:

```python
register_view("dataframe-columns", View(columns))
FileSystem.from_spec({"mount": [{"glob": "*.pkl"}], "children": {
    "readable": {"path": "results.pkl", "children": {"table": {"attrs": {"view": "dataframe-columns"}}}},
}}, relative_to=DATA_DIR)
```

```text
readable/table/index          ['s1', 's2', 's3']
readable/table/temperature    [300.0, 325.0, 350.0]
readable/table/energy         [-1.25, -1.1, -0.95]
readable/table/steps          [1000, 2000, 4000]
```

`columns` follows the pickled layout of pandas 2 and 3, including string and
date columns and default `RangeIndex` labels; other versions may lay a
DataFrame out differently. It is example code, not part of lazyfields.
