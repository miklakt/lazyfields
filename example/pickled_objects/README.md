# Pickled objects

A pickle can hold objects of any class, such as a pandas DataFrame or a class
only the writing script defines. lazyfields reads pickles as data: it never
imports those classes or runs their code, unless its pickle reader explicitly
trusts that class (see below). An untrusted object remains an inert record.

From the repository root, with Tk available:

```bash
python3 example/pickled_objects/generate_dataset.py
python3 example/pickled_objects/browse.py
```

`generate_dataset.py` writes `data/results.pkl`: a DataFrame `table`, a `run`
of the script's own `Run` class, a `trajectory` array, a `label`, and `plot`,
a matplotlib `Figure`. `table`, `run`, and `plot` are inert records by default:
nothing imports pandas, the script's own classes, or matplotlib to read them.
`trajectory` and `label` are plain data (a NumPy array and a string), so they
read, slice, and browse exactly as any other in-memory array or string would.

## A trusted type is just a value

Trust belongs only to `pickle_reader`. Trusting `"matplotlib"` makes every
class under that package unpickle for real, not just `Figure`: a complex
object nests many classes in its pickled state. `trust` also takes a class,
or a `(module, name)` pair, to trust one global exactly.

Trusting a prefix runs that library's own unpickling code on the file, the
same as `pickle.load` would for that code. Configure it only for pickle files
you trust.

`browse.py` does the same for `table`, mounting the file a second time as
`readable` with `table` read through a view:

```python
def columns(table):
    """A DataFrame's index and columns, as plain values; ordinary pandas code once pandas is trusted."""
    return dict(table.items()) | {"index": list(table.index)}

register_view("dataframe-columns", View(columns))
trusted = pickle_reader(trust=["pandas", "matplotlib"])
FileSystem.from_spec({"mount": [{"glob": "*.pkl"}], "children": {
    "readable": {"path": "results.pkl", "reader": "trusted-pickle",
                 "children": {"table": {"attrs": {"view": "dataframe-columns"}}}},
}}, relative_to=DATA_DIR, readers={"trusted-pickle": trusted})
```

```text
readable/table/index          ['s1', 's2', 's3']
readable/table/temperature    [300.0, 325.0, 350.0]
readable/table/energy         [-1.25, -1.1, -0.95]
readable/table/steps          [1000, 2000, 4000]
```

Under `readable`, pandas and matplotlib are trusted, so `table` is a real
`DataFrame`, `plot` is a real Figure handled by the existing Figure viewer,
and `columns` is ordinary pandas code - no hand-parsing of pandas' internal
pickle layout, and nothing to maintain across pandas versions.
