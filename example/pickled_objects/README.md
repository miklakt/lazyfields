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

`browse.py` mounts the file a second time as `readable` with a trusted reader:

```python
trusted = pickle_reader(trust=["pandas", "matplotlib"])
FileSystem.from_spec({"mount": [{"glob": "*.pkl"}], "children": {
    "readable": {"path": "results.pkl", "reader": "trusted-pickle"},
}}, relative_to=DATA_DIR, readers={"trusted-pickle": trusted})
```

Under `readable`, pandas and matplotlib are trusted, so `table` is a real
`DataFrame` opened by the Table viewer, and `plot` is a real Figure handled by
the existing Figure viewer. The default `results.pkl` mount remains inert.
