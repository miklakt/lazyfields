import json
from pathlib import Path

import pandas as pd

from ..readers import Reader


def CSV_WITH_HEADER(comment: str = "#", key_value_separator: str = ":", skiprows: int = 0) -> Reader:
    def metadata(path):
        values = {}
        with Path(path).open() as file:
            for _ in range(skiprows): next(file, None)
            for line in file:
                if not line.startswith(comment): break
                key, value = line[len(comment):].split(key_value_separator, 1)
                value = json.loads(value)
                if not pd.api.types.is_scalar(value): raise ValueError("CSV header values must be scalar.")
                values[key.strip()] = value
        return values

    def load(path):
        table = pd.read_csv(path, comment=comment, skiprows=skiprows)
        return {**metadata(path), **{key: table[key].to_numpy() for key in table}}

    def read(path, key, selection=None):
        values = metadata(path)
        value = values[key] if key in values else pd.read_csv(path, comment=comment, skiprows=skiprows, usecols=[key])[key].to_numpy()
        return value if selection is None else value[selection]

    def inspect(path):
        yield from ((key, True, value) for key, value in metadata(path).items())
        columns = pd.read_csv(path, comment=comment, skiprows=skiprows, nrows=0).columns
        yield from ((key, False, None) for key in columns)

    return Reader((".csv",), load, read, inspect, (KeyError, TypeError, IndexError, ValueError))
