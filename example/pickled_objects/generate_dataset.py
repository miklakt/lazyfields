"""Pickle a pandas DataFrame and an object of this script's own class, beside plain data."""

import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass
class Run:
    """A class that only this script defines: reading the pickle elsewhere cannot import it."""

    temperature: float
    started: datetime
    positions: np.ndarray = field(default_factory=lambda: np.zeros((4, 3)))


def _plot():
    """A Figure: importable, but not a data type, so it comes back as a record too, never rendered live."""
    figure = Figure()
    figure.add_subplot().plot([-1.25, -1.10, -0.95])
    return figure


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame({"temperature": [300.0, 325.0, 350.0], "energy": [-1.25, -1.10, -0.95],
                          "steps": [1000, 2000, 4000]}, index=["s1", "s2", "s3"])
    results = {
        "table": table,
        "run": Run(300.0, datetime(2026, 10, 28, 9, 30), np.arange(12.0).reshape(4, 3)),
        "trajectory": np.random.default_rng(0).normal(size=(100, 3)),
        "label": "pickled objects example",
        "weights": np.linspace(0.0, 1.0, 5),
        "plot": _plot(),
        "plot_callable": _plot,
    }
    (DATA_DIR / "results.pkl").write_bytes(pickle.dumps(results))
    print(f"Created results.pkl in {DATA_DIR}")


if __name__ == "__main__":
    main()
