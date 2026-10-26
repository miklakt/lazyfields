from pathlib import Path

import lazyfields as lf
from lazyfields.readers.csv import CSV_WITH_HEADER


root = Path(__file__).parent
data_dir = root / "data"
reference_path = root / "reference_table.pkl"
lf.register_reader(CSV_WITH_HEADER(skiprows=2))
lf.create_reference_table(data_dir).to_pickle(reference_path)
