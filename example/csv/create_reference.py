from pathlib import Path

import lazyfields as lf
from lazyfields.extra_readers import CSV_WITH_HEADER


root = Path(__file__).parent
reference_path = root / "reference_table.pkl"
lf.register_reader(CSV_WITH_HEADER(skiprows=2))
lf.create_reference_table(root / "data", relative_to=reference_path).to_pickle(reference_path)
