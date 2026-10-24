#%%
from pathlib import Path

import pandas as pd
import lazyfields as lf
from lazyfields.readers.csv import CSV_WITH_HEADER


lf.register_reader(CSV_WITH_HEADER(skiprows=2))
reference_df = pd.read_pickle(Path(__file__).parent / "reference_table.pkl")
#%%
print(reference_df.iloc[0].store["signal", 1:-1])

# %%
