"""Open a pickled reference table in the table explorer."""

import sys

from .table import explore_table

if len(sys.argv) != 2:
    sys.exit("usage: python -m lazyfields.explorer REFERENCE_TABLE.pkl")
explore_table(sys.argv[1])
