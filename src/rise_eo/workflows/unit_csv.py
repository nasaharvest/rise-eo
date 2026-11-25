"""Read `input.csv` and the shard CSVs with the id column kept as text."""
from __future__ import annotations

import pandas as pd

COL_ID = "id"


def read_unit_csv(f, *id_cols: str) -> pd.DataFrame:
    """`pd.read_csv` with `id` (and any `id_cols`) as `str`, so `"001"` does not come back as `1`."""
    return pd.read_csv(f, dtype={c: str for c in (COL_ID, *id_cols)})
