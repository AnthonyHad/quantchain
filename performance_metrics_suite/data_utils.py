import os
import json
import pandas as pd

def load_raw_equity(path: str,
                    ts_col: str = "timestamp",
                    eq_col: str = "portfolio_value") -> pd.Series:
    """
    Read CSV (or JSON) at `path`, parse the timestamp column, sort by time,
    keep only the equity column, rename it to 'equity', and return as a Series.
    """
    df = (
        pd.read_csv(path, parse_dates=[ts_col])
          .set_index(ts_col)
          .sort_index()[[eq_col]]
          .rename(columns={eq_col: "equity"})
    )
    return df["equity"]

def validate_equity_curve(eq: pd.Series, freq_expected: str = "D") -> None:
    """
    Assert that:
      - Index is increasing
      - Frequency matches freq_expected
      - Values are floats
      - No NaNs
    """
    assert eq.index.is_monotonic_increasing, "Timestamps must be sorted!"
    freq = pd.infer_freq(eq.index)
    assert freq == freq_expected, f"Expected frequency {freq_expected}, got {freq}"
    assert eq.dtype in ("float32", "float64"), f"Equity dtype must be float, got {eq.dtype}"
    assert not eq.isna().any(), "NaN values found in equity curve!"

def save_clean_run(eq: pd.Series, meta: dict, run_name: str) -> None:
    """
    Save cleaned equity and metadata under runs/{run_name}/
    - equity.feather for the Series
    - meta.json for the metadata dict
    """
    folder = os.path.join("runs", run_name)
    os.makedirs(folder, exist_ok=True)
    eq.to_feather(os.path.join(folder, "equity.feather"))
    with open(os.path.join(folder, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)