"""
Vectorised SMA Crossover Back-test
Run:  python 01_vector_backtester/backtest.py
"""


import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from utils.indicators import sma, ema   # we’ll use SMA first

# --------- parameters you can tweak -----------
FAST = 20      # “fast” SMA days
SLOW = 50      # “slow” SMA days
ASSETS = ("btc", "eth")
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# ----------------------------------------------

def load_price(asset: str) -> pd.DataFrame:
    """Load CSV into a DataFrame indexed by timestamp."""
    df = pd.read_csv(DATA_DIR / f"{asset}_usdt_1d.csv",
                     parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df

def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """Compute signals, strategy returns, and equity curve."""
    df["fast"]   = sma(df["close"], FAST)
    df["slow"]   = sma(df["close"], SLOW)

    # signal: 1 if fast > slow else 0  (long-only)
    df["signal"] = (df["fast"] > df["slow"]).astype(int)

    # daily % returns of the asset
    df["ret"]    = df["close"].pct_change()

    # strategy return = yesterday’s signal × today’s asset return
    df["strat"]  = df["signal"].shift(1) * df["ret"]

    # cumulative product gives equity curve (start at 1.0)
    df["equity"] = (1 + df["strat"]).cumprod()

    return df

# -------------- script entry point -------------
if __name__ == "__main__":
    for asset in ASSETS:
        df   = load_price(asset)
        res  = run_backtest(df)

        final = res["equity"].iloc[-1]
        res["equity"].plot(title=f"{asset.upper()} SMA {FAST}/{SLOW}")
        plt.savefig(f"01_vector_backtester/{asset}_equity.png")
        print(f"{asset.upper():<3} final equity: {final:.2f}×")
