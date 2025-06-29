"""
Vectorised SMA Crossover Back-test
Run:  python 01_vector_backtester/backtest.py
"""


import matplotlib.pyplot as plt
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.indicators import sma, ema   # we’ll use SMA first
from performance_metrics_suite.data_utils import load_raw_equity, validate_equity_curve
from performance_metrics_suite.performance_metrics import simple_returns, sharpe_ratio, max_drawdown

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

    # also compute buy‐and‐hold equity curve
    df["hold_eq"] = (1 + df["ret"]).cumprod()

    # drop NaNs before returning
    df.dropna(subset=["equity"], inplace=True)

    return df

# -------------- script entry point -------------
if __name__ == "__main__":
    for asset in ASSETS:
        plt.figure()
        df   = load_price(asset)
        res  = run_backtest(df)

         # ——— PERFORMANCE METRICS ———
        eq = res["equity"]
        validate_equity_curve(eq)
        rts = simple_returns(eq)
        print(f"{asset.upper()} Sharpe: {sharpe_ratio(rts, rf_rate=0.05):.2f}")
        print(f"{asset.upper()} Max drawdown: {max_drawdown(eq):.2%}")
        # ——————————————————————————————

        final = res["equity"].iloc[-1]
        # Plot buy‐and‐hold vs SMA‐crossover equity
        eq_df = res[["hold_eq", "equity"]]
        ax = eq_df.plot(
            title   = f"{asset.upper()} SMA {FAST}/{SLOW}",
            color   = ["blue", "orange"]   # hold_eq in blue, equity in orange
        )
        ax.legend(["Buy & Hold", "SMA Crossover"])
        ax.set_ylabel("Growth Factor")
        plt.savefig(f"01_vector_backtester/{asset}_equity.png")
        plt.close()
        print(f"{asset.upper():<3} final equity: {final:.2f}×")
