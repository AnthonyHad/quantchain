import numpy as np
import pandas as pd

def simple_returns(equity: pd.Series) -> pd.Series:
    """
    Compute simple percent-change returns from an equity series.
    Drops the first NaN and returns a returns series indexed like the input.
    """
    return equity.pct_change().dropna()

def sharpe_ratio(returns: pd.Series,
                 rf_rate: float,
                 periods_per_year: int = 252) -> float:
    """
    Annualised Sharpe Ratio assuming simple returns.

    - returns: daily (or whatever freq) returns series
    - rf_rate: annual risk-free rate (as a decimal, e.g. 0.05)
    - periods_per_year: number of periods per year (252 for daily)
    """
    # convert annual rf rate to per-period
    rf_per_period = rf_rate / periods_per_year
    excess = returns - rf_per_period
    return excess.mean() / excess.std() * np.sqrt(periods_per_year)

def max_drawdown(equity: pd.Series) -> float:
    """
    Maximum drawdown of an equity curve: the worst peak-to-trough decline.
    Returns a negative number (e.g. -0.25 for a 25% drawdown).
    """
    # Running maximum
    running_max = equity.cummax()
    # Drawdown at each point
    drawdowns = equity / running_max - 1
    return drawdowns.min()