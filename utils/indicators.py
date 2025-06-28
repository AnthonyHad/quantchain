import pandas as pd

# ---------- Simple Moving Average ----------
def sma(series: pd.Series, window: int) -> pd.Series:
    """
    Simple Moving Average (mean of last *window* values).
    Example: sma(df["close"], 20)
    """
    return series.rolling(window).mean()

# ---------- Exponential Moving Average ----------
def ema(series: pd.Series, span: int) -> pd.Series:
    """
    Exponential Moving Average with decay based on *span*.
    Example: ema(df["close"], 50)
    """
    return series.ewm(span=span, adjust=False).mean()
