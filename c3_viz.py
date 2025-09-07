# c3_viz.py  (MultiIndex-safe)
from __future__ import annotations
import warnings
from typing import Iterable, Optional
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------- helpers -----------------------------------------------------------

def _norm(s) -> str:
    """Normalize a name: lowercased, no spaces."""
    return str(s).strip().lower().replace(" ", "")

def _all_levels_as_names(col) -> set[str]:
    """
    Return a set of normalized names contained in a column identifier.
    Works for both single columns ('Close') and MultiIndex tuples
    (e.g. ('Close','AAPL')).
    """
    if isinstance(col, tuple):
        return {_norm(x) for x in col}
    return {_norm(col)}

def _find_col(df: pd.DataFrame, candidates: set[str]):
    """
    Find the first column in df whose *any* level matches one of 'candidates'
    (case/space-insensitive). Return the original column key.
    """
    for c in df.columns:
        names = _all_levels_as_names(c)
        if names & candidates:
            return c
    return None

def _ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure we have a DataFrame with columns ['Open','High','Low','Close'].
    Accepts yfinance DataFrames with single or MultiIndex columns.
    Prefers 'Close' but falls back to 'Adj Close' if necessary.
    """
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    c_open  = _find_col(df, {"open"})
    c_high  = _find_col(df, {"high"})
    c_low   = _find_col(df, {"low"})
    c_close = _find_col(df, {"close"}) or _find_col(df, {"adjclose", "adj_close"})

    missing = []
    if c_open is None:  missing.append("open")
    if c_high is None:  missing.append("high")
    if c_low is None:   missing.append("low")
    if c_close is None: missing.append("close/adjclose")
    if missing:
        raise KeyError(
            f"Missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    out = df[[c_open, c_high, c_low, c_close]].copy()
    out.columns = ["Open", "High", "Low", "Close"]

    # Ensure DatetimeIndex for mplfinance
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
        out = out[out.index.notna()]

    return out

def _resample_ohlc(ohlc: pd.DataFrame, n: int) -> pd.DataFrame:
    """Resample OHLC to n-day bars (calendar days)."""
    if not n or n <= 1:
        return ohlc
    rule = f"{n}D"
    return (
        ohlc.resample(rule)
            .agg({"Open":"first","High":"max","Low":"min","Close":"last"})
            .dropna(how="any")
    )

# ---------- public API --------------------------------------------------------

def plot_candlestick(
    df: pd.DataFrame,
    n: int = 1,
    mav: Optional[Iterable[int]] = (10, 20),
    title: Optional[str] = None,
    savepath: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Candlestick chart.
    - df: yfinance DataFrame (single or MultiIndex columns)
    - n : resample to n-day bars (1 = daily)
    - mav: moving averages (tuple) or None
    """
    ohlc = _ensure_ohlc(df)
    ohlc = _resample_ohlc(ohlc, n)

    kwargs = dict(
        type="candle",
        style="yahoo",
        title=title or "",
        volume=False,
        mav=mav if (mav and len(mav) > 0) else None,
        show_nontrading=False,
        tight_layout=True,
    )

    if savepath:
        mpf.plot(ohlc, **kwargs, savefig=savepath)
    else:
        mpf.plot(ohlc, **kwargs)

    if not show:
        plt.close("all")

def plot_boxplot_moving_window(
    df: pd.DataFrame,
    window: int = 20,
    step: int = 5,
    column: str = "Close",
    title: Optional[str] = None,
    savepath: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Sequence of boxplots over a moving window of the chosen column.
    """
    ohlc = _ensure_ohlc(df)

    # Allow 'Adj Close' request; fall back to Close
    colkey = _find_col(df, {"adjclose", "adj_close"}) if _norm(column) in {"adjclose","adj_close"} else None
    series = pd.Series(df[colkey].values, index=ohlc.index, name="Adj Close") if colkey is not None else ohlc["Close"]

    data, labels = [], []
    idx = series.index
    for i in range(0, len(series) - window + 1, step):
        data.append(series.iloc[i:i+window].values)
        labels.append(idx[i + window - 1].strftime("%Y-%m-%d"))

    if not data:
        raise ValueError("Not enough data for the given window/step.")

    plt.figure(figsize=(12, 5))
    plt.boxplot(data, positions=range(len(data)), widths=0.6, showfliers=False, patch_artist=True)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Thin x labels to avoid clutter
    if len(labels) > 20:
        k = max(1, len(labels)//10)
        ticks = list(range(0, len(labels), k))
    else:
        ticks = list(range(len(labels)))

    plt.xticks(ticks, [labels[i] for i in ticks], rotation=45, ha="right")
    plt.ylabel(column)
    plt.title(title or f"Moving-window boxplots (window={window}, step={step})")
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()
