# c5_data.py
# Dataset builders for Task C.5:
#  - Univariate multi-step (X: close, y: next k closes)
#  - Multivariate single-step (X: OHLCAV window, y: next close)
#  - Multivariate multi-step (X: OHLCAV window, y: next k closes)
#
# The loader below is robust to yfinance MultiIndex columns and normalizes
# column names to a consistent schema.

from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------------------------------- #
# Data loading (MultiIndex-safe)                                              #
# --------------------------------------------------------------------------- #

def load_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download daily OHLCV with yfinance, handle possible MultiIndex columns,
    and standardize column names to:
        ["Open", "High", "Low", "Close", "Adj close", "Volume"]

    Notes:
    - auto_adjust=True puts adjusted prices into 'Close' (convenient for targets)
    - group_by="column" usually avoids MultiIndex; if not, we collapse it.
    """
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        group_by="column",   # try to avoid MultiIndex
        progress=False,
    )

    # If MultiIndex (e.g., ('Close','CBA.AX')), collapse sensibly:
    if isinstance(df.columns, pd.MultiIndex):
        try:
            # If the last level contains the ticker symbol, select that slice
            if ticker in df.columns.get_level_values(-1):
                # Keep price field names as columns (Close/Open/...), drop ticker level
                df = df.xs(ticker, axis=1, level=-1)
            else:
                # Fallback: keep the first element of each tuple ('Close', ...) -> 'Close'
                df = df.copy()
                df.columns = [c[0] if isinstance(c, tuple) else str(c) for c in df.columns]
        except Exception:
            df = df.copy()
            df.columns = [c[0] if isinstance(c, tuple) else str(c) for c in df.columns]

    # Standardize to title case, then rename "Adj Close" -> "Adj close"
    df = df.copy()
    df.columns = [str(c).strip().title() for c in df.columns]
    if "Adj Close" not in df.columns and "Close" in df.columns:
        # If there is no explicit Adj Close, mirror Close (reasonable with auto_adjust=True)
        df["Adj Close"] = df["Close"]

    # Ensure all expected columns exist
    expected = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan

    # Final column names and order (project expects "Adj close" lower-cased 'c')
    df = df.rename(columns={"Adj Close": "Adj close"})
    df = df[["Open", "High", "Low", "Close", "Adj close", "Volume"]]

    # Clean and sort
    df = df.dropna(how="any").sort_index()
    return df

# --------------------------------------------------------------------------- #
# Feature scalers                                                             #
# --------------------------------------------------------------------------- #

def fit_feature_scalers(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, MinMaxScaler]:
    """Fit an independent MinMax scaler for each feature (0..1)."""
    scalers: Dict[str, MinMaxScaler] = {}
    for c in feature_cols:
        s = MinMaxScaler()
        arr = df[c].astype("float32").values.reshape(-1, 1)
        s.fit(arr)
        scalers[c] = s
    return scalers

def apply_feature_scalers(df: pd.DataFrame, scalers: Dict[str, MinMaxScaler]) -> pd.DataFrame:
    """Apply previously fitted feature scalers column-wise."""
    out = df.copy()
    for c, sc in scalers.items():
        out[c] = sc.transform(out[c].astype("float32").values.reshape(-1, 1))
    return out

# --------------------------------------------------------------------------- #
# Windowing helpers                                                           #
# --------------------------------------------------------------------------- #

def build_univariate_multistep(
    series: pd.Series,
    lookback: int,
    horizon: int,
    scaler: Optional[MinMaxScaler] = None,
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Univariate, multi-step sequence:
      X shape: (samples, lookback, 1)
      y shape: (samples, horizon)  # next k values
    """
    s = series.astype("float32").values.reshape(-1, 1)
    if scaler is None:
        scaler = MinMaxScaler()
        s = scaler.fit_transform(s)
    else:
        s = scaler.transform(s)

    X, y = [], []
    for i in range(lookback, len(s) - horizon + 1):
        X.append(s[i - lookback:i, 0])   # past 'lookback' points
        y.append(s[i:i + horizon, 0])    # next 'horizon' points
    X = np.array(X, dtype=np.float32).reshape(-1, lookback, 1)
    y = np.array(y, dtype=np.float32)
    return X, y, scaler

def build_multivariate_singlestep(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    lookback: int,
    horizon: int = 1,
    scalers: Optional[Dict[str, MinMaxScaler]] = None,
    target_scaler: Optional[MinMaxScaler] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, MinMaxScaler], MinMaxScaler]:
    """
    Multivariate, single-step:
      X shape: (samples, lookback, n_features)
      y shape: (samples, 1) for horizon=1
    """
    if scalers is None:
        scalers = fit_feature_scalers(df, feature_cols)
    df_scaled = apply_feature_scalers(df[feature_cols], scalers)

    t = df[target_col].astype("float32").values.reshape(-1, 1)
    if target_scaler is None:
        target_scaler = MinMaxScaler()
        t = target_scaler.fit_transform(t)
    else:
        t = target_scaler.transform(t)

    X, y = [], []
    for i in range(lookback, len(df_scaled) - horizon + 1):
        X.append(df_scaled.iloc[i - lookback:i].values)
        y.append(t[i + horizon - 1, 0])   # the next-step target (for horizon=1)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    return X, y, scalers, target_scaler

def build_multivariate_multistep(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    lookback: int,
    horizon: int,
    scalers: Optional[Dict[str, MinMaxScaler]] = None,
    target_scaler: Optional[MinMaxScaler] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, MinMaxScaler], MinMaxScaler]:
    """
    Multivariate, multi-step:
      X shape: (samples, lookback, n_features)
      y shape: (samples, horizon)
    """
    if scalers is None:
        scalers = fit_feature_scalers(df, feature_cols)
    df_scaled = apply_feature_scalers(df[feature_cols], scalers)

    t = df[target_col].astype("float32").values.reshape(-1, 1)
    if target_scaler is None:
        target_scaler = MinMaxScaler()
        t = target_scaler.fit_transform(t)
    else:
        t = target_scaler.transform(t)

    X, Y = [], []
    for i in range(lookback, len(df_scaled) - horizon + 1):
        X.append(df_scaled.iloc[i - lookback:i].values)
        Y.append(t[i:i + horizon, 0])     # next k targets
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    return X, Y, scalers, target_scaler

# --------------------------------------------------------------------------- #
# Splits                                                                       #
# --------------------------------------------------------------------------- #

def split_by_ratio(X: np.ndarray, y: np.ndarray, train=0.7, val=0.15):
    """
    Date-ordered ratio split: [0:train) -> train, [train:train+val) -> val, rest -> test.
    """
    n = len(X)
    n_tr = int(n * train)
    n_va = int(n * val)
    Xtr, ytr = X[:n_tr], y[:n_tr]
    Xva, yva = X[n_tr:n_tr+n_va], y[n_tr:n_tr+n_va]
    Xte, yte = X[n_tr+n_va:], y[n_tr+n_va:]
    return Xtr, ytr, Xva, yva, Xte, yte
