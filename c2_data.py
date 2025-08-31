# c2_min.py

from __future__ import annotations
import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_and_process(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    features: Optional[List[str]] = None,
    cache_dir: str = "data",
    use_cache: bool = True,
    split_method: str = "ratio",      
    split_date: Optional[str] = None, 
    test_size: float = 0.2,
    scale: bool = True,
) -> Dict[str, object]:
   
    os.makedirs(cache_dir, exist_ok=True)
    csv_path = os.path.join(cache_dir, f"{ticker}.csv")

    if use_cache and os.path.isfile(csv_path):
        df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
    else:
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
        if df is None or df.empty:
            raise RuntimeError(f"Aucune donnée pour {ticker}.")
        def flat(c): return "_".join(str(x) for x in c if x is not None) if isinstance(c, tuple) else str(c)
        cols = [flat(c) for c in df.columns]
        tkr = ticker.lower()
        cols = [c[:-len(tkr)-1] if c.lower().endswith(f"_{tkr}") else c for c in cols]
        df.columns = [c.lower().replace(" ", "") for c in cols]
        df.index.name = "Date"
        df.to_csv(csv_path)

    def flat(c): return "_".join(str(x) for x in c if x is not None) if isinstance(c, tuple) else str(c)
    cols = [flat(c) for c in df.columns]
    tkr = ticker.lower()
    cols = [c[:-len(tkr)-1] if c.lower().endswith(f"_{tkr}") else c for c in cols]
    df.columns = [c.lower().replace(" ", "") for c in cols]
    df.index.name = "Date"

    if features is None:
        base = ["open", "high", "low", "close", "adjclose", "volume"]
        features = [c for c in base if c in df.columns]
        if not features:
            raise ValueError("Aucune feature standard disponible.")
    if "close" in df.columns:
        label_col = "close"
    elif "adjclose" in df.columns:
        label_col = "adjclose"
    else:
        label_col = features[0]

    Xdf = df[features].copy().ffill().bfill().dropna()
    ydf = df[[label_col]].loc[Xdf.index]

    X_all = Xdf.values.astype(np.float32)
    y_all = ydf.values.reshape(-1).astype(np.float32)

    split_info = {"method": split_method, "test_size": test_size, "split_date": split_date, "n_total": len(y_all)}
    if split_method == "ratio":
        n = len(X_all); n_tr = int((1 - test_size) * n)
        X_train, X_test = X_all[:n_tr], X_all[n_tr:]
        y_train, y_test = y_all[:n_tr], y_all[n_tr:]
    elif split_method == "date":
        if not split_date: raise ValueError("split_date requis pour split_method='date'.")
        mask = (Xdf.index < pd.to_datetime(split_date))
        X_train, X_test = X_all[mask], X_all[~mask]
        y_train, y_test = y_all[mask], y_all[~mask]
    elif split_method == "random":
        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=test_size, shuffle=True, random_state=42)
    else:
        raise ValueError("split_method doit être 'ratio', 'date' ou 'random'.")
    split_info.update({"n_train": len(y_train), "n_test": len(y_test)})

    scalers = None
    if scale:
        scalers = {}
        Xtr = np.empty_like(X_train); Xte = np.empty_like(X_test)
        for i, col in enumerate(features):
            s = MinMaxScaler()
            Xtr[:, i:i+1] = s.fit_transform(X_train[:, i:i+1])
            Xte[:, i:i+1] = s.transform(X_test[:, i:i+1])
            scalers[col] = s
        X_train, X_test = Xtr, Xte

    return {
        "feature_columns": features,
        "label_col": label_col,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "column_scalers": scalers,
        "split_info": split_info,
        "cache_file": csv_path,
        "df": df,  
    }
