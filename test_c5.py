# test_c5.py
# Task C.5 experiments:
#   A) Univariate multi-step (predict next k closes directly)
#   B) Multivariate single-step (OHLCAV -> next close)
#   C) Multivariate multi-step (OHLCAV -> next k closes)
#
# Saves a summary CSV with scaled and inverse-scaled metrics:
#   outputs/c5_results.csv

from __future__ import annotations
import os, math, time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Optional: silence most TF warnings (not errors)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from c5_data import (
    load_ohlcv, build_univariate_multistep,
    build_multivariate_singlestep, build_multivariate_multistep,
    split_by_ratio
)

# --------------------------- config ----------------------------------------- #
TICKER   = "CBA.AX"
START    = "2020-01-01"
END      = "2024-12-31"
LOOKBACK = 60
HORIZON  = 5      # predict next k days
EPOCHS   = 8
BATCH    = 32

OUT = os.path.join(os.getcwd(), "outputs")
os.makedirs(OUT, exist_ok=True)
CSV_PATH = os.path.join(OUT, "c5_results.csv")

# -------------------------- model factory ----------------------------------- #
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN, Bidirectional

def make_model(input_shape, layers, out_dim=1, loss="mse", optimizer="adam"):
    """
    Build a small Sequential RNN stack from a list of layer specs, then Dense(out_dim).
    Each spec supports keys:
      - type: "LSTM" | "GRU" | "RNN"
      - units: int
      - dropout: float in [0,1] (optional)
      - bidirectional: bool (optional)
      - return_sequences: bool (optional; auto True for all but last)
    """
    RNN = {"LSTM": LSTM, "GRU": GRU, "RNN": SimpleRNN}
    m = Sequential()
    for i, spec in enumerate(layers):
        Cell = RNN.get(str(spec.get("type", "LSTM")).upper(), LSTM)
        units = int(spec.get("units", 64))
        rs    = bool(spec.get("return_sequences", i < len(layers)-1))
        drop  = float(spec.get("dropout", 0.0))
        bi    = bool(spec.get("bidirectional", False))
        layer = Cell(units, return_sequences=rs, input_shape=input_shape if i == 0 else None)
        m.add(Bidirectional(layer) if bi else layer)
        if drop > 0:
            m.add(Dropout(drop))
    m.add(Dense(out_dim, activation="linear"))
    m.compile(loss=loss, optimizer=optimizer, metrics=["mae"])
    return m

# --------------------------- metrics ---------------------------------------- #

def _scaled_metrics(y_true, y_pred):
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    return mae, rmse

def _inverse_metrics(y_true_scaled, y_pred_scaled, scaler):
    """
    Inverse-transform scaled targets/preds back to original currency before scoring.
    """
    y_true = scaler.inverse_transform(y_true_scaled.reshape(-1,1)).ravel()
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    return mae, rmse

def evaluate_multistep(y_true_s, y_pred_s, scaler):
    """
    y_* shape: (samples, horizon). Returns a dict with overall (flattened)
    and per-step MAE/RMSE in both scaled and inverse (raw) spaces.
    """
    res = {}
    # overall (flatten across steps)
    mae_s, rmse_s = _scaled_metrics(y_true_s.ravel(), y_pred_s.ravel())
    mae_r, rmse_r = _inverse_metrics(y_true_s.ravel(), y_pred_s.ravel(), scaler)
    res.update(dict(
        overall_mae_scaled=mae_s, overall_rmse_scaled=rmse_s,
        overall_mae_raw=mae_r, overall_rmse_raw=rmse_r
    ))
    # per-step
    k = y_true_s.shape[1]
    for h in range(k):
        ms, rs = _scaled_metrics(y_true_s[:, h], y_pred_s[:, h])
        mr, rr = _inverse_metrics(y_true_s[:, h], y_pred_s[:, h], scaler)
        res[f"step{h+1}_mae_scaled"]  = ms
        res[f"step{h+1}_rmse_scaled"] = rs
        res[f"step{h+1}_mae_raw"]     = mr
        res[f"step{h+1}_rmse_raw"]    = rr
    return res

# ----------------------------- main ----------------------------------------- #

def main():
    # 1) Load data (robust to MultiIndex) and define features
    df = load_ohlcv(TICKER, START, END)
    feats = ["Open", "High", "Low", "Close", "Adj close", "Volume"]

    results = []

    # A) Univariate multi-step: X = Close only, y = next HORIZON closes
    X, Y, close_scaler = build_univariate_multistep(
        df["Close"], lookback=LOOKBACK, horizon=HORIZON
    )
    Xtr, Ytr, Xva, Yva, Xte, Yte = split_by_ratio(X, Y)
    model = make_model(
        (LOOKBACK, 1),
        layers=[{"type": "GRU", "units": 64, "dropout": 0.2},
                {"type": "GRU", "units": 32, "dropout": 0.2}],
        out_dim=HORIZON,
    )
    t0 = time.time()
    model.fit(Xtr, Ytr, validation_data=(Xva, Yva), epochs=EPOCHS, batch_size=BATCH, verbose=0)
    sec = time.time() - t0
    Yhat = model.predict(Xte, verbose=0)
    resA = dict(
        experiment="A_univariate_multistep",
        lookback=LOOKBACK, horizon=HORIZON,
        layers="GRU(64)->GRU(32)", train_sec=round(sec, 2)
    )
    resA.update(evaluate_multistep(Yte, Yhat, close_scaler))
    results.append(resA)

    # B) Multivariate single-step: X = OHLCAV window, y = next Close
    X, y, scalers, tsc = build_multivariate_singlestep(
        df, feats, target_col="Close", lookback=LOOKBACK, horizon=1
    )
    Xtr, ytr, Xva, yva, Xte, yte = split_by_ratio(X, y)
    model = make_model(
        (LOOKBACK, X.shape[2]),
        layers=[{"type": "LSTM", "units": 64, "dropout": 0.2},
                {"type": "LSTM", "units": 32, "dropout": 0.2}],
        out_dim=1,
    )
    t0 = time.time()
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=EPOCHS, batch_size=BATCH, verbose=0)
    sec = time.time() - t0
    yhat = model.predict(Xte, verbose=0).reshape(-1, 1)
    mae_s, rmse_s = _scaled_metrics(yte, yhat)
    mae_r, rmse_r = _inverse_metrics(yte, yhat, tsc)
    resB = dict(
        experiment="B_multivariate_singlestep",
        lookback=LOOKBACK, features=len(feats),
        layers="LSTM(64)->LSTM(32)", train_sec=round(sec, 2),
        overall_mae_scaled=float(mae_s), overall_rmse_scaled=float(rmse_s),
        overall_mae_raw=float(mae_r),    overall_rmse_raw=float(rmse_r),
    )
    results.append(resB)

    # C) Multivariate multi-step: X = OHLCAV window, y = next HORIZON closes
    X, Y, scalers, tsc = build_multivariate_multistep(
        df, feats, target_col="Close", lookback=LOOKBACK, horizon=HORIZON
    )
    Xtr, Ytr, Xva, Yva, Xte, Yte = split_by_ratio(X, Y)
    model = make_model(
        (LOOKBACK, X.shape[2]),
        layers=[{"type": "LSTM", "units": 64, "dropout": 0.2},
                {"type": "GRU",  "units": 32, "dropout": 0.2}],
        out_dim=HORIZON,
    )
    t0 = time.time()
    model.fit(Xtr, Ytr, validation_data=(Xva, Yva), epochs=EPOCHS, batch_size=BATCH, verbose=0)
    sec = time.time() - t0
    Yhat = model.predict(Xte, verbose=0)
    resC = dict(
        experiment="C_multivariate_multistep",
        lookback=LOOKBACK, horizon=HORIZON, features=len(feats),
        layers="LSTM(64)->GRU(32)", train_sec=round(sec, 2)
    )
    resC.update(evaluate_multistep(Yte, Yhat, tsc))
    results.append(resC)

    # Save and echo results
    dfres = pd.DataFrame(results)
    dfres.to_csv(CSV_PATH, index=False)
    print(f"\nSaved results -> {CSV_PATH}")
    print(dfres.to_string(index=False))

if __name__ == "__main__":
    main()
