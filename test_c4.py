# test_c4.py
# Runs a few small experiments using the model factory.
# Produces a CSV of results in ./outputs/c4_results.csv

import os, time, math
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from c4_model import make_model

TICKER = "CBA.AX"
START  = "2020-01-01"
END    = "2024-12-31"
LOOKBACK = 60  # timesteps (same spirit as v0.1)
EPOCHS   = 8   # keep small to run quickly at first
BATCH    = 32

OUTDIR = os.path.join(os.getcwd(), "outputs"); os.makedirs(OUTDIR, exist_ok=True)
RESULTS_CSV = os.path.join(OUTDIR, "c4_results.csv")

def load_series(ticker=TICKER, start=START, end=END):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    # Prefer 'Close' (auto_adjust gives adj close into 'Close')
    s = df["Close"].dropna().astype("float32")
    return s

def to_supervised(series: pd.Series, lookback=LOOKBACK):
    """
    Build X,y for univariate sequence prediction:
    X: (samples, lookback, 1), y: next value
    """
    arr = series.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    arr_scaled = scaler.fit_transform(arr)

    X, y = [], []
    for i in range(lookback, len(arr_scaled)):
        X.append(arr_scaled[i - lookback:i, 0])
        y.append(arr_scaled[i, 0])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # add feature dimension
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, scaler

def train_val_test_split(X, y, train=0.7, val=0.15):
    n = len(X)
    n_train = int(n * train)
    n_val   = int(n * val)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val,   y_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test,  y_test  = X[n_train+n_val:], y[n_train+n_val:]
    return X_train, y_train, X_val, y_val, X_test, y_test

def evaluate(model, X_test, y_test):
    # y_pred in scaled space → compare directly in scaled MAE/MSE for fairness across runs
    y_pred = model.predict(X_test, verbose=0).reshape(-1)
    y_true = y_test.reshape(-1)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    return mae, rmse

def main():
    s = load_series()
    X, y, scaler = to_supervised(s, LOOKBACK)
    Xtr, ytr, Xval, yval, Xte, yte = train_val_test_split(X, y)

    experiments = [
        {
            "name": "LSTM_64x2",
            "layers": [
                {"type": "LSTM", "units": 64, "dropout": 0.2},
                {"type": "LSTM", "units": 32, "dropout": 0.2},
            ],
            "epochs": EPOCHS, "batch": BATCH
        },
        {
            "name": "GRU_64x2",
            "layers": [
                {"type": "GRU", "units": 64, "dropout": 0.2},
                {"type": "GRU", "units": 32, "dropout": 0.2},
            ],
            "epochs": EPOCHS, "batch": BATCH
        },
        {
            "name": "RNN_64_bidir",
            "layers": [
                {"type": "RNN", "units": 64, "dropout": 0.1, "bidirectional": True},
            ],
            "epochs": EPOCHS, "batch": BATCH
        },
    ]

    rows = []
    for exp in experiments:
        model = make_model(input_shape=(LOOKBACK, 1), layers=exp["layers"])
        t0 = time.time()
        history = model.fit(
            Xtr, ytr,
            validation_data=(Xval, yval),
            epochs=exp["epochs"],
            batch_size=exp["batch"],
            verbose=0,
        )
        train_sec = time.time() - t0
        mae, rmse = evaluate(model, Xte, yte)
        rows.append({
            "name": exp["name"],
            "layers": exp["layers"],
            "epochs": exp["epochs"],
            "batch": exp["batch"],
            "val_last": float(history.history["val_loss"][-1]),
            "test_mae_scaled": mae,
            "test_rmse_scaled": rmse,
            "train_time_s": round(train_sec, 2),
        })
        print(f"{exp['name']}: test MAE={mae:.4f} (scaled), RMSE={rmse:.4f} | time={train_sec:.1f}s")

    pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved results → {RESULTS_CSV}")

if __name__ == "__main__":
    main()
