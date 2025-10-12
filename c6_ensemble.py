"""
C6 — Classical + DL Ensembling (ARIMA + LSTM/GRU + Linear Stacking)
"""

from __future__ import annotations
from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

from statsmodels.tsa.statespace.sarimax import SARIMAX
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout


# ------------------------------------------------------------------------------
# 0) Data loading (Business-Day frequency to avoid statsmodels warnings)
# ------------------------------------------------------------------------------

def load_series(ticker: str = "AAPL",
                start: str = "2018-01-01",
                end: str | None = None) -> pd.Series:
    """
    Download Close prices and give the series a Business-Day ('B') frequency.
    This keeps the calendar consistent and avoids 'no frequency' warnings.

    NOTE: We set the Series name via attribute assignment (avoid rename bug).
    """
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} between {start} and {end}.")

    # Clean index and ensure uniqueness
    df.index = pd.DatetimeIndex(df.index)
    df = df[~df.index.duplicated(keep="last")]

    # Use Close; name it safely (avoid Series.rename("y") callable confusion)
    y = df["Close"].copy()
    y.name = "y"

    # Enforce Business-Day frequency; fill non-trading days forward
    y = y.asfreq("B").ffill()
    return y


# ------------------------------------------------------------------------------
# 1) Time-based split
# ------------------------------------------------------------------------------

def train_val_test_split(y: pd.Series,
                         train_ratio: float = 0.7,
                         val_ratio: float = 0.15) -> Tuple[pd.Series, pd.Series, pd.Series]:
    n = len(y)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = y.iloc[:n_train]
    val   = y.iloc[n_train:n_train + n_val]
    test  = y.iloc[n_train + n_val:]
    return train, val, test


# ------------------------------------------------------------------------------
# 2) Metrics with safe alignment
# ------------------------------------------------------------------------------

def metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """
    Align by index to avoid shape/index mismatches, then compute MAE/RMSE.
    """
    y_true_aligned, y_pred_aligned = y_true.align(y_pred, join="inner", axis=0)
    mae  = float(mean_absolute_error(y_true_aligned, y_pred_aligned))
    rmse = float(np.sqrt(mean_squared_error(y_true_aligned, y_pred_aligned)))
    return {"mae": mae, "rmse": rmse}


# ------------------------------------------------------------------------------
# 3) ARIMA (SARIMAX) — simple one-step-ahead rolling on test horizon
# ------------------------------------------------------------------------------

def fit_forecast_arima(train: pd.Series,
                       test: pd.Series,
                       order: Tuple[int, int, int] = (1, 0, 0)) -> pd.Series:
    """
    Fit SARIMAX on train, then forecast len(test) steps ahead.
    Index is set to test.index for clean alignment with ground truth.
    """
    train = train.dropna()

    model = SARIMAX(train, order=order,
                    enforce_stationarity=True,
                    enforce_invertibility=True)
    res = model.fit(disp=False, maxiter=50)

    # Direct multi-step forecast to match test horizon
    pred = res.get_forecast(steps=len(test)).predicted_mean
    pred.index = test.index
    pred.name = "arima"
    return pred


# ------------------------------------------------------------------------------
# 4) LSTM/GRU helpers
# ------------------------------------------------------------------------------

def _make_xy(series_1d: np.ndarray, lookback: int):
    """
    Turn a 1D scaled array into supervised learning (X,y) with given lookback.
    X shape -> (samples, lookback, 1)
    y shape -> (samples,)
    """
    X, y = [], []
    for i in range(lookback, len(series_1d)):
        X.append(series_1d[i - lookback:i])
        y.append(series_1d[i])
    X = np.array(X)[..., np.newaxis]
    y = np.array(y)
    return X, y


def _prep_val_or_test_sequences(history: pd.Series,
                                segment: pd.Series,
                                scaler: MinMaxScaler,
                                lookback: int):
    """
    For val/test prediction we need 'lookback' history. We stitch the last
    'lookback' points of the preceding data to the current segment, scale,
    and build sequences. We return X for the segment and the segment index.
    """
    stitch = pd.concat([history.iloc[-lookback:], segment])
    stitch_scaled = scaler.transform(stitch.values.reshape(-1, 1)).ravel()
    X_seg, _ = _make_xy(stitch_scaled, lookback)
    target_index = segment.index
    return X_seg, target_index


def _build_lstm(input_steps: int,
                units=(64, 32),
                dropout: float = 0.2) -> Sequential:
    """
    Simple stacked LSTM regressor: (LSTM->Dropout->LSTM->Dropout->Dense)
    """
    model = Sequential()
    model.add(LSTM(units[0], return_sequences=True, input_shape=(input_steps, 1)))
    model.add(Dropout(dropout))
    model.add(LSTM(units[1], return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model


def _build_gru(input_steps: int,
               units=(64, 32),
               dropout: float = 0.2) -> Sequential:
    """
    Simple stacked GRU regressor: (GRU->Dropout->GRU->Dropout->Dense)
    """
    model = Sequential()
    model.add(GRU(units[0], return_sequences=True, input_shape=(input_steps, 1)))
    model.add(Dropout(dropout))
    model.add(GRU(units[1], return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model


def fit_forecast_lstm(train: pd.Series,
                      val: pd.Series,
                      test: pd.Series,
                      lookback: int = 60,
                      epochs: int = 8,
                      batch_size: int = 32,
                      rnn_type: str = "LSTM"):
    """
    Train an LSTM/GRU on (train), then produce 1-step-ahead predictions
    aligned to (val) and (test) indices.

    Returns:
      yhat_val  : pd.Series of predictions indexed like val
      yhat_test : pd.Series of predictions indexed like test
    """
    # Scale on train only to avoid data leakage
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1)).ravel()
    X_tr, y_tr = _make_xy(train_scaled, lookback)

    # Build model
    if rnn_type.upper() == "GRU":
        model = _build_gru(lookback)
    else:
        model = _build_lstm(lookback)

    # Train
    model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size, verbose=0, shuffle=True)

    # Predict on validation
    X_val, idx_val = _prep_val_or_test_sequences(train, val, scaler, lookback)
    yhat_val = model.predict(X_val, verbose=0).ravel()
    yhat_val = scaler.inverse_transform(yhat_val.reshape(-1, 1)).ravel()
    yhat_val = pd.Series(yhat_val, index=idx_val, name="lstm")

    # Predict on test (history includes train + val)
    train_plus_val = pd.concat([train, val])
    X_test, idx_test = _prep_val_or_test_sequences(train_plus_val, test, scaler, lookback)
    yhat_test = model.predict(X_test, verbose=0).ravel()
    yhat_test = scaler.inverse_transform(yhat_test.reshape(-1, 1)).ravel()
    yhat_test = pd.Series(yhat_test, index=idx_test, name="lstm")

    return yhat_val, yhat_test


# ------------------------------------------------------------------------------
# 5) Linear stacking (learn weights on validation, apply to test)
# ------------------------------------------------------------------------------

def stacking_linear(y_true_val: pd.Series,
                    y_arima_val: pd.Series,
                    y_lstm_val: pd.Series,
                    y_arima_test: pd.Series,
                    y_lstm_test: pd.Series) -> pd.Series:
    """
    Fit a linear regressor on validation targets vs. (ARIMA, LSTM) preds,
    then combine ARIMA/LSTM test predictions with the learned weights.

    Returns:
      y_stack: pd.Series of stacked predictions on the test index.
    """
    # Align validation
    y_true_val, y_arima_val = y_true_val.align(y_arima_val, join="inner", axis=0)
    y_true_val, y_lstm_val  = y_true_val.align(y_lstm_val,  join="inner", axis=0)

    X_val = pd.DataFrame({
        "arima": y_arima_val,
        "lstm":  y_lstm_val
    }, index=y_true_val.index)

    reg = LinearRegression()
    reg.fit(X_val.values, y_true_val.values)

    # Align test
    y_arima_test, y_lstm_test = y_arima_test.align(y_lstm_test, join="inner", axis=0)
    X_test = pd.DataFrame({
        "arima": y_arima_test,
        "lstm":  y_lstm_test
    }, index=y_arima_test.index)

    # Predict and FLATTEN to 1-D to satisfy pd.Series
    y_stack = reg.predict(X_test.values).ravel()
    y_stack = pd.Series(y_stack, index=X_test.index, name="stack")
    return y_stack
