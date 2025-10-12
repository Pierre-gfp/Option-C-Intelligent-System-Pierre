"""
C6 — Quick driver to run ARIMA + LSTM + linear stacking.
It prints metrics for (ARIMA, LSTM, STACK) on both validation and test sets.

How to run:
    python test_c6.py
"""

from __future__ import annotations
import pandas as pd

from c6_ensemble import (
    load_series,
    train_val_test_split,
    fit_forecast_arima,
    fit_forecast_lstm,
    stacking_linear,
    metrics,
)

# ------------------------------- config ---------------------------------------
TICKER   = "AAPL"
START    = "2018-01-01"
ORDER    = (1, 0, 0)   # ARIMA(p,d,q)
LOOKBACK = 60
EPOCHS   = 8
# ------------------------------------------------------------------------------

def main():
    # 1) Data
    y = load_series(TICKER, start=START)
    train, val, test = train_val_test_split(y, train_ratio=0.7, val_ratio=0.15)

    # 2) Models
    # ARIMA (classical)
    yhat_ar_val  = fit_forecast_arima(train, val,  order=ORDER)
    yhat_ar_test = fit_forecast_arima(pd.concat([train, val]), test, order=ORDER)

    # LSTM (DL)
    yhat_lstm_val, yhat_lstm_test = fit_forecast_lstm(
        train, val, test, lookback=LOOKBACK, epochs=EPOCHS, rnn_type="LSTM"
    )

    # 3) Stacking (learn weights on val, apply to test)
    yhat_stack_test = stacking_linear(
        y_true_val=val,
        y_arima_val=yhat_ar_val,
        y_lstm_val=yhat_lstm_val,
        y_arima_test=yhat_ar_test,
        y_lstm_test=yhat_lstm_test
    )

    # 4) Metrics
    res = {
        "VAL_ARIMA": metrics(val,  yhat_ar_val),
        "VAL_LSTM":  metrics(val,  yhat_lstm_val),
        "TEST_ARIMA": metrics(test, yhat_ar_test),
        "TEST_LSTM":  metrics(test, yhat_lstm_test),
        "TEST_STACK": metrics(test, yhat_stack_test),
    }

    # Pretty print
    print(f"\n=== Results for {TICKER} (start={START}) ===")
    for k, v in res.items():
        print(f"{k:>11s}  MAE={v['mae']:.4f}  RMSE={v['rmse']:.4f}")

    # Optional: save a small summary CSV
    df_out = pd.DataFrame(res).T
    df_out.to_csv("outputs/c6_summary.csv", index=True)
    print("\nSaved -> outputs/c6_summary.csv")


if __name__ == "__main__":
    main()
