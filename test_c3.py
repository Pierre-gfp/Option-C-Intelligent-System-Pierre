# test_c3.py
import yfinance as yf
import c3_viz as viz

TICKER = "AAPL"
START  = "2022-01-01"
END    = "2024-12-31"

def main():
    df = yf.download(TICKER, start=START, end=END, auto_adjust=False)  # works with MultiIndex too

    viz.plot_candlestick(
        df, n=1, mav=(10, 20),
        title=f"{TICKER} — Daily",
        savepath="candles_daily.png"
    )

    viz.plot_candlestick(
        df, n=5, mav=(10, 20),
        title=f"{TICKER} — 5-Day Bars",
        savepath="candles_5d.png"
    )

    viz.plot_boxplot_moving_window(
        df, window=20, step=5, column="Close",
        title=f"{TICKER} — Boxplots (20-day window, step 5)",
        savepath="boxplot_20_5.png"
    )

if __name__ == "__main__":
    main()
