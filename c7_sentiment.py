"""
C7 — News Sentiment Feature + Binary Direction Classifier

What this script does
---------------------
1) Download OHLCV prices from yfinance.
2) Pull recent headlines (yfinance Ticker.news), score them with FinBERT,
   and aggregate to a daily sentiment signal.
3) Engineer technical features, align + merge with shifted/rolled sentiment
   (to avoid look-ahead bias and reduce noise).
4) Train/evaluate a Logistic Regression baseline (no sentiment) vs with sentiment.
5) Print accuracy/precision/recall/F1 + confusion matrix for both settings.

Notes
-----
- Sentiment is shifted by +1 day (we only use information known at day J to
  predict day J+1). Then we apply a 3-day rolling mean to densify/smooth.
- We standardize numeric features (fit on train only).
- We use class_weight='balanced' to cope with label imbalance.
- Optionally you can evaluate a probability threshold ≠ 0.5.

Dependencies
------------
pip install yfinance transformers torch scikit-learn pandas numpy
"""

from __future__ import annotations
import os
import math
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# HuggingFace / Transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
)

# ---------------------------------------------------------------------
# 0) Utilities
# ---------------------------------------------------------------------

def _to_bday(series: pd.Series) -> pd.Series:
    """Ensure Business-Day frequency with forward-fill (keeps a regular calendar)."""
    s = series.copy()
    s.index = pd.DatetimeIndex(s.index)
    s = s[~s.index.duplicated(keep="last")]
    return s.asfreq("B").ffill()

def _confmat_to_list(cm: np.ndarray) -> List[List[int]]:
    return [[int(cm[0,0]), int(cm[0,1])],
            [int(cm[1,0]), int(cm[1,1])]]

# ---------------------------------------------------------------------
# 1) Data loading
# ---------------------------------------------------------------------

def load_prices(ticker: str = "AAPL",
                start: str = "2018-01-01",
                end: str | None = None) -> pd.DataFrame:
    """
    Download OHLCV data and return a clean DataFrame with B-day index.
    """
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} between {start} and {end}.")
    df.index = pd.DatetimeIndex(df.index)
    df = df[~df.index.duplicated(keep="last")]
    # Regularize to business days (forward-fill price fields)
    df = df.asfreq("B").ffill()
    return df

# ---------------------------------------------------------------------
# 2) News → FinBERT sentiment
# ---------------------------------------------------------------------

@dataclass
class FinBertScorer:
    model_name: str = "ProsusAI/finbert"

    def __post_init__(self):
        # Load once (CPU is fine). Cache on disk.
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.pipe = TextClassificationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            return_all_scores=True,   # we’ll compute a signed score
            device=-1                 # CPU
        )

    def score_texts(self, texts: List[str]) -> List[float]:
        """
        Convert FinBERT 3-way probs (neg, neutral, pos) into a signed score:
        score = P(pos) - P(neg)  in [-1, +1]
        """
        if len(texts) == 0:
            return []
        # Truncate very long texts to keep things snappy
        texts = [t if len(t) < 512 else t[:512] for t in texts]

        outputs = self.pipe(texts, truncation=True, padding=True)
        signed = []
        for scores in outputs:
            # scores is list of dicts: [{'label':'positive','score':...}, ...]
            probs = {d["label"].lower(): float(d["score"]) for d in scores}
            pos = probs.get("positive", 0.0)
            neg = probs.get("negative", 0.0)
            signed.append(pos - neg)
        return signed

def fetch_daily_sentiment(ticker: str = "AAPL",
                          start: str = "2018-01-01",
                          scorer: FinBertScorer | None = None) -> pd.Series:
    """
    Fetch headlines via yfinance Ticker.news, score with FinBERT,
    and aggregate to a daily mean signed score. Returns a B-day series.
    """
    tk = yf.Ticker(ticker)
    news_items = tk.news or []
    if len(news_items) == 0:
        # Some tickers/locales may return no news; return empty series
        return pd.Series(dtype=float)

    # Extract (date, title)
    rows = []
    for item in news_items:
        title = item.get("title") or ""
        ts = item.get("providerPublishTime")
        if not title or ts is None:
            continue
        dt = pd.to_datetime(int(ts), unit="s").normalize()
        rows.append((dt, title))

    if not rows:
        return pd.Series(dtype=float)

    df_news = pd.DataFrame(rows, columns=["date", "title"]).sort_values("date")
    df_news = df_news[df_news["date"] >= pd.to_datetime(start)]

    if scorer is None:
        scorer = FinBertScorer()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = scorer.score_texts(df_news["title"].tolist())

    df_news["score"] = scores
    # Daily mean signed score
    daily = (
        df_news.groupby("date")["score"]
        .mean()
        .sort_index()
    )
    # Put on business-day grid, ffill (headlines not daily)
    daily = _to_bday(daily)
    daily.name = "sent_daily"
    return daily

# ---------------------------------------------------------------------
# 3) Feature engineering + label
# ---------------------------------------------------------------------

def build_features(df: pd.DataFrame,
                   sent_daily: pd.Series | None = None) -> pd.DataFrame:
    """
    Create classic technical features and (optionally) merge sentiment.
    Label y_up = 1 if next day's return > 0 else 0.
    """
    out = pd.DataFrame(index=df.index.copy())
    out["Close"] = df["Close"].copy()

    # Technicals
    out["ret_1d"] = out["Close"].pct_change()
    out["ma_5"] = out["Close"].rolling(5).mean()
    out["ma_10"] = out["Close"].rolling(10).mean()
    out["ma_gap"] = out["ma_5"] - out["ma_10"]
    out["vol_5"] = out["ret_1d"].rolling(5).std()

    # Sentiment (shift + roll to avoid leakage and reduce noise)
    if sent_daily is not None and not sent_daily.empty:
        out = out.join(sent_daily, how="left")
        out["sent_daily_shift"] = out["sent_daily"].shift(1)               # J -> J+1
        out["sent_3d"] = out["sent_daily_shift"].rolling(3, min_periods=1).mean()
    else:
        out["sent_3d"] = np.nan  # keep column to be consistent

    # Label (direction of next day's return)
    out["ret_fwd_1d"] = out["Close"].pct_change().shift(-1)
    out["y_up"] = (out["ret_fwd_1d"] > 0).astype(int)

    return out

# ---------------------------------------------------------------------
# 4) Split + scale
# ---------------------------------------------------------------------

def time_split_index(idx: pd.DatetimeIndex,
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15) -> Tuple[pd.Index, pd.Index, pd.Index]:
    n = len(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx

@dataclass
class PreparedData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    scaler: StandardScaler
    feature_names: List[str]

def prepare_xy(df: pd.DataFrame,
               use_sentiment: bool = False,
               train_idx: pd.Index | None = None,
               val_idx: pd.Index | None = None,
               test_idx: pd.Index | None = None) -> PreparedData:
    """
    Select features, split by provided indices, fill NaNs, and scale (fit on train).
    """
    base_cols = ["ret_1d", "ma_5", "ma_10", "ma_gap", "vol_5"]
    feat_cols = base_cols + (["sent_3d"] if use_sentiment else [])

    # We drop rows with missing label; features NaN -> 0 before scaling
    df_ = df.copy()
    df_ = df_[~df_["y_up"].isna()]  # remove last row(s) without forward return
    df_[feat_cols] = df_[feat_cols].fillna(0.0)

    # Slicing by the precomputed time splits
    X_train = df_.loc[train_idx.intersection(df_.index), feat_cols]
    y_train = df_.loc[train_idx.intersection(df_.index), "y_up"].astype(int)

    X_val = df_.loc[val_idx.intersection(df_.index), feat_cols]
    y_val = df_.loc[val_idx.intersection(df_.index), "y_up"].astype(int)

    X_test = df_.loc[test_idx.intersection(df_.index), feat_cols]
    y_test = df_.loc[test_idx.intersection(df_.index), "y_up"].astype(int)

    # Standardize (fit on train only)
    scaler = StandardScaler()
    X_train[feat_cols] = scaler.fit_transform(X_train[feat_cols])
    X_val[feat_cols]   = scaler.transform(X_val[feat_cols])
    X_test[feat_cols]  = scaler.transform(X_test[feat_cols])

    return PreparedData(
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
        scaler=scaler,   feature_names=feat_cols
    )

# ---------------------------------------------------------------------
# 5) Train / Evaluate
# ---------------------------------------------------------------------

@dataclass
class EvalResult:
    acc: float
    prec: float
    rec: float
    f1: float
    cm: List[List[int]]
    features: List[str]

def _eval_from_probs(y_true: pd.Series,
                     probs: np.ndarray,
                     threshold: float = 0.5) -> EvalResult:
    y_hat = (probs >= threshold).astype(int)
    acc = accuracy_score(y_true, y_hat)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_hat, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_hat)
    return EvalResult(
        acc=float(acc),
        prec=float(prec),
        rec=float(rec),
        f1=float(f1),
        cm=_confmat_to_list(cm),
        features=[]
    )

def train_and_eval(prep: PreparedData,
                   threshold: float = 0.5) -> Tuple[EvalResult, EvalResult]:
    """
    Train on train set; evaluate on val and test, returning (val_res, test_res).
    """
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        n_jobs=None
    )
    clf.fit(prep.X_train.values, prep.y_train.values)

    p_val  = clf.predict_proba(prep.X_val.values)[:, 1]
    p_test = clf.predict_proba(prep.X_test.values)[:, 1]

    val_res  = _eval_from_probs(prep.y_val,  p_val,  threshold)
    test_res = _eval_from_probs(prep.y_test, p_test, threshold)

    # Add feature names to the result for traceability
    val_res.features  = [(c, "") for c in prep.feature_names]
    test_res.features = [(c, "") for c in prep.feature_names]
    return val_res, test_res

# ---------------------------------------------------------------------
# 6) Main runner
# ---------------------------------------------------------------------

def run_experiment(ticker: str = "AAPL",
                   start: str = "2018-01-01",
                   threshold: float = 0.5) -> Dict[str, dict]:
    df = load_prices(ticker, start=start)
    y_close = df["Close"]

    # Sentiment (can be empty if API returns no news)
    try:
        scorer = FinBertScorer()
        sent_daily = fetch_daily_sentiment(ticker, start=start, scorer=scorer)
    except Exception as e:
        warnings.warn(f"Sentiment fetch failed ({e}); continuing without sentiment.")
        sent_daily = pd.Series(dtype=float)

    # Build feature table + label
    feats = build_features(df, sent_daily=sent_daily)

    # Time splits
    train_idx, val_idx, test_idx = time_split_index(feats.index, 0.7, 0.15)

    # Baseline (no sentiment)
    prep_base = prepare_xy(feats, use_sentiment=False,
                           train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    val_base, test_base = train_and_eval(prep_base, threshold=threshold)

    # With sentiment
    prep_sent = prepare_xy(feats, use_sentiment=True,
                           train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    val_sent, test_sent = train_and_eval(prep_sent, threshold=threshold)

    # Quick diagnostics on sentiment coverage
    coverage = float(feats["sent_3d"].notna().mean())
    desc = feats["sent_3d"].describe().to_dict()

    results = {
        "baseline_val":  vars(val_base),
        "baseline_test": vars(test_base),
        "sentiment_val": vars(val_sent),
        "sentiment_test": vars(test_sent),
        "sentiment_coverage": coverage,
        "sentiment_desc": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                           for k, v in desc.items()}
    }
    return results

def main():
    TICKER = os.environ.get("TICKER", "AAPL")
    START = os.environ.get("START", "2018-01-01")
    THRESH = float(os.environ.get("THRESH", "0.5"))

    results = run_experiment(TICKER, START, threshold=THRESH)

    print("=== Baseline (no sentiment) ===")
    print(results["baseline_val"])
    print(results["baseline_test"])

    print("=== + Sentiment ===")
    print(results["sentiment_val"])
    print(results["sentiment_test"])

    print(f"Sentiment coverage (non-NaN share): {results['sentiment_coverage']:.3f}")
    print("Sentiment 3d describe:", results["sentiment_desc"])

    # Optional: save JSON/CSV summary if you want
    outdir = "outputs"
    os.makedirs(outdir, exist_ok=True)
    # Flatten two main rows into a quick CSV-like table
    rows = []
    for tag in ["baseline_val", "baseline_test", "sentiment_val", "sentiment_test"]:
        r = results[tag].copy()
        rows.append({
            "split": tag,
            "acc": r["acc"], "prec": r["prec"], "rec": r["rec"], "f1": r["f1"],
            "cm": r["cm"],
            "features": ";".join([f for f,_ in r["features"]]) if "features" in r else ""
        })
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "c7_results.csv"), index=False)
    print(f"Saved -> {os.path.join(outdir, 'c7_results.csv')}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
