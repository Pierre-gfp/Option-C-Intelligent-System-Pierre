"""
Minimal harness for C7.
- Runs the experiment for AAPL since 2018-01-01
- First with default threshold 0.5, then with 0.4 to check recall impact.
"""

import os
import warnings
from c7_sentiment import run_experiment

def pretty(tag, d):
    print(tag)
    print({k: v for k, v in d.items() if k in ("acc", "prec", "rec", "f1", "cm")})

def main():
    warnings.filterwarnings("ignore")
    TICKER = "AAPL"
    START = "2018-01-01"

    # Default threshold 0.5
    res = run_experiment(TICKER, START, threshold=0.5)
    pretty("BASELINE_VAL@0.5", res["baseline_val"])
    pretty("BASELINE_TEST@0.5", res["baseline_test"])
    pretty("SENT_VAL@0.5", res["sentiment_val"])
    pretty("SENT_TEST@0.5", res["sentiment_test"])
    print(f"Coverage sent_3d: {res['sentiment_coverage']:.3f}")
    print("Desc sent_3d:", res["sentiment_desc"])

    # Try a different decision threshold (e.g., favor recall)
    res2 = run_experiment(TICKER, START, threshold=0.4)
    pretty("SENT_VAL@0.4", res2["sentiment_val"])
    pretty("SENT_TEST@0.4", res2["sentiment_test"])

if __name__ == "__main__":
    main()
