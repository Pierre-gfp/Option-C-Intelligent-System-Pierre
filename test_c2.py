# test_c2_min.py
from c2_data import load_and_process

out = load_and_process(
    ticker="AAPL",
    start_date="2020-01-01",
    end_date="2025-01-01",
    features=["open","high","low","close","volume"],
    cache_dir="data",
    use_cache=True,
    split_method="date",     # "ratio" | "date" | "random"
    split_date="2024-01-01",
    test_size=0.2,
    scale=True,
)
print("Features:", out["feature_columns"])
print("Label   :", out["label_col"])
print("Split   :", out["split_info"])
print("X_train/X_test:", out["X_train"].shape, out["X_test"].shape)
print("y_train/y_test:", out["y_train"].shape, out["y_test"].shape)
print("Cache CSV:", out["cache_file"])
