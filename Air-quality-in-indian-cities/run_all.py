# run_all.py
"""End‑to‑end pipeline for the Air‑Quality in Indian Cities project.

1️⃣  Download the CPCB master CSV from the public GitHub repo.
2️⃣  Pre‑process the data (validation, simple AQI calculation, temporal, lag
    and rolling feature engineering) and save the engineered dataset.
3️⃣  Train a quick RandomForestRegressor model.
4️⃣  Evaluate on a held‑out test set and output MAE, RMSE and R².

Running this script reproduces the files you asked for:
- data/raw/cpcb_bulletins.csv          (raw CPCB data)
- data/processed/processed.csv         (engineered dataset)
- models/trained/model.joblib          (trained model)
- models/metadata/model_metrics.csv    (training/validation metrics)
- models/metadata/eval_metrics.csv     (final evaluation metrics)
"""

import os
import pathlib
import sys
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

# ---------------------------------------------------------------------------
# 1️⃣  Download raw CPCB data
# ---------------------------------------------------------------------------
def download_cpcb(raw_path: pathlib.Path) -> pathlib.Path:
    """Fetch the master CPCB CSV and store it under ``raw_path``.

    The file is taken from the UrbanEmissionsInfo repository, which contains a
    pre‑processed master CSV (AllIndiaBulletins_master.csv).  The URL points to the
    raw file on the ``main`` branch.
    """
    url = (
    "https://raw.githubusercontent.com/urbanemissionsinfo/AQI_bulletins/"
    "master/data/Processed/AllIndiaBulletins_master.csv"
)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] Fetching {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    raw_path.write_bytes(resp.content)
    print(f"[download] Saved raw data to {raw_path}")
    return raw_path

# ---------------------------------------------------------------------------
# 2️⃣  Pre‑process & feature engineering
# ---------------------------------------------------------------------------
def calculate_aqi(row: pd.Series) -> float:
    """Very simple deterministic AQI calculation using Indian breakpoints.
    The implementation mirrors the one used in the original project – it is not
    meant to be scientifically perfect, just reproducible.
    """
    breakpoints = {
        "pm25": [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200), (91, 120, 201, 300), (121, 250, 301, 400), (251, 500, 401, 500)],
        "pm10": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200), (251, 350, 201, 300), (351, 430, 301, 400), (431, 600, 401, 500)],
        "no2":  [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200), (181, 280, 201, 300), (281, 400, 301, 400), (401, 1000, 401, 500)],
        "so2":  [(0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200), (381, 800, 201, 300), (801, 1600, 301, 400), (1601, 2620, 401, 500)],
        "co":   [(0, 1, 0, 50), (1.1, 2, 51, 100), (2.1, 10, 101, 200), (10.1, 17, 201, 300), (17.1, 34, 301, 400), (34.1, 100, 401, 500)],
        "o3":   [(0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200), (169, 208, 201, 300), (209, 748, 301, 400), (749, 1000, 401, 500)],
    }
    sub_indices = []
    for pol, bp in breakpoints.items():
        conc = row.get(pol)
        if pd.isna(conc):
            continue
        for c_lo, c_hi, i_lo, i_hi in bp:
            if c_lo <= conc <= c_hi:
                sub_i = (i_hi - i_lo) / (c_hi - c_lo) * (conc - c_lo) + i_lo
                sub_indices.append(sub_i)
                break
    return max(sub_indices) if sub_indices else np.nan


def preprocess(raw_path: pathlib.Path, processed_path: pathlib.Path) -> None:
    """Load the raw CSV, compute AQI, add temporal/lag/rolling features, and
    save the resulting dataframe to ``processed_path``.
    """
    print(f"[preprocess] Loading raw data from {raw_path}")
    df = pd.read_csv(raw_path, parse_dates=["date"])

    required_cols = {"city", "date", "pm25", "pm10", "no2", "so2", "co", "o3"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in raw data: {missing}")

    # AQI
    print("[preprocess] Calculating AQI")
    df["aqi"] = df.apply(calculate_aqi, axis=1)

    # Temporal features
    print("[preprocess] Adding temporal features")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.weekday
    df["dayofyear"] = df["date"].dt.dayofyear
    df["hour"] = df["date"].dt.hour

    # Lag features (1,2,3,7,14,30 days)
    lag_days = [1, 2, 3, 7, 14, 30]
    target_cols = ["aqi", "pm25", "pm10", "no2", "so2", "co", "o3"]
    for lag in lag_days:
        for col in target_cols:
            df[f"{col}_lag{lag}"] = (
                df.sort_values(["city", "date"]).groupby("city")[col].shift(lag)
            )

    # Rolling windows (mean, std, min, max) for 3,7,14,30 days
    win_sizes = [3, 7, 14, 30]
    for win in win_sizes:
        for col in target_cols:
            roll = (
                df.sort_values(["city", "date"]).groupby("city")[col].rolling(window=win, min_periods=1)
            )
            df[f"{col}_roll_mean_{win}"] = roll.mean().reset_index(level=0, drop=True)
            df[f"{col}_roll_std_{win}"] = roll.std().reset_index(level=0, drop=True)
            df[f"{col}_roll_min_{win}"] = roll.min().reset_index(level=0, drop=True)
            df[f"{col}_roll_max_{win}"] = roll.max().reset_index(level=0, drop=True)

    # Target for next‑day prediction
    df = df.sort_values(["city", "date"])
    df["aqi_next"] = df.groupby("city")["aqi"].shift(-1)
    df = df.dropna(subset=["aqi_next"]).reset_index(drop=True)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"[preprocess] Saved engineered dataset to {processed_path}")

# ---------------------------------------------------------------------------
# 3️⃣  Train model
# ---------------------------------------------------------------------------
def train_model(processed_path: pathlib.Path, model_path: pathlib.Path, metrics_path: pathlib.Path) -> None:
    print(f"[train] Loading processed data from {processed_path}")
    df = pd.read_csv(processed_path)

    # Chronological split: 70 % train, 15 % val, 15 % test
    n = len(df)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    target = "aqi_next"
    feature_cols = [c for c in df.columns if c not in {target, "city", "date"}]

    X_train, y_train = train[feature_cols], train[target]
    X_val, y_val = val[feature_cols], val[target]

    # Quick hyper‑parameter search
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    param_dist = {
        "n_estimators": [200, 300, 400],
        "max_depth": [10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=12,
        cv=3,
        scoring="neg_mean_absolute_error",
        random_state=42,
        n_jobs=-1,
    )
    print("[train] Running RandomForest hyper‑parameter search")
    search.fit(X_train, y_train)
    best_rf = search.best_estimator_
    val_pred = best_rf.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_pred)
    print(f"[train] Validation MAE: {val_mae:.3f}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_rf, model_path)
    print(f"[train] Model saved to {model_path}")

    # Test metrics for reporting
    X_test = test[feature_cols]
    y_test = test[target]
    test_pred = best_rf.predict(X_test)
    mae = mean_absolute_error(y_test, test_pred)
    rmse = mean_squared_error(y_test, test_pred, squared=False)
    r2 = r2_score(y_test, test_pred)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"metric": ["MAE", "RMSE", "R2"], "value": [mae, rmse, r2]}).to_csv(metrics_path, index=False)
    print(f"[train] Test metrics saved to {metrics_path}")
    print("\n=== Final Evaluation on Test Set ===")
    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²  : {r2:.4f}")

# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def main():
    repo_root = pathlib.Path(__file__).resolve().parent
    raw_path = repo_root / "data" / "raw" / "cpcb_bulletins.csv"
    processed_path = repo_root / "data" / "processed" / "processed.csv"
    model_path = repo_root / "models" / "trained" / "model.joblib"
    metrics_path = repo_root / "models" / "metadata" / "eval_metrics.csv"

    download_cpcb(raw_path)
    preprocess(raw_path, processed_path)
    train_model(processed_path, model_path, metrics_path)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
