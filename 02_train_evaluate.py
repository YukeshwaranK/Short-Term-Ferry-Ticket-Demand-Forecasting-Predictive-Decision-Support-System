"""
Stage 2: Baseline & ML Model Training + Multi-Horizon Evaluation
Short-Term Ferry Ticket Demand Forecasting Project
"""
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = "/home/claude/ferry_project/data/processed_features.pkl"
MODELS_DIR = "/home/claude/ferry_project/models"
RESULTS_DIR = "/home/claude/ferry_project/results"

df = pd.read_pickle(DATA_PATH)
print(f"Loaded feature table: {df.shape}")

HORIZONS = {"15min": 1, "30min": 2, "1h": 4, "2h": 8}

FEATURE_COLS = [c for c in df.columns if c.startswith(("lag_", "roll_"))] + \
    ["hour", "dow", "month", "is_weekend", "hour_sin", "hour_cos", "month_sin", "month_cos"]

# ---------------------------------------------------------------------------
# Time-based split: last 6 months = test set (rolling-forecast style holdout)
# ---------------------------------------------------------------------------
split_date = df.index.max() - pd.Timedelta(days=182)
train_start = df.index.max() - pd.Timedelta(days=182 + 730)  # last ~2 years for training
train_df = df[(df.index <= split_date) & (df.index >= train_start)]
test_df = df[df.index > split_date]
print(f"Train: {train_df.shape[0]:,} rows ({train_df.index.min()} -> {train_df.index.max()})")
print(f"Test:  {test_df.shape[0]:,} rows ({test_df.index.min()} -> {test_df.index.max()})")


def mape(y_true, y_pred):
    mask = y_true > 5  # avoid exploding % error on near-zero overnight counts
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate(y_true, y_pred):
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 3),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 3),
        "MAPE": round(mape(y_true.values, np.asarray(y_pred)), 2),
    }


all_results = []

for h_name, steps in HORIZONS.items():
    target_col = f"target_{h_name}"
    X_train, y_train = train_df[FEATURE_COLS], train_df[target_col]
    X_test, y_test = test_df[FEATURE_COLS], test_df[target_col]

    print(f"\n=== Horizon: {h_name} ===")

    # --- Baseline 1: Naive (persistence) — forecast = last known value ---
    naive_pred = test_df["Sales Count"]  # value at t, forecasting t+steps
    m = evaluate(y_test, naive_pred)
    m.update({"horizon": h_name, "model": "Naive (Persistence)"})
    all_results.append(m)

    # --- Baseline 2: Moving Average (last 4 intervals = 1h) ---
    ma_pred = test_df["roll_mean_4"]
    m = evaluate(y_test, ma_pred)
    m.update({"horizon": h_name, "model": "Moving Average (1h)"})
    all_results.append(m)

    # --- Linear Regression with lag features ---
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)
    m = evaluate(y_test, pred)
    m.update({"horizon": h_name, "model": "Linear Regression"})
    all_results.append(m)
    joblib.dump(lr, f"{MODELS_DIR}/linear_regression_{h_name}.joblib")

    # --- Random Forest Regressor ---
    rf = RandomForestRegressor(
        n_estimators=60, max_depth=10, min_samples_leaf=5,
        n_jobs=-1, random_state=42
    )
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    m = evaluate(y_test, pred)
    m.update({"horizon": h_name, "model": "Random Forest"})
    all_results.append(m)
    joblib.dump(rf, f"{MODELS_DIR}/random_forest_{h_name}.joblib")

    # --- Gradient Boosting Regressor ---
    gb = GradientBoostingRegressor(
        n_estimators=80, max_depth=3, learning_rate=0.1, random_state=42,
        subsample=0.5
    )
    gb.fit(X_train, y_train)
    pred = gb.predict(X_test)
    m = evaluate(y_test, pred)
    m.update({"horizon": h_name, "model": "Gradient Boosting"})
    all_results.append(m)
    joblib.dump(gb, f"{MODELS_DIR}/gradient_boosting_{h_name}.joblib")

    for r in all_results[-5:]:
        print(f"  {r['model']:<22} MAE={r['MAE']:<8} RMSE={r['RMSE']:<8} MAPE={r['MAPE']}%")

results_df = pd.DataFrame(all_results)
results_df.to_csv(f"{RESULTS_DIR}/model_comparison.csv", index=False)
print(f"\nSaved results -> {RESULTS_DIR}/model_comparison.csv")

# Save test predictions from best model per horizon for the dashboard + report
best_per_horizon = results_df.loc[results_df.groupby("horizon")["RMSE"].idxmin()]
best_per_horizon.to_csv(f"{RESULTS_DIR}/best_models.csv", index=False)
print("\nBest model per horizon:")
print(best_per_horizon[["horizon", "model", "MAE", "RMSE", "MAPE"]])

# Save test set predictions (using Random Forest, generally strongest) for
# the Streamlit "predicted vs actual" view
export = test_df[["Sales Count", "Redemption Count"]].copy()
for h_name in HORIZONS:
    rf = joblib.load(f"{MODELS_DIR}/random_forest_{h_name}.joblib")
    export[f"pred_{h_name}"] = rf.predict(test_df[FEATURE_COLS])
export.to_pickle(f"{RESULTS_DIR}/test_predictions.pkl")
print(f"Saved test predictions -> {RESULTS_DIR}/test_predictions.pkl")
