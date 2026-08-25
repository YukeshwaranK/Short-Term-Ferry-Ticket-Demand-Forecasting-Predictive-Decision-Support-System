"""
Shared feature-engineering logic for the Ferry Demand Forecasting project.
Used identically by the training pipeline (02_train_evaluate.py) and the
Streamlit app (app.py) so that features never drift between train and serve.
"""
import pandas as pd
import numpy as np

HORIZONS = {"15min": 1, "30min": 2, "1h": 4, "2h": 8}

# NOTE: order matters — this must exactly match the column construction order
# used at training time in 02_train_evaluate.py (built from df.columns, which
# follows the creation order in add_features() below: lag_N / lag_redemption_N
# interleaved per lag, then roll_mean/std/max interleaved per window).
FEATURE_COLS = (
    [c for l in [1, 2, 4, 8] for c in (f"lag_{l}", f"lag_redemption_{l}")]
    + [c for w in [4, 8, 16] for c in (f"roll_mean_{w}", f"roll_std_{w}", f"roll_max_{w}")]
    + ["hour", "dow", "month", "is_weekend", "hour_sin", "hour_cos", "month_sin", "month_cos"]
)


def load_raw(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").drop_duplicates(subset="Timestamp").reset_index(drop=True)
    df = df.set_index("Timestamp")[["Sales Count", "Redemption Count"]]
    return df


def build_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex to a strict 15-min grid, interpolate short gaps, flag operating status."""
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="15min")
    out = df.reindex(full_idx)
    out.index.name = "Timestamp"
    out["Sales Count"] = out["Sales Count"].interpolate(limit=16, limit_direction="both")
    out["Redemption Count"] = out["Redemption Count"].interpolate(limit=16, limit_direction="both")
    out["is_operating"] = out["Sales Count"].notna()
    return out


def add_features(calendar_df: pd.DataFrame, with_targets: bool = True) -> pd.DataFrame:
    """Take an operating-only calendar frame and add lag/rolling/temporal features."""
    d = calendar_df.copy()
    TARGET = "Sales Count"

    for lag in [1, 2, 4, 8]:
        d[f"lag_{lag}"] = d[TARGET].shift(lag)
        d[f"lag_redemption_{lag}"] = d["Redemption Count"].shift(lag)

    for window in [4, 8, 16]:
        d[f"roll_mean_{window}"] = d[TARGET].shift(1).rolling(window).mean()
        d[f"roll_std_{window}"] = d[TARGET].shift(1).rolling(window).std()
        d[f"roll_max_{window}"] = d[TARGET].shift(1).rolling(window).max()

    d["hour"] = d.index.hour
    d["dow"] = d.index.dayofweek
    d["month"] = d.index.month
    d["is_weekend"] = (d["dow"] >= 5).astype(int)
    d["hour_sin"] = np.sin(2 * np.pi * d["hour"] / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["hour"] / 24)
    d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12)

    if with_targets:
        for name, steps in HORIZONS.items():
            d[f"target_{name}"] = d[TARGET].shift(-steps)

    return d


def prepare_full_pipeline(csv_path: str):
    """Convenience wrapper: raw CSV -> (full_calendar, model_ready_df)."""
    raw = load_raw(csv_path)
    calendar = build_calendar(raw)
    model_df = calendar[calendar["is_operating"]].drop(columns=["is_operating"])
    model_df = add_features(model_df, with_targets=True).dropna()
    return calendar, model_df
