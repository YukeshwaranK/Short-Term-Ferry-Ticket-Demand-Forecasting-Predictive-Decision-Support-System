"""
Stage 1: Data Preparation & Feature Engineering
Short-Term Ferry Ticket Demand Forecasting Project
"""
import pandas as pd
import numpy as np

RAW_PATH = "/mnt/user-data/uploads/Toronto_Island_Ferry_Tickets.csv"
OUT_PATH = "/home/claude/ferry_project/data/processed_features.pkl"

print("Loading raw data...")
df = pd.read_csv(RAW_PATH)
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").drop_duplicates(subset="Timestamp").reset_index(drop=True)
df = df.set_index("Timestamp")
df = df[["Sales Count", "Redemption Count"]]

print(f"Raw rows: {len(df):,}  Range: {df.index.min()} -> {df.index.max()}")

# ---------------------------------------------------------------------------
# Reindex to a strict 15-minute grid so gaps are explicit, then fill.
# Ferry operations are seasonal/hours-limited, so long gaps (overnight,
# off-season closures) are genuine "not operating" periods, not missing data.
# We mask any gap longer than 4 hours as non-operating (excluded from
# training) and interpolate short gaps (<= 4 hours = 16 intervals).
# ---------------------------------------------------------------------------
full_idx = pd.date_range(df.index.min(), df.index.max(), freq="15min")
df = df.reindex(full_idx)
df.index.name = "Timestamp"

gap_mask = df["Sales Count"].isna()
# short gaps: interpolate linearly
df["Sales Count"] = df["Sales Count"].interpolate(limit=16, limit_direction="both")
df["Redemption Count"] = df["Redemption Count"].interpolate(limit=16, limit_direction="both")
df["is_operating"] = df["Sales Count"].notna()

print(f"After reindex to full 15-min grid: {len(df):,} rows")
print(f"Operating intervals: {df['is_operating'].sum():,} "
      f"({df['is_operating'].mean()*100:.1f}%)")

# Drop rows that are still NaN (long non-operating stretches) for modeling,
# but keep a copy of the full calendar for the app.
df.to_pickle("/home/claude/ferry_project/data/full_calendar.pkl")

model_df = df[df["is_operating"]].copy()
model_df = model_df.drop(columns=["is_operating"])

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
TARGET = "Sales Count"

for lag in [1, 2, 4, 8]:
    model_df[f"lag_{lag}"] = model_df[TARGET].shift(lag)
    model_df[f"lag_redemption_{lag}"] = model_df["Redemption Count"].shift(lag)

for window in [4, 8, 16]:  # 1h, 2h, 4h rolling windows
    model_df[f"roll_mean_{window}"] = model_df[TARGET].shift(1).rolling(window).mean()
    model_df[f"roll_std_{window}"] = model_df[TARGET].shift(1).rolling(window).std()
    model_df[f"roll_max_{window}"] = model_df[TARGET].shift(1).rolling(window).max()

model_df["hour"] = model_df.index.hour
model_df["dow"] = model_df.index.dayofweek
model_df["month"] = model_df.index.month
model_df["is_weekend"] = (model_df["dow"] >= 5).astype(int)
model_df["hour_sin"] = np.sin(2 * np.pi * model_df["hour"] / 24)
model_df["hour_cos"] = np.cos(2 * np.pi * model_df["hour"] / 24)
model_df["month_sin"] = np.sin(2 * np.pi * model_df["month"] / 12)
model_df["month_cos"] = np.cos(2 * np.pi * model_df["month"] / 12)

# Multi-horizon targets (steps of 15 min): 15m=1, 30m=2, 1h=4, 2h=8
HORIZONS = {"15min": 1, "30min": 2, "1h": 4, "2h": 8}
for name, steps in HORIZONS.items():
    model_df[f"target_{name}"] = model_df[TARGET].shift(-steps)

model_df = model_df.dropna()
print(f"Final feature table: {model_df.shape}")

model_df.to_pickle(OUT_PATH)
print(f"Saved -> {OUT_PATH}")
