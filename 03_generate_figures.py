import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 140, "font.size": 9})

FIG_DIR = "/home/claude/ferry_project/figures"
df_full = pd.read_pickle("/home/claude/ferry_project/data/full_calendar.pkl")
results = pd.read_csv("/home/claude/ferry_project/results/model_comparison.csv")
test_pred = pd.read_pickle("/home/claude/ferry_project/results/test_predictions.pkl")

# 1. Historical sales trend (monthly aggregate)
monthly = df_full["Sales Count"].resample("MS").sum()
plt.figure(figsize=(8, 3.2))
plt.plot(monthly.index, monthly.values, color="#1f6feb", linewidth=1.3)
plt.title("Monthly Ticket Sales Volume (2015–2025)")
plt.ylabel("Tickets sold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_monthly_trend.png")
plt.close()

# 2. Average daily demand curve by hour
hourly = df_full.groupby(df_full.index.hour)["Sales Count"].mean()
plt.figure(figsize=(7, 3.2))
plt.bar(hourly.index, hourly.values, color="#2ea043")
plt.title("Average Ticket Sales by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Avg tickets / 15-min interval")
plt.xticks(range(0, 24, 2))
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_hourly_pattern.png")
plt.close()

# 3. Day-of-week pattern
dow = df_full.groupby(df_full.index.dayofweek)["Sales Count"].mean()
labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
plt.figure(figsize=(6, 3.2))
plt.bar(labels, dow.values, color="#d29922")
plt.title("Average Ticket Sales by Day of Week")
plt.ylabel("Avg tickets / 15-min interval")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_dow_pattern.png")
plt.close()

# 4. Model comparison — MAE by horizon
plt.figure(figsize=(7.5, 3.8))
horizons_order = ["15min", "30min", "1h", "2h"]
models_order = ["Naive (Persistence)", "Moving Average (1h)", "Linear Regression",
                "Random Forest", "Gradient Boosting"]
colors = ["#8b949e", "#c9a227", "#1f6feb", "#2ea043", "#f85149"]
width = 0.15
x = np.arange(len(horizons_order))
for i, m in enumerate(models_order):
    vals = [results[(results.horizon == h) & (results.model == m)]["MAE"].values[0]
            for h in horizons_order]
    plt.bar(x + i * width, vals, width=width, label=m, color=colors[i])
plt.xticks(x + width * 2, horizons_order)
plt.ylabel("MAE (tickets)")
plt.title("Forecast Error (MAE) by Model and Horizon")
plt.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_mae_by_horizon.png")
plt.close()

# 5. Predicted vs Actual (1h horizon, a representative week)
week = test_pred.loc["2025-08-04":"2025-08-10"]
plt.figure(figsize=(9, 3.2))
plt.plot(week.index, week["Sales Count"].shift(-4), label="Actual", color="black", linewidth=1)
plt.plot(week.index, week["pred_1h"], label="Random Forest — 1h forecast", color="#1f6feb", linewidth=1)
plt.title("Predicted vs Actual Ticket Sales — 1-Hour Horizon (Sample Week, Aug 2025)")
plt.ylabel("Tickets / 15-min interval")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_pred_vs_actual_1h.png")
plt.close()

# 6. Error drift across horizons (RMSE)
plt.figure(figsize=(6.5, 3.2))
rf_results = results[results.model == "Random Forest"].set_index("horizon").loc[horizons_order]
plt.plot(horizons_order, rf_results["MAE"], marker="o", label="MAE", color="#1f6feb")
plt.plot(horizons_order, rf_results["RMSE"], marker="o", label="RMSE", color="#f85149")
plt.title("Random Forest Error Drift Across Forecast Horizons")
plt.ylabel("Error (tickets)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/06_error_drift.png")
plt.close()

print("Figures saved to", FIG_DIR)
import os
print(os.listdir(FIG_DIR))
