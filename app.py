"""
Toronto Island Ferry — Short-Term Demand Forecasting Dashboard
Streamlit application: Stage 3 deliverable.

Run locally with:
    streamlit run app.py

Expects, in the same folder:
    Toronto_Island_Ferry_Tickets.csv   (raw dataset)
    models/random_forest_<horizon>.joblib
    models/gradient_boosting_<horizon>.joblib
    models/linear_regression_<horizon>.joblib
    features.py                        (shared feature module)
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path

from features import prepare_full_pipeline, FEATURE_COLS, HORIZONS

st.set_page_config(page_title="Ferry Demand Forecast", layout="wide", page_icon="⛴️")

BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "Toronto_Island_Ferry_Tickets.csv"
MODELS_DIR = BASE_DIR / "models"

MODEL_FILES = {
    "Random Forest": "random_forest_{h}.joblib",
    "Gradient Boosting": "gradient_boosting_{h}.joblib",
    "Linear Regression": "linear_regression_{h}.joblib",
}


@st.cache_data(show_spinner="Loading and preparing ticket data...")
def get_data():
    calendar, model_df = prepare_full_pipeline(str(CSV_PATH))
    return calendar, model_df


@st.cache_resource(show_spinner=False)
def get_model(model_name, horizon):
    fname = MODEL_FILES[model_name].format(h=horizon)
    path = MODELS_DIR / fname
    if not path.exists():
        return None
    return joblib.load(path)


def naive_and_ma_forecast(model_df, horizon_steps):
    naive = model_df["Sales Count"]
    ma = model_df["roll_mean_4"]
    return naive, ma


def rf_prediction_interval(rf_model, X_row, lower_q=0.1, upper_q=0.9):
    """Approximate a prediction interval from the spread of individual tree
    predictions inside the Random Forest ensemble."""
    X_arr = X_row.values  # bypass per-tree feature-name validation warnings
    tree_preds = np.array([t.predict(X_arr)[0] for t in rf_model.estimators_])
    return (
        float(np.quantile(tree_preds, lower_q)),
        float(np.mean(tree_preds)),
        float(np.quantile(tree_preds, upper_q)),
    )


st.title("⛴️ Toronto Island Ferry — Short-Term Demand Forecast")
st.caption(
    "Predictive decision support for ferry ticket sales & redemptions — "
    "15-minute to 2-hour ahead forecasts."
)

if not CSV_PATH.exists():
    st.error(
        f"Dataset not found at `{CSV_PATH.name}`. Place "
        "`Toronto_Island_Ferry_Tickets.csv` in the same folder as this app."
    )
    st.stop()

calendar, model_df = get_data()

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Forecast Controls")

horizon_label = st.sidebar.selectbox("Forecast horizon", list(HORIZONS.keys()), index=2)
horizon_steps = HORIZONS[horizon_label]

model_choice = st.sidebar.selectbox("Model", list(MODEL_FILES.keys()), index=0)

min_date = model_df.index.min().date()
max_date = model_df.index.max().date()
sel_date = st.sidebar.date_input(
    "Reference date", value=max_date, min_value=min_date, max_value=max_date
)
available_times = model_df.loc[str(sel_date)].index
if len(available_times) == 0:
    st.sidebar.warning("No operating data for this date.")
    st.stop()

sel_time = st.sidebar.select_slider(
    "Reference time (forecast made 'as of' this moment)",
    options=list(available_times.strftime("%H:%M")),
    value=available_times.strftime("%H:%M")[len(available_times) // 2],
)
show_interval = st.sidebar.checkbox("Show confidence band (Random Forest only)", value=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📈 Live Forecast", "🔬 Model Comparison", "📊 Historical Patterns"])

# === TAB 1: Live forecast ===================================================
with tab1:
    ref_ts = pd.Timestamp(f"{sel_date} {sel_time}")
    if ref_ts not in model_df.index:
        st.warning("Selected timestamp not in the operating dataset.")
    else:
        row = model_df.loc[[ref_ts]]
        model = get_model(model_choice, horizon_label)

        col1, col2, col3, col4 = st.columns(4)
        current_val = row["Sales Count"].values[0]
        col1.metric("Current interval sales", f"{current_val:.0f} tickets")

        if model is not None:
            pred = model.predict(row[FEATURE_COLS])[0]
            actual_future_ts = ref_ts + pd.Timedelta(minutes=15 * horizon_steps)
            actual_future = (
                model_df.loc[actual_future_ts, "Sales Count"]
                if actual_future_ts in model_df.index else None
            )
            col2.metric(f"Forecast (+{horizon_label})", f"{pred:.0f} tickets")
            if actual_future is not None:
                delta = pred - actual_future
                col3.metric("Actual (if already happened)", f"{actual_future:.0f} tickets",
                            delta=f"{delta:+.0f} forecast error")
            else:
                col3.metric("Actual (if already happened)", "— (in the future)")
            surge_flag = "🔴 High demand" if pred > model_df["Sales Count"].quantile(0.9) else "🟢 Normal"
            col4.metric("Demand signal", surge_flag)
        else:
            st.info(f"No saved model file for {model_choice} / {horizon_label}. "
                    "Run 02_train_evaluate.py to generate it.")
            pred = None

        # Chart: recent history + forecast point (+ interval if RF)
        window_start = ref_ts - pd.Timedelta(hours=6)
        window_end = ref_ts + pd.Timedelta(hours=3)
        hist = model_df.loc[window_start:window_end, ["Sales Count"]]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist["Sales Count"],
                                  name="Actual sales", line=dict(color="#1f6feb")))
        fig.add_vline(x=ref_ts, line_dash="dot", line_color="gray",
                      annotation_text="Now")

        if model is not None and pred is not None:
            future_ts = ref_ts + pd.Timedelta(minutes=15 * horizon_steps)
            if show_interval and model_choice == "Random Forest":
                lo, mid, hi = rf_prediction_interval(model, row[FEATURE_COLS])
                fig.add_trace(go.Scatter(
                    x=[future_ts, future_ts], y=[lo, hi], mode="lines",
                    line=dict(color="rgba(31,111,235,0.3)", width=10),
                    name="80% confidence band"
                ))
            fig.add_trace(go.Scatter(
                x=[future_ts], y=[pred], mode="markers", marker=dict(size=12, color="#f85149"),
                name=f"Forecast (+{horizon_label})"
            ))

        fig.update_layout(height=420, margin=dict(t=30, b=10),
                           yaxis_title="Tickets / 15-min interval")
        st.plotly_chart(fig, use_container_width=True)

# === TAB 2: Model comparison =================================================
with tab2:
    st.subheader("Model Performance Across Forecast Horizons")
    results_path = BASE_DIR / "results" / "model_comparison.csv"
    if results_path.exists():
        results = pd.read_csv(results_path)
        metric = st.radio("Metric", ["MAE", "RMSE", "MAPE"], horizontal=True)
        pivot = results.pivot(index="model", columns="horizon", values=metric)
        pivot = pivot[["15min", "30min", "1h", "2h"]]
        st.dataframe(pivot.style.background_gradient(cmap="RdYlGn_r", axis=None), use_container_width=True)

        fig2 = go.Figure()
        for m in pivot.index:
            fig2.add_trace(go.Scatter(x=pivot.columns, y=pivot.loc[m], mode="lines+markers", name=m))
        fig2.update_layout(height=380, yaxis_title=metric, xaxis_title="Forecast horizon",
                            title=f"{metric} by model and horizon (lower is better)")
        st.plotly_chart(fig2, use_container_width=True)

        st.caption(
            "Baseline models (Naive, Moving Average) show rising error at longer "
            "horizons ('error drift'). Random Forest maintains the flattest error "
            "curve, making it the recommended production model."
        )
    else:
        st.info("Run 02_train_evaluate.py first to generate results/model_comparison.csv")

    st.subheader("Predicted vs Actual — Sample Period")
    pred_path = BASE_DIR / "results" / "test_predictions.pkl"
    if pred_path.exists():
        test_pred = pd.read_pickle(pred_path)
        default_start = test_pred.index[len(test_pred) // 3]
        pick_date = st.date_input(
            "Week to inspect", value=default_start.date(),
            min_value=test_pred.index.min().date(), max_value=test_pred.index.max().date(),
            key="compare_week"
        )
        week = test_pred.loc[str(pick_date): str(pd.Timestamp(pick_date) + pd.Timedelta(days=7))]
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=week.index, y=week["Sales Count"], name="Actual", line=dict(color="black")))
        fig3.add_trace(go.Scatter(x=week.index, y=week[f"pred_{horizon_label}"],
                                   name=f"{horizon_label} forecast (Random Forest)", line=dict(color="#1f6feb")))
        fig3.update_layout(height=380, yaxis_title="Tickets / 15-min interval")
        st.plotly_chart(fig3, use_container_width=True)

# === TAB 3: Historical patterns ==============================================
with tab3:
    st.subheader("Historical Demand Patterns")
    c1, c2 = st.columns(2)
    with c1:
        hourly = calendar.groupby(calendar.index.hour)["Sales Count"].mean()
        fig4 = go.Figure(go.Bar(x=hourly.index, y=hourly.values, marker_color="#2ea043"))
        fig4.update_layout(title="Average sales by hour of day", height=320,
                            xaxis_title="Hour", yaxis_title="Avg tickets/interval")
        st.plotly_chart(fig4, use_container_width=True)
    with c2:
        dow = calendar.groupby(calendar.index.dayofweek)["Sales Count"].mean()
        labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        fig5 = go.Figure(go.Bar(x=labels, y=dow.values, marker_color="#d29922"))
        fig5.update_layout(title="Average sales by day of week", height=320,
                            yaxis_title="Avg tickets/interval")
        st.plotly_chart(fig5, use_container_width=True)

    monthly = calendar["Sales Count"].resample("MS").sum()
    fig6 = go.Figure(go.Scatter(x=monthly.index, y=monthly.values, line=dict(color="#1f6feb")))
    fig6.update_layout(title="Monthly ticket sales volume (2015–2025)", height=320,
                        yaxis_title="Total tickets sold")
    st.plotly_chart(fig6, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Data: Toronto Island Park ferry ticket system · "
    "Models trained on ~2 years of 15-min interval data · "
    "Test period: most recent 6 months (rolling holdout)."
)
