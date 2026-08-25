# Toronto Island Ferry — Short-Term Demand Forecasting

Predictive decision support system for Toronto Island Park ferry operations.
Forecasts ticket sales/redemptions 15 minutes to 2 hours ahead using
historical 15-minute interval data (2015–2025).

## Folder contents

```
ferry_project/
├── Toronto_Island_Ferry_Tickets.csv   # raw dataset
├── features.py                        # shared feature-engineering module
├── 01_prepare_data.py                 # Stage 1: cleaning + feature engineering
├── 02_train_evaluate.py               # Stage 2: model training + evaluation
├── 03_generate_figures.py             # Stage 2b: report figures
├── app.py                             # Stage 3: Streamlit dashboard
├── requirements.txt
├── models/                            # trained model files (.joblib)
├── results/                           # model_comparison.csv, test predictions
├── figures/                           # PNG charts used in the research paper
└── data/                              # processed feature tables (cache)
```

## Setup

```bash
pip install -r requirements.txt
```

## Reproducing the pipeline from scratch

```bash
python 01_prepare_data.py       # builds data/processed_features.pkl
python 02_train_evaluate.py     # trains models, writes results/ and models/
python 03_generate_figures.py   # writes figures/*.png
```

## Running the dashboard

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## Methodology summary

- **Data**: ~261k raw ticket-count rows reindexed to a strict 15-minute grid
  (~373k intervals, 2015-05 to 2025-12); short gaps (≤4h) interpolated,
  99.6% of the calendar has genuine operating data.
- **Features**: lags (t-1,2,4,8) on sales & redemptions, rolling mean/std/max
  over 1h/2h/4h windows, cyclical hour/month encodings, day-of-week, weekend flag.
- **Split**: strict time-based — most recent 6 months held out as test set;
  models trained on the 2 years prior (rolling-forecast style, no shuffling).
- **Models compared**: Naive persistence, Moving Average, Linear Regression,
  Random Forest, Gradient Boosting — at 15-min/30-min/1h/2h horizons.
- **Result**: Random Forest gives the lowest error at every horizon and the
  flattest "error drift" as horizon lengthens — see `results/model_comparison.csv`.

### Not run in this build (no internet access in the build sandbox)
ARIMA/SARIMA, Facebook Prophet, and XGBoost were specified in the original
brief as optional/extension models. `requirements.txt` lists the packages
needed; the modeling script is structured so any of them can be added as one
more model branch inside the horizon loop in `02_train_evaluate.py`, using the
same `FEATURE_COLS` / train-test split already in place.
