# AlphaPredictor

<img width="1321" height="603" alt="Screenshot (473)" src="https://github.com/user-attachments/assets/d5277b26-ec9d-4f92-a7a8-950604964ce9" />

AlphaPredictor is a stock analytics web app built with **Streamlit**. It downloads 5 years of historical price data, engineers return-based features, trains three machine learning models, and predicts the next-day closing price — all with an interactive dashboard.

---

## Features

- Fetches 5 years of historical market data via `yfinance`
- Trains three independent models:
  - **Linear Regression** — fast baseline
  - **Random Forest Regressor** — ensemble tree model
  - **XGBoost Regressor** — gradient-boosted trees
- Predicts next-day closing price as a **3-model ensemble average**
- Confidence score derived from how closely the three models agree
- Bullish / Bearish trend signal (`▲` / `▼`)
- **Model Comparison table** — MAE, RMSE, and MAPE on a held-out test set
- **Actual vs Predicted chart** — visual comparison of all three models on the test period
- **Historical price chart** with 20-day and 50-day moving averages (Plotly)
- Modular codebase, clean error handling, and type-annotated throughout




## Project Structure

```text
AlphaPredictor/
├── app.py                 # Streamlit entry point
├── modules/
│   ├── data_loader.py     # Fetch & clean historical stock data (yfinance)
│   ├── model.py           # Feature engineering + LR / RF / XGBoost training
│   └── visualization.py   # Historical price chart (Plotly)
├── data/                  # Optional folder for cached CSVs
├── requirements.txt
└── README.md
```

---

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Run the App

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

---

## How It Works

1. **Enter a ticker** such as `AAPL`, `TSLA`, or `MSFT` and click **Predict →**.
2. The app downloads 5 years of daily OHLCV data.
3. **Feature engineering** converts raw prices to stationary daily returns, then builds:
   - Three lag features (`lag_1`, `lag_2`, `lag_3`)
   - Two rolling-average features (`ma_5`, `ma_10`)
4. **Train / test split** — oldest 80% for training, newest 20% for held-out evaluation (no shuffle — time-series integrity is preserved).
5. Each model is **evaluated on the test window** → MAE, RMSE, MAPE reported.
6. All three models are **retrained on 100% of data** before predicting tomorrow, so the most recent market behaviour is captured.
7. Tomorrow's predicted return is converted back to a price: `price = latest_close × (1 + predicted_return)`.
8. The **ensemble prediction** is the simple average of all three model prices.
9. **Confidence** is estimated from the standard deviation of the three predicted returns — lower spread = higher confidence.

---

## Output Explained

| Field | Description |
|---|---|
| Latest Close | Most recent closing price |
| Predicted Next Close | Ensemble (average of LR + RF + XGB) price forecast |
| Expected Change | % difference between ensemble prediction and latest close |
| Confidence | 0–100 score; higher = more agreement between models |
| Signal | ▲ Bullish if prediction > latest close, ▼ Bearish otherwise |

---

## Model Comparison Table

After prediction, the app shows a per-model breakdown on the held-out test set:

| Model | MAE | RMSE | MAPE (%) |
|---|---|---|---|
| Linear Regression | — | — | — |
| Random Forest | — | — | — |
| XGBoost | — | — | — |

---

## Design Decisions

- **Returns instead of raw prices** — keeps the target stationary; Random Forest and XGBoost can extrapolate correctly even at all-time highs.
- **Two-phase training** — eval phase on the test split gives honest error metrics; final phase retrains on all data before forecasting.
- **No data leakage** — the latest row (for tomorrow's forecast) is captured *before* `dropna()` removes it from the training set.
- **MultiIndex compatibility** — newer `yfinance` versions return a MultiIndex column structure; columns are flattened automatically after download.

---

## Ideas for Extension

- Add RSI, MACD, Bollinger Bands as features
- Experiment with LSTM or Transformer-based models
- Add ticker autocomplete / search
- Cache downloaded data with `st.cache_data` to speed up repeated runs
- Deploy to Streamlit Community Cloud

---

## Disclaimer

This project is for educational purposes only and is **not financial advice**. Stock markets are inherently unpredictable; never make investment decisions based solely on model output.
