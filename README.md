# 📈 AI-Powered Sales Prediction System

An interactive dashboard built with **Python and Streamlit** that lets you upload a sales dataset, explore it,
compare machine-learning models and forecast future sales — no coding needed to use it.

> Data Science portfolio project · Python · Streamlit · Pandas · NumPy · Scikit-learn · Plotly · Matplotlib · Seaborn

---

## ✨ Features

| Area | What it does |
| --- | --- |
| **Upload** | CSV / Excel (`.xlsx`, `.xls`, multi-sheet supported), preview, row/column counts, auto-detection of date, sales, product, category and region columns |
| **Cleaning** | Removes duplicates, parses dates (incl. day-first `31/12/2024`), converts text money like `$1,200` to numbers, fills missing values, adds Year / Month / Day / Weekday |
| **Dashboard** | KPI cards (total, average, highest, lowest, transactions); monthly & daily trends, sales by product / category / region, weekday pattern, distribution and correlation heatmap (Seaborn) |
| **Models** | Linear Regression, Random Forest, Decision Tree |
| **Evaluation** | MAE, MSE, RMSE, R² with a **chronological** train/test split (no data leakage) + a naive baseline for reference |
| **Prediction** | Pick a model, choose daily (1–365 days) or monthly (1–24 months) horizon, interactive forecast chart + table |
| **Reports** | Download predictions (CSV), processed dataset (CSV), monthly summary (CSV) and a written analysis report (Markdown) |

## 🗂️ Project structure

```
sales_prediction_system/
├── app.py                  # the whole Streamlit application
├── requirements.txt        # Python dependencies
├── sample_sales_data.csv   # 3 years of demo data (with a few deliberate data-quality issues)
├── test_pipeline.py        # automated tests for the data + ML functions
├── README.md
├── .gitignore
└── .streamlit/config.toml  # theme and upload-size settings
```

## 🚀 Run it locally

```bash
# 1. (recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. start the app
streamlit run app.py
```

The app opens at <http://localhost:8501>. Python 3.10 – 3.13 is recommended.

## 🧭 How to use it

1. **📤 Upload Dataset** – upload your file, or click **Load the sample dataset**.
   Check the auto-detected *date* and *sales* columns (change them if needed), then click **Clean & prepare data**.
2. **📊 Data Analysis** – explore KPIs and charts. Use the date-range filter to zoom in.
3. **🏆 Model Performance** – choose *Daily* or *Monthly*, click **Train & evaluate models**, and compare the three models.
4. **🔮 Sales Prediction** – choose a model and a horizon, click **Generate forecast**.
5. **📥 Download Reports** – export predictions, the processed dataset and the analysis report.

**Your file needs** at least a *date* column and a *numeric sales/revenue* column. Product, category and region
columns are optional (they unlock extra charts). **Data volume:** daily forecasting needs ≥ 60 days of history,
monthly forecasting ≥ 18 months.

## 🧠 How the machine learning works

Sales are summed per day (or per month) to make a time series. Each period becomes one training row with:

* **calendar features** – year, month, day of week, day of year, weekend flag (daily) / quarter (monthly)
* **time index `t`** – lets models capture the long-term trend
* **lag features** – sales 1, 7 and 14 days ago (daily) or 1, 2 and 3 months ago (monthly)
* **rolling average** of the previous 7 days / 3 months

**Evaluation – no data leakage.** The data is *never shuffled*. The models train on the earlier part of the history
and are tested on the most recent part (default: last 20 %). Lag features only look backwards in time.
The result is compared with a naive baseline ("same as the previous period") — a useful model should beat it.

**Forecasting.** The chosen model is re-trained on *all* history and predicts one period at a time; each prediction
is fed back in as history for the next one (*recursive forecasting*). Predictions are clipped at 0.

**Metrics:** MAE (average absolute error) · MSE (average squared error) · RMSE (√MSE, same unit as sales) ·
R² (share of variance explained; 1 = perfect, ≤ 0 = no better than predicting the average).

## ✅ Testing

```bash
pip install pytest
pytest -q
```

The tests cover file loading (CSV/Excel/bad files), column detection, cleaning, feature construction (checking that
lags only use the past), the chronological split, all metrics, daily and monthly forecasts, and report/CSV export.

## ☁️ Deploy on Streamlit Community Cloud (free)

### Step 1 – Create a GitHub repository
1. Create a free account at <https://github.com> and click **New repository**.
2. Name it e.g. `sales-prediction-system`, choose **Public**, and **do not** tick "Add a README" (you already have one).

### Step 2 – Push the project
Open a terminal **inside the project folder** and run (replace `YOUR-USERNAME`):

```bash
git init
git add .
git commit -m "Initial commit: AI-powered sales prediction system"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/sales-prediction-system.git
git push -u origin main
```

If Git asks for a password, use a **Personal Access Token** (GitHub → Settings → Developer settings → Tokens),
or use GitHub Desktop / `gh auth login`.

### Step 3 – Deploy
1. Go to <https://share.streamlit.io> and sign in **with GitHub** (authorise access to your repositories).
2. Click **Create app** (or **New app**) → *Deploy a public app from GitHub*.
3. Fill in:
   * **Repository:** `YOUR-USERNAME/sales-prediction-system`
   * **Branch:** `main`
   * **Main file path:** `app.py`
   * **App URL:** choose a custom sub-domain if you like
4. Open **Advanced settings** and select Python **3.11** or **3.12**. No secrets are needed.
5. Click **Deploy**. The first build installs `requirements.txt` and takes a few minutes.

### Step 4 – Maintain it
* Every `git push` to `main` re-deploys the app automatically.
* Free apps go to sleep after a period of inactivity; visitors just click "wake up".
* Use **⋮ → Reboot app** or **Manage app → logs** if something looks wrong.
* Add the live URL and a screenshot to this README for your portfolio.

## 🛠️ Troubleshooting

| Problem | Fix |
| --- | --- |
| "No valid rows left after cleaning" | Wrong date/sales column chosen, or dates are day-first — tick **Dates are day-first** |
| "Only N months of history…" | Use *Daily* frequency or upload a longer history |
| Excel file fails to load | Make sure `openpyxl` is installed (`pip install -r requirements.txt`); for `.xls` also `xlrd` |
| Build fails on Streamlit Cloud | Check the logs; try Python 3.11/3.12 in Advanced settings |

## ⚠️ Limitations & ideas for improvement

* Forecasts rely on historical patterns only — promotions, prices, holidays or events are not modelled.
* Days/months with no transactions are treated as zero sales; a partial last month will look low.
* Possible extensions: holiday features, XGBoost/LightGBM, Prophet/ARIMA, hyper-parameter tuning,
  per-product or per-region forecasts, prediction intervals, PDF/Excel report export.

## 📄 License

Free to use for learning and portfolio purposes.# AI-POWERD-SLAES-PREDICTION-SYSTEM-
