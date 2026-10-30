"""
AI-Powered Sales Prediction System
==================================
A Streamlit dashboard that lets you upload a sales dataset, explore it,
compare machine-learning models and forecast future sales.

Run locally with:   streamlit run app.py

Code layout (top to bottom)
    1. Configuration & constants
    2. Data loading, column detection and preprocessing   (pure functions)
    3. Machine learning: features, training, evaluation, forecasting (pure functions)
    4. Charts
    5. Report generation
    6. UI helpers and the six pages
    7. main()
"""
from __future__ import annotations

import html
import io
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend, required on servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

# =============================================================================
# 1. CONFIGURATION & CONSTANTS
# =============================================================================
APP_TITLE = "AI-Powered Sales Prediction System"
SAMPLE_FILE = Path(__file__).parent / "sample_sales_data.csv"

PALETTE = ["#4F46E5", "#06B6D4", "#F59E0B", "#10B981", "#EF4444", "#8B5CF6", "#EC4899", "#64748B"]
MODEL_COLORS = {"Linear Regression": "#4F46E5", "Random Forest": "#10B981", "Decision Tree": "#F59E0B"}
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

PAGE_HOME = "🏠 Home Dashboard"
PAGE_UPLOAD = "📤 Upload Dataset"
PAGE_ANALYSIS = "📊 Data Analysis"
PAGE_PREDICT = "🔮 Sales Prediction"
PAGE_PERFORMANCE = "🏆 Model Performance"
PAGE_REPORTS = "📥 Download Reports"
PAGES = [PAGE_HOME, PAGE_UPLOAD, PAGE_ANALYSIS, PAGE_PREDICT, PAGE_PERFORMANCE, PAGE_REPORTS]

# Settings that depend on whether we forecast daily or monthly totals.
#   lags  -> "sales N periods ago" are used as features
#   roll  -> size of the rolling average (of previous periods) used as a feature
FREQ_CONFIG = {
    "Daily": {
        "rule": "D", "unit": "days", "lags": [1, 7, 14], "roll": 7,
        "min_points": 60, "max_horizon": 365, "default_horizon": 30,
    },
    "Monthly": {
        "rule": "MS", "unit": "months", "lags": [1, 2, 3], "roll": 3,
        "min_points": 18, "max_horizon": 24, "default_horizon": 6,
    },
}

# Keywords used to guess which column is which (checked in this order).
CATEGORY_HINTS = ["categ", "segment", "type", "department", "class"]
REGION_HINTS = ["region", "area", "zone", "state", "country", "city", "location", "store", "territory"]
PRODUCT_HINTS = ["product", "item", "sku"]
SALES_HINTS = ["sales", "revenue", "amount", "turnover", "income", "total", "price"]
DATE_HINTS = ["date", "time", "timestamp", "day", "period"]

CSS = """
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
.hero {background: linear-gradient(120deg,#4F46E5 0%,#7C3AED 60%,#06B6D4 100%); border-radius: 16px;
       padding: 24px 30px; color: #fff; margin-bottom: 22px; box-shadow: 0 6px 20px rgba(79,70,229,.25);}
.hero-title {font-size: 1.85rem; font-weight: 700; margin: 0 0 4px 0; line-height: 1.25;}
.hero-sub {margin: 0; opacity: .92; font-size: 1rem;}
.kpi-card {border-radius: 14px; padding: 16px 18px; color: #fff; box-shadow: 0 4px 14px rgba(0,0,0,.12); margin-bottom: 12px;}
.kpi-label {font-size: .75rem; opacity: .92; text-transform: uppercase; letter-spacing: .06em;}
.kpi-value {font-size: 1.5rem; font-weight: 700; margin-top: 4px; word-break: break-word;}
.kpi-1 {background: linear-gradient(135deg,#4F46E5,#6366F1);}
.kpi-2 {background: linear-gradient(135deg,#0891B2,#06B6D4);}
.kpi-3 {background: linear-gradient(135deg,#059669,#10B981);}
.kpi-4 {background: linear-gradient(135deg,#D97706,#F59E0B);}
.kpi-5 {background: linear-gradient(135deg,#7C3AED,#8B5CF6);}
.step-card {border: 1px solid rgba(128,128,128,.28); border-radius: 12px; padding: 14px 18px; min-height: 120px;}
.step-card h4 {margin: 0 0 6px 0;}
.step-card p {margin: 0; font-size: .92rem; opacity: .85;}
@media (max-width: 768px) {
  .hero-title {font-size: 1.35rem;} .kpi-value {font-size: 1.25rem;} .hero {padding: 18px;}
}
</style>
"""


# =============================================================================
# 2. DATA LOADING, COLUMN DETECTION & PREPROCESSING  (no Streamlit code here)
# =============================================================================
def load_dataset(file_bytes: bytes, filename: str, sheet_name=None) -> pd.DataFrame:
    """Read a CSV / Excel file (given as bytes) into a DataFrame."""
    ext = Path(filename).suffix.lower()
    try:
        if ext == ".csv":
            # Try the most common encodings in order; latin-1 never fails.
            for encoding in ("utf-8-sig", "latin-1"):
                try:
                    df = pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name if sheet_name else 0)
        else:
            raise ValueError("Unsupported file type. Please upload a .csv, .xlsx or .xls file.")
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The file is empty.") from exc
    except ValueError:
        raise
    except Exception as exc:  # corrupt file, missing engine, ...
        raise ValueError(f"Could not read the file: {exc}") from exc

    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    if df.empty:
        raise ValueError("The file contains no data rows.")
    return df


def get_excel_sheets(file_bytes: bytes) -> list[str]:
    """Return the sheet names of an Excel workbook."""
    return pd.ExcelFile(io.BytesIO(file_bytes)).sheet_names


def clean_numeric(series: pd.Series) -> pd.Series:
    """Convert a column to numbers, stripping symbols such as '$', ',' or 'Rs'."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.astype(str).str.replace(r"[^\d.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def parse_dates(series: pd.Series, dayfirst: bool = False) -> pd.Series:
    """Convert a column to datetime (invalid values become NaT) without timezone info."""
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = series
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(series, errors="coerce", dayfirst=dayfirst, utc=True)
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed


def date_parse_ratio(series: pd.Series, dayfirst: bool) -> float:
    """Share of (non-empty) values that can be parsed as dates; 0 for numeric columns."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return 1.0
    if pd.api.types.is_numeric_dtype(series):
        return 0.0
    sample = series.dropna().head(200)
    if sample.empty:
        return 0.0
    return float(parse_dates(sample, dayfirst).notna().mean())


def _looks_like_date(series: pd.Series) -> bool:
    """True if (nearly) all values parse as dates in month-first OR day-first notation."""
    return max(date_parse_ratio(series, False), date_parse_ratio(series, True)) >= 0.8


def suggest_dayfirst(series: pd.Series) -> bool:
    """
    Decide whether dates like 05/03/2024 should be read day-first.
    Evidence: a first number above 12 (25/03/2024) can only be a day; a second number above 12
    (03/25/2024) can only be a day in month-first data. Without evidence we default to month-first.
    """
    if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_numeric_dtype(series):
        return False
    parts = series.dropna().head(500).astype(str).str.extract(r"^\s*(\d{1,2})[/.\-](\d{1,2})[/.\-]\d{2,4}")
    parts = parts.dropna().astype(int)
    if parts.empty:
        return False
    return bool((parts[0] > 12).any() and not (parts[1] > 12).any())


def _find_by_hints(columns, hints, used):
    """Return the first column whose name contains one of the hints (hint order = priority)."""
    for hint in hints:
        for col in columns:
            if col not in used and hint in str(col).lower():
                return col
    return None


def detect_columns(df: pd.DataFrame) -> dict:
    """Guess the date, sales, product, category and region columns. Missing ones are None."""
    found = {"date": None, "sales": None, "product": None, "category": None, "region": None}
    used: set = set()

    # --- date column: name hint + must look like dates, otherwise any date-like column
    date_candidates = [c for c in df.columns if _looks_like_date(df[c])]
    found["date"] = _find_by_hints(date_candidates, DATE_HINTS, used) or (
        date_candidates[0] if date_candidates else None)
    if found["date"] is not None:
        used.add(found["date"])

    # --- sales column: numeric (or numeric-looking) column
    numeric_candidates = [
        c for c in df.columns
        if c not in used and (pd.api.types.is_numeric_dtype(df[c]) or clean_numeric(df[c].head(200)).notna().mean() >= 0.8)
    ]
    found["sales"] = _find_by_hints(numeric_candidates, SALES_HINTS, used)
    if found["sales"] is None:  # fall back to the first numeric column that is not an ID-like column
        skip = ("id", "code", "zip", "postal", "year", "month", "day", "qty", "quantity")
        fallback = [c for c in numeric_candidates if not any(s in str(c).lower() for s in skip)]
        found["sales"] = fallback[0] if fallback else None
    if found["sales"] is not None:
        used.add(found["sales"])

    # --- descriptive (text) columns
    text_cols = [c for c in df.columns if c not in used and not pd.api.types.is_numeric_dtype(df[c])
                 and not pd.api.types.is_datetime64_any_dtype(df[c])]
    for key, hints in (("category", CATEGORY_HINTS), ("region", REGION_HINTS), ("product", PRODUCT_HINTS)):
        col = _find_by_hints(text_cols, hints, used)
        found[key] = col
        if col is not None:
            used.add(col)
    return found


def preprocess_data(df: pd.DataFrame, date_col: str, sales_col: str, dayfirst: bool = False,
                    numeric_fill: str = "median", drop_duplicates: bool = True):
    """
    Clean the raw data and add calendar features.

    Steps: remove duplicates -> parse dates -> convert sales to numbers -> drop rows where the
    date or sales value is unusable -> fill other missing values -> add Year/Month/Day/Weekday.

    Returns (clean_dataframe, log_dictionary).
    """
    if date_col == sales_col:
        raise ValueError("The date column and the sales column must be different.")
    df = df.copy()
    log = {"rows_raw": len(df), "duplicates_removed": 0, "invalid_dates_removed": 0,
           "invalid_sales_removed": 0, "missing_filled": {}}

    # 1. duplicates
    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        log["duplicates_removed"] = before - len(df)

    # 2. dates
    df[date_col] = parse_dates(df[date_col], dayfirst=dayfirst)
    log["invalid_dates_removed"] = int(df[date_col].isna().sum())
    df = df.dropna(subset=[date_col])

    # 3. sales values
    df[sales_col] = clean_numeric(df[sales_col])
    log["invalid_sales_removed"] = int(df[sales_col].isna().sum())
    df = df.dropna(subset=[sales_col])

    if df.empty:
        raise ValueError("No valid rows left after cleaning. Please check that you selected the "
                         "correct date and sales columns (and the day-first option for dates).")

    # 4. remaining missing values: numbers -> median/mean/0, text -> "Unknown"
    for col in df.columns:
        if col in (date_col, sales_col):
            continue
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            value = {"median": df[col].median(), "mean": df[col].mean(), "zero": 0}[numeric_fill]
            df[col] = df[col].fillna(0 if pd.isna(value) else value)
        else:
            df[col] = df[col].fillna("Unknown")
        log["missing_filled"][col] = n_missing

    # 5. calendar features + chronological order
    df = df.sort_values(date_col, kind="stable").reset_index(drop=True)
    df["Year"] = df[date_col].dt.year
    df["Month"] = df[date_col].dt.month
    df["Day"] = df[date_col].dt.day
    df["Weekday"] = df[date_col].dt.day_name()

    log["rows_clean"] = len(df)
    return df, log


def compute_kpis(df: pd.DataFrame, sales_col: str) -> dict:
    """Headline numbers for the dashboard cards."""
    s = df[sales_col]
    return {"total": float(s.sum()), "average": float(s.mean()), "highest": float(s.max()),
            "lowest": float(s.min()), "transactions": int(len(df))}


def aggregate_sales(df: pd.DataFrame, date_col: str, sales_col: str, freq: str = "Daily") -> pd.DataFrame:
    """Sum sales per day ('Daily') or per month ('Monthly'). Periods with no sales become 0."""
    rule = FREQ_CONFIG[freq]["rule"]
    series = df.set_index(date_col)[sales_col].resample(rule).sum()
    out = series.reset_index()
    out.columns = ["date", "sales"]
    return out


# =============================================================================
# 3. MACHINE LEARNING
# =============================================================================
def get_models() -> dict:
    """Fresh (untrained) instances of the three models."""
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=2, random_state=42),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, min_samples_leaf=3, random_state=42),
    }


def calendar_features(index: pd.DatetimeIndex, freq: str) -> pd.DataFrame:
    """Calendar-based features for each date in the index."""
    feats = pd.DataFrame(index=index)
    feats["year"] = index.year
    feats["month"] = index.month
    if freq == "Daily":
        feats["day"] = index.day
        feats["dayofweek"] = index.dayofweek
        feats["dayofyear"] = index.dayofyear
        feats["is_weekend"] = (index.dayofweek >= 5).astype(int)
    else:
        feats["quarter"] = index.quarter
    return feats


def feature_columns(freq: str) -> list[str]:
    """Ordered list of model input columns (calendar + time index + lags + rolling mean)."""
    cfg = FREQ_CONFIG[freq]
    calendar = list(calendar_features(pd.DatetimeIndex(["2024-01-01"]), freq).columns)
    return calendar + ["t"] + [f"lag_{k}" for k in cfg["lags"]] + [f"roll_mean_{cfg['roll']}"]


def build_training_frame(series_df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Turn the (date, sales) series into a supervised-learning table.
    Lag features only look at *previous* periods, so no future information leaks in.
    """
    cfg = FREQ_CONFIG[freq]
    s = series_df.set_index("date")["sales"]
    frame = calendar_features(s.index, freq)
    frame["t"] = np.arange(len(s))  # running time index (captures the long-term trend)
    for lag in cfg["lags"]:
        frame[f"lag_{lag}"] = s.shift(lag)
    frame[f"roll_mean_{cfg['roll']}"] = s.shift(1).rolling(cfg["roll"]).mean()
    frame["target"] = s
    return frame.dropna()


def regression_metrics(y_true, y_pred) -> dict:
    """MAE, MSE, RMSE and R² for one set of predictions."""
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": float(mean_absolute_error(y_true, y_pred)), "MSE": float(mse),
            "RMSE": float(np.sqrt(mse)), "R²": float(r2_score(y_true, y_pred))}


def check_series_length(series_df: pd.DataFrame, freq: str) -> None:
    """Raise a friendly error if there is too little history to train on."""
    need = FREQ_CONFIG[freq]["min_points"]
    if len(series_df) < need:
        raise ValueError(f"Only {len(series_df)} {FREQ_CONFIG[freq]['unit']} of history were found, but at least "
                         f"{need} are needed for {freq.lower()} forecasting. Try a different frequency or a "
                         f"longer date range.")


def train_and_evaluate(series_df: pd.DataFrame, freq: str, test_size: float = 0.2) -> dict:
    """
    Train all models on the *earlier* part of the history and test them on the *latest* part
    (chronological split – the data is never shuffled, so the models never see the future).
    """
    check_series_length(series_df, freq)
    frame = build_training_frame(series_df, freq)
    feats = feature_columns(freq)

    split = int(round(len(frame) * (1 - test_size)))
    if split < 8 or len(frame) - split < 2:
        raise ValueError("Not enough data for a train/test split. Use a smaller test size or more history.")
    train, test = frame.iloc[:split], frame.iloc[split:]

    results = {}
    for name, model in get_models().items():
        model.fit(train[feats], train["target"])
        pred = np.clip(model.predict(test[feats]), 0, None)  # sales cannot be negative
        results[name] = {
            "metrics": regression_metrics(test["target"], pred),
            "predictions": pd.DataFrame({"date": test.index, "actual": test["target"].values, "predicted": pred}),
            "model": model,
        }

    # Reference point: "tomorrow will look like today" (previous-period baseline)
    baseline = regression_metrics(test["target"], test["lag_1"])
    best = min(results, key=lambda m: results[m]["metrics"]["RMSE"])
    return {"results": results, "baseline": baseline, "best_model": best, "freq": freq, "test_size": test_size,
            "train_rows": len(train), "test_rows": len(test), "split_date": test.index[0],
            "feature_names": feats}


def forecast_future(model_name: str, series_df: pd.DataFrame, freq: str, horizon: int) -> pd.DataFrame:
    """
    Train the chosen model on ALL history, then predict one period at a time.
    Each prediction is fed back in as history for the next period (recursive forecasting).
    """
    check_series_length(series_df, freq)
    cfg = FREQ_CONFIG[freq]
    feats = feature_columns(freq)

    frame = build_training_frame(series_df, freq)
    model = get_models()[model_name]
    model.fit(frame[feats], frame["target"])

    history = series_df["sales"].astype(float).tolist()
    last_date = series_df["date"].iloc[-1]
    future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=cfg["rule"])[1:]
    calendar = calendar_features(future_dates, freq)

    predictions = []
    for i in range(horizon):
        row = calendar.iloc[i].to_dict()
        row["t"] = len(series_df) + i
        for lag in cfg["lags"]:
            row[f"lag_{lag}"] = history[-lag]
        row[f"roll_mean_{cfg['roll']}"] = float(np.mean(history[-cfg["roll"]:]))
        value = max(float(model.predict(pd.DataFrame([row])[feats])[0]), 0.0)
        history.append(value)
        predictions.append(value)

    return pd.DataFrame({"date": future_dates, "predicted_sales": np.round(predictions, 2)})


# =============================================================================
# 4. CHARTS
# =============================================================================
def style_fig(fig, title: str = "", height: int = 420):
    """Apply a consistent look to every Plotly figure."""
    fig.update_layout(template="plotly_white", height=height, colorway=PALETTE,
                      margin=dict(l=20, r=20, t=60 if title else 30, b=20))
    if title:
        fig.update_layout(title=title)
    return fig


def trend_chart(agg: pd.DataFrame, title: str, ma_window: int = 0):
    """Line chart of aggregated sales, with an optional moving average."""
    fig = px.line(agg, x="date", y="sales", labels={"date": "Date", "sales": "Sales"})
    fig.update_traces(line_color=PALETTE[0], name="Sales", showlegend=True)
    if ma_window:
        fig.add_trace(go.Scatter(x=agg["date"], y=agg["sales"].rolling(ma_window, min_periods=1).mean(),
                                 mode="lines", name=f"{ma_window}-day average",
                                 line=dict(color=PALETTE[2], width=2, dash="dash")))
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return style_fig(fig, title)


def sales_by_group_chart(df, group_col, sales_col, top_n=10, horizontal=False, title=""):
    """Bar chart of total sales for the top-N values of a descriptive column."""
    grouped = df.groupby(group_col)[sales_col].sum().sort_values(ascending=False).head(top_n).reset_index()
    grouped[group_col] = grouped[group_col].astype(str)
    if horizontal:
        fig = px.bar(grouped, x=sales_col, y=group_col, orientation="h", text_auto=".3s")
        fig.update_yaxes(autorange="reversed")
    else:
        fig = px.bar(grouped, x=group_col, y=sales_col, text_auto=".3s")
    fig.update_traces(marker_color=PALETTE[0])
    return style_fig(fig, title)


def share_donut(df, group_col, sales_col, top_n=8, title=""):
    """Donut chart showing each group's share of total sales."""
    grouped = df.groupby(group_col)[sales_col].sum().sort_values(ascending=False).head(top_n).reset_index()
    grouped[group_col] = grouped[group_col].astype(str)
    fig = px.pie(grouped, names=group_col, values=sales_col, hole=0.45, color_discrete_sequence=PALETTE)
    return style_fig(fig, title)


def weekday_chart(daily: pd.DataFrame):
    """Average daily sales for each weekday."""
    tmp = daily.copy()
    tmp["weekday"] = tmp["date"].dt.day_name()
    avg = tmp.groupby("weekday")["sales"].mean().reindex(WEEKDAYS).reset_index()
    fig = px.bar(avg, x="weekday", y="sales", labels={"weekday": "", "sales": "Average daily sales"})
    fig.update_traces(marker_color=PALETTE[1])
    return style_fig(fig, "Average Sales by Weekday")


def actual_vs_predicted_chart(results: dict, selected: list[str]):
    """Actual sales vs the predictions of the chosen models on the test period."""
    fig = go.Figure()
    first = results[selected[0]]["predictions"]
    fig.add_trace(go.Scatter(x=first["date"], y=first["actual"], mode="lines", name="Actual",
                             line=dict(color="#111827", width=2.5)))
    for name in selected:
        pred = results[name]["predictions"]
        fig.add_trace(go.Scatter(x=pred["date"], y=pred["predicted"], mode="lines", name=f"{name} (predicted)",
                                 line=dict(color=MODEL_COLORS.get(name, PALETTE[0]), width=2, dash="dash")))
    fig.update_layout(xaxis_title="Date", yaxis_title="Sales",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return style_fig(fig, "Actual vs Predicted Sales (test period)")


def metrics_chart(metrics_table: pd.DataFrame):
    """One small bar chart per metric (each has its own y-axis because the scales differ)."""
    long = metrics_table.melt(id_vars="Model", value_vars=["MAE", "MSE", "RMSE", "R²"],
                              var_name="Metric", value_name="Value")
    fig = px.bar(long, x="Model", y="Value", color="Model", facet_col="Metric", facet_col_wrap=2,
                 color_discrete_map=MODEL_COLORS)
    fig.update_yaxes(matches=None, showticklabels=True, title_text="")
    fig.update_xaxes(showticklabels=False, title_text="")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return style_fig(fig, "Model Comparison", height=520)


def forecast_chart(history: pd.DataFrame, forecast: pd.DataFrame, model_name: str):
    """Recent history plus the forecast. The forecast line starts at the last real data point."""
    last = history.iloc[[-1]]
    connect_x = pd.concat([last["date"], forecast["date"]])
    connect_y = pd.concat([last["sales"], forecast["predicted_sales"]])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["date"], y=history["sales"], mode="lines", name="Historical sales",
                             line=dict(color=PALETTE[0], width=2)))
    fig.add_trace(go.Scatter(x=connect_x, y=connect_y, mode="lines+markers", name=f"Forecast ({model_name})",
                             line=dict(color=PALETTE[2], width=2.5, dash="dash"), marker=dict(size=5)))
    fig.update_layout(xaxis_title="Date", yaxis_title="Sales",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return style_fig(fig, "Sales Forecast", height=460)


def importance_chart(model, feature_names):
    """Feature importance bars for tree-based models."""
    imp = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
    imp = imp.sort_values("importance")
    fig = px.bar(imp, x="importance", y="feature", orientation="h")
    fig.update_traces(marker_color=PALETTE[3])
    return style_fig(fig, "Feature Importance", height=380)


def distribution_figure(df: pd.DataFrame, sales_col: str):
    """Seaborn histogram of transaction values."""
    fig, ax = plt.subplots(figsize=(6, 3.8))
    try:
        sns.histplot(df[sales_col], bins=40, kde=True, color=PALETTE[0], ax=ax)
    except Exception:  # e.g. KDE fails on constant data
        sns.histplot(df[sales_col], bins=40, kde=False, color=PALETTE[0], ax=ax)
    ax.set_title("Distribution of Sales Values")
    ax.set_xlabel(sales_col)
    fig.tight_layout()
    return fig


def correlation_figure(df: pd.DataFrame, sales_col: str):
    """Seaborn correlation heatmap of numeric columns (None if there are fewer than two)."""
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return None
    cols = [sales_col] + [c for c in numeric.columns if c != sales_col][:11]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    sns.heatmap(numeric[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                annot_kws={"size": 8})
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    return fig


# =============================================================================
# 5. REPORT GENERATION
# =============================================================================
def fmt_money(value, symbol: str = "$") -> str:
    """Format a number as money, e.g. $1,234,567 or $12.50."""
    if value is None or pd.isna(value):
        return "N/A"
    sign = "-" if value < 0 else ""
    value = abs(value)
    return f"{sign}{symbol}{value:,.0f}" if value >= 1000 else f"{sign}{symbol}{value:,.2f}"


def md_table(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a Markdown table (no extra dependency needed)."""
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    divider = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, divider] + rows)


def metrics_table(evaluation: dict) -> pd.DataFrame:
    """Model comparison table, sorted from best to worst RMSE."""
    rows = [{"Model": name, **res["metrics"]} for name, res in evaluation["results"].items()]
    table = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    table.insert(0, "Rank", range(1, len(table) + 1))
    return table


def generate_report(df, cfg, log, evaluation=None, forecast=None, forecast_meta=None, symbol="$") -> str:
    """Build a Markdown sales-analysis report from everything done in the app so far."""
    date_col, sales_col = cfg["date"], cfg["sales"]
    kpis = compute_kpis(df, sales_col)
    monthly = aggregate_sales(df, date_col, sales_col, "Monthly")
    daily = aggregate_sales(df, date_col, sales_col, "Daily")
    L: list[str] = []

    L += ["# Sales Analysis Report", f"*Generated on {datetime.now():%Y-%m-%d %H:%M}*", ""]

    L += ["## 1. Dataset Overview",
          f"- **Records analysed:** {len(df):,}",
          f"- **Columns:** {df.shape[1]}",
          f"- **Period covered:** {df[date_col].min():%Y-%m-%d} to {df[date_col].max():%Y-%m-%d}",
          f"- **Date column:** {date_col}  |  **Sales column:** {sales_col}", ""]

    L += ["## 2. Data Cleaning Summary",
          f"- Raw rows: {log['rows_raw']:,}",
          f"- Duplicate rows removed: {log['duplicates_removed']:,}",
          f"- Rows removed (invalid / missing date): {log['invalid_dates_removed']:,}",
          f"- Rows removed (invalid / missing sales): {log['invalid_sales_removed']:,}",
          f"- Rows after cleaning: {log['rows_clean']:,}"]
    if log["missing_filled"]:
        filled = ", ".join(f"{c} ({n})" for c, n in log["missing_filled"].items())
        L.append(f"- Missing values filled: {filled}")
    L.append("")

    L += ["## 3. Key Performance Indicators",
          f"- **Total sales:** {fmt_money(kpis['total'], symbol)}",
          f"- **Average sale per transaction:** {fmt_money(kpis['average'], symbol)}",
          f"- **Highest sale:** {fmt_money(kpis['highest'], symbol)}",
          f"- **Lowest sale:** {fmt_money(kpis['lowest'], symbol)}",
          f"- **Total transactions:** {kpis['transactions']:,}", ""]

    L.append("## 4. Insights")
    best_m, worst_m = monthly.loc[monthly["sales"].idxmax()], monthly.loc[monthly["sales"].idxmin()]
    L.append(f"- **Best month:** {best_m['date']:%B %Y} ({fmt_money(best_m['sales'], symbol)})")
    partial_note = ""
    if worst_m["date"] in (monthly["date"].iloc[0], monthly["date"].iloc[-1]):
        partial_note = " – this is the first/last month in the data and may be incomplete"
    L.append(f"- **Weakest month:** {worst_m['date']:%B %Y} ({fmt_money(worst_m['sales'], symbol)}){partial_note}")
    weekday_avg = daily.assign(w=daily["date"].dt.day_name()).groupby("w")["sales"].mean()
    L.append(f"- **Strongest weekday (average daily sales):** {weekday_avg.idxmax()}")
    if len(monthly) >= 6:
        last3, prev3 = monthly["sales"].tail(3).mean(), monthly["sales"].iloc[-6:-3].mean()
        if prev3 > 0:
            L.append(f"- **Last 3 months vs the 3 months before:** {(last3 / prev3 - 1) * 100:+.1f}% on average "
                     f"(not adjusted for seasonality)")
    for key, label in (("product", "Top product"), ("category", "Top category"), ("region", "Top region")):
        col = cfg.get(key)
        if col:
            totals = df.groupby(col)[sales_col].sum()
            L.append(f"- **{label}:** {totals.idxmax()} ({fmt_money(totals.max(), symbol)}, "
                     f"{totals.max() / totals.sum() * 100:.1f}% of sales)")
    L.append("")

    L.append("## 5. Model Evaluation")
    if evaluation:
        table = metrics_table(evaluation).round({"MAE": 2, "MSE": 2, "RMSE": 2, "R²": 4})
        L += [f"Chronological split: trained on {evaluation['train_rows']} {evaluation['freq'].lower()} periods, "
              f"tested on the latest {evaluation['test_rows']} (from {evaluation['split_date']:%Y-%m-%d}).", "",
              md_table(table), "",
              f"**Best model (lowest RMSE): {evaluation['best_model']}**", ""]
    else:
        L += ["*Models have not been evaluated yet (use the Model Performance page).*", ""]

    L.append("## 6. Sales Forecast")
    if forecast is not None and forecast_meta:
        total = forecast["predicted_sales"].sum()
        peak = forecast.loc[forecast["predicted_sales"].idxmax()]
        L += [f"- **Model:** {forecast_meta['model']}  |  **Horizon:** {forecast_meta['horizon']} "
              f"{FREQ_CONFIG[forecast_meta['freq']]['unit']}",
              f"- **Total predicted sales:** {fmt_money(total, symbol)}",
              f"- **Average per period:** {fmt_money(forecast['predicted_sales'].mean(), symbol)}",
              f"- **Peak period:** {peak['date']:%Y-%m-%d} ({fmt_money(peak['predicted_sales'], symbol)})", ""]
        preview = forecast.head(15).copy()
        preview["date"] = preview["date"].dt.strftime("%Y-%m-%d")
        preview["predicted_sales"] = preview["predicted_sales"].map("{:,.2f}".format)
        L += [md_table(preview.rename(columns={"date": "Date", "predicted_sales": "Predicted Sales"}))]
        if len(forecast) > 15:
            L.append(f"*…and {len(forecast) - 15} more periods (see the downloadable CSV).*")
        L.append("")
    else:
        L += ["*No forecast generated yet (use the Sales Prediction page).*", ""]

    L += ["## 7. Notes",
          "- Forecasts are statistical estimates based on historical patterns; they cannot anticipate "
          "promotions, price changes or other future events.",
          "- Days/months without any transactions are treated as zero sales."]
    return "\n".join(L)


# =============================================================================
# 6. UI HELPERS & PAGES
# =============================================================================
def esc(text: str) -> str:
    """Escape '$' so Streamlit does not treat currency amounts as LaTeX."""
    return text.replace("$", "\\$")


def init_state() -> None:
    """Create all session-state keys on the first run."""
    defaults = {"raw_df": None, "file_name": None, "dataset_id": 0, "last_upload_sig": None,
                "clean_df": None, "cfg": None, "clean_log": None,
                "evaluation": None, "forecast": None, "forecast_meta": None, "currency": "$", "nav": PAGE_HOME}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def optional_col(value: str):
    """Turn the '(none)' choice of an optional selectbox into None."""
    return None if value == "(none)" else value


def go_to(page: str) -> None:
    """Button callback: switch the sidebar navigation to another page."""
    st.session_state["nav"] = page


def set_raw_data(df: pd.DataFrame, name: str) -> None:
    """Store a freshly loaded dataset and reset everything that depends on the previous one."""
    st.session_state.update({"raw_df": df, "file_name": name, "clean_df": None, "cfg": None,
                             "clean_log": None, "evaluation": None, "forecast": None, "forecast_meta": None})
    st.session_state["dataset_id"] += 1


def require_data() -> bool:
    """Show a warning and return False if no cleaned dataset is available yet."""
    if st.session_state["clean_df"] is None:
        st.warning("⚠️ No processed dataset yet. Please go to **📤 Upload Dataset**, load a file and click "
                   "**Clean & prepare data**.")
        st.button("Go to Upload page", on_click=go_to, args=(PAGE_UPLOAD,),
                  key=f"goto_upload_{st.session_state['nav']}")
        return False
    return True


def render_header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="hero"><div class="hero-title">{html.escape(title)}</div>'
                f'<p class="hero-sub">{html.escape(subtitle)}</p></div>', unsafe_allow_html=True)


def kpi_card(col, label: str, value: str, css_class: str) -> None:
    col.markdown(f'<div class="kpi-card {css_class}"><div class="kpi-label">{html.escape(label)}</div>'
                 f'<div class="kpi-value">{html.escape(value)}</div></div>', unsafe_allow_html=True)


def render_kpis(kpis: dict, symbol: str) -> None:
    items = [("Total Sales", fmt_money(kpis["total"], symbol), "kpi-1"),
             ("Average Sales", fmt_money(kpis["average"], symbol), "kpi-2"),
             ("Highest Sales", fmt_money(kpis["highest"], symbol), "kpi-3"),
             ("Lowest Sales", fmt_money(kpis["lowest"], symbol), "kpi-4"),
             ("Total Transactions", f"{kpis['transactions']:,}", "kpi-5")]
    for col, (label, value, css) in zip(st.columns(len(items)), items):
        kpi_card(col, label, value, css)


def show_figure(fig) -> None:
    """Display a Matplotlib figure and free its memory."""
    if fig is None:
        return
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------- Home page
def page_home() -> None:
    render_header("📈 " + APP_TITLE, "Upload your sales data, explore it, compare ML models and forecast the future.")
    df, cfg, symbol = st.session_state["clean_df"], st.session_state["cfg"], st.session_state["currency"]

    if df is None:
        if st.session_state["raw_df"] is None:
            st.info("👋 Welcome! No dataset is loaded yet. Follow these four steps to get started.")
        else:
            st.info("Your file is loaded but not cleaned yet. Open **Upload Dataset** and click "
                    "**Clean & prepare data**.")
        steps = [("1️⃣ Upload", "Upload a CSV or Excel file — or load the built-in sample dataset."),
                 ("2️⃣ Clean", "Confirm the detected columns and clean the data in one click."),
                 ("3️⃣ Analyse", "Explore KPIs and interactive charts for products, categories and regions."),
                 ("4️⃣ Predict", "Compare three ML models and forecast future sales.")]
        for col, (title, text) in zip(st.columns(4), steps):
            col.markdown(f'<div class="step-card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)
        st.write("")
        st.button("🚀 Go to Upload page", on_click=go_to, args=(PAGE_UPLOAD,))
        return

    date_col, sales_col = cfg["date"], cfg["sales"]
    st.subheader("Business Overview")
    render_kpis(compute_kpis(df, sales_col), symbol)

    left, right = st.columns([2, 1])
    with left:
        monthly = aggregate_sales(df, date_col, sales_col, "Monthly")
        st.plotly_chart(trend_chart(monthly, "Monthly Sales Trend"))
    with right:
        st.markdown("##### Dataset at a glance")
        st.write(f"**File:** {st.session_state['file_name']}")
        st.write(f"**Records:** {len(df):,}")
        st.write(f"**Period:** {df[date_col].min():%d %b %Y} → {df[date_col].max():%d %b %Y}")
        for key, label in (("product", "Top product"), ("category", "Top category"), ("region", "Top region")):
            if cfg.get(key):
                st.write(f"**{label}:** {df.groupby(cfg[key])[sales_col].sum().idxmax()}")

    st.subheader("Project Progress")
    evaluation, forecast = st.session_state["evaluation"], st.session_state["forecast"]
    cols = st.columns(4)
    cols[0].success("✅ Data loaded")
    cols[1].success("✅ Data cleaned")
    if evaluation:
        cols[2].success(f"✅ Models evaluated (best: {evaluation['best_model']})")
    else:
        cols[2].info("⏳ Models not evaluated")
    if forecast is not None:
        cols[3].success("✅ Forecast generated")
    else:
        cols[3].info("⏳ No forecast yet")


# ---------------------------------------------------------------- Upload page
def page_upload() -> None:
    render_header("📤 Upload Dataset", "Load your sales data, check its quality and prepare it for analysis.")

    # ---- 1. choose a source
    uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"],
                                help="The file needs at least a date column and a sales/revenue column.")
    if st.button("📦 Or load the sample dataset"):
        try:
            set_raw_data(load_dataset(SAMPLE_FILE.read_bytes(), SAMPLE_FILE.name), SAMPLE_FILE.name)
            st.success("Sample dataset loaded.")
        except Exception as exc:
            st.error(f"Could not load the sample dataset: {exc}")

    if uploaded is not None:
        file_bytes, sheet = uploaded.getvalue(), None
        try:
            if Path(uploaded.name).suffix.lower() in (".xlsx", ".xls"):
                sheets = get_excel_sheets(file_bytes)
                if len(sheets) > 1:
                    sheet = st.selectbox("Excel sheet to use", sheets)
            signature = (uploaded.name, len(file_bytes), sheet)
            if signature != st.session_state["last_upload_sig"]:  # only (re)load when something changed
                set_raw_data(load_dataset(file_bytes, uploaded.name, sheet), uploaded.name)
                st.session_state["last_upload_sig"] = signature
                st.success(f"✅ **{uploaded.name}** uploaded successfully.")
        except Exception as exc:
            st.error(f"❌ {exc}")

    raw = st.session_state["raw_df"]
    if raw is None:
        st.info("Upload a file above, or click the sample-dataset button to try the app.")
        return

    # ---- 2. dataset overview
    st.subheader(f"Dataset: {st.session_state['file_name']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{raw.shape[0]:,}")
    c2.metric("Columns", raw.shape[1])
    c3.metric("Missing values", f"{int(raw.isna().sum().sum()):,}")
    c4.metric("Duplicate rows", f"{int(raw.duplicated().sum()):,}")
    st.dataframe(raw.head(100))
    with st.expander("Column details (types and missing values)"):
        info = pd.DataFrame({"Column": raw.columns, "Type": raw.dtypes.astype(str).values,
                             "Missing": raw.isna().sum().values,
                             "Missing %": (raw.isna().mean() * 100).round(2).values,
                             "Unique values": raw.nunique().values})
        st.dataframe(info, hide_index=True)

    # ---- 3. column mapping & cleaning options
    st.subheader("Select columns & clean the data")
    detected = detect_columns(raw)
    all_cols, optional = list(raw.columns), ["(none)"] + list(raw.columns)
    uid = st.session_state["dataset_id"]  # new keys for each dataset -> widgets reset to detected values
    if detected["date"] is None or detected["sales"] is None:
        st.warning("Could not automatically detect the date and/or sales column — please choose them below.")
    else:
        st.success(f"🔍 Auto-detected  →  date: **{detected['date']}**, sales: **{detected['sales']}**")

    def pick(container, label, options, key, default):
        index = options.index(default) if default in options else 0
        return container.selectbox(label, options, index=index, key=f"{key}_{uid}")

    a, b = st.columns(2)
    date_col = pick(a, "📅 Date column (required)", all_cols, "map_date", detected["date"])
    sales_col = pick(b, "💰 Sales column (required)", all_cols, "map_sales", detected["sales"])
    p, q, r = st.columns(3)
    product_col = pick(p, "Product column (optional)", optional, "map_product", detected["product"])
    category_col = pick(q, "Category column (optional)", optional, "map_category", detected["category"])
    region_col = pick(r, "Region column (optional)", optional, "map_region", detected["region"])

    o1, o2, o3 = st.columns(3)
    dayfirst_default = suggest_dayfirst(raw[date_col])
    dayfirst = o1.checkbox("Dates are day-first (31/12/2024)", value=dayfirst_default,
                           key=f"dayfirst_{uid}_{date_col}",
                           help="Tick this if your dates look like DD/MM/YYYY. It is pre-ticked when the data clearly "
                                "needs it, but ambiguous dates such as 03/04/2024 need your decision.")
    numeric_fill = o2.selectbox("Fill missing numbers with", ["median", "mean", "zero"], key=f"fill_{uid}")
    drop_dupes = o3.checkbox("Remove duplicate rows", value=True, key=f"dupes_{uid}")

    if st.button("🧹 Clean & prepare data", type="primary"):
        try:
            with st.spinner("Cleaning data…"):
                clean, log = preprocess_data(raw, date_col, sales_col, dayfirst, numeric_fill, drop_dupes)
            st.session_state.update({
                "clean_df": clean, "clean_log": log, "evaluation": None, "forecast": None, "forecast_meta": None,
                "cfg": {"date": date_col, "sales": sales_col, "product": optional_col(product_col),
                        "category": optional_col(category_col), "region": optional_col(region_col)}})
            st.success(f"✅ Data prepared: {log['rows_clean']:,} clean rows are ready for analysis.")
        except Exception as exc:
            st.error(f"❌ {exc}")

    # ---- 4. cleaning results
    clean, log = st.session_state["clean_df"], st.session_state["clean_log"]
    if clean is not None:
        st.subheader("Cleaning summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Duplicates removed", log["duplicates_removed"])
        m2.metric("Invalid dates removed", log["invalid_dates_removed"])
        m3.metric("Invalid sales removed", log["invalid_sales_removed"])
        m4.metric("Final rows", f"{log['rows_clean']:,}")
        if log["missing_filled"]:
            st.info("Missing values filled → " + ", ".join(f"{c}: {n}" for c, n in log["missing_filled"].items()))
        st.markdown("**Processed data preview** (with the new Year, Month, Day and Weekday columns)")
        st.dataframe(clean.head(100))
        st.button("Continue to Data Analysis ➜", on_click=go_to, args=(PAGE_ANALYSIS,))


# ---------------------------------------------------------------- Analysis page
def page_analysis() -> None:
    render_header("📊 Data Analysis", "KPIs and interactive charts that show how your sales behave.")
    if not require_data():
        return
    df, cfg, symbol = st.session_state["clean_df"], st.session_state["cfg"], st.session_state["currency"]
    date_col, sales_col = cfg["date"], cfg["sales"]

    # Date-range filter
    min_d, max_d = df[date_col].min().date(), df[date_col].max().date()
    rng = st.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    start, end = rng if isinstance(rng, (tuple, list)) and len(rng) == 2 else (min_d, max_d)
    data = df[(df[date_col] >= pd.Timestamp(start)) & (df[date_col] < pd.Timestamp(end) + pd.Timedelta(days=1))]
    if data.empty:
        st.warning("No records in the selected date range.")
        return

    render_kpis(compute_kpis(data, sales_col), symbol)
    tabs = st.tabs(["📈 Trends", "📦 Products & Categories", "🌍 Regions", "📐 Distribution", "🎯 Actual vs Predicted"])

    with tabs[0]:
        monthly = aggregate_sales(data, date_col, sales_col, "Monthly")
        daily = aggregate_sales(data, date_col, sales_col, "Daily")
        st.plotly_chart(trend_chart(monthly, "Monthly Sales Trend"))
        st.plotly_chart(trend_chart(daily, "Daily Sales Trend", ma_window=7))
        st.plotly_chart(weekday_chart(daily))

    with tabs[1]:
        if not cfg["product"] and not cfg["category"]:
            st.info("No product or category column was selected on the Upload page.")
        if cfg["product"]:
            top_n = st.slider("Number of products to show", 3, 25, 10)
            st.plotly_chart(sales_by_group_chart(data, cfg["product"], sales_col, top_n, True,
                                                 f"Top {top_n} Products by Sales"))
        if cfg["category"]:
            c1, c2 = st.columns(2)
            c1.plotly_chart(sales_by_group_chart(data, cfg["category"], sales_col, 15, False, "Sales by Category"))
            c2.plotly_chart(share_donut(data, cfg["category"], sales_col, 8, "Category Share"))

    with tabs[2]:
        if cfg["region"]:
            c1, c2 = st.columns(2)
            c1.plotly_chart(sales_by_group_chart(data, cfg["region"], sales_col, 15, False, "Sales by Region"))
            c2.plotly_chart(share_donut(data, cfg["region"], sales_col, 8, "Region Share"))
        else:
            st.info("No region column was selected on the Upload page.")

    with tabs[3]:
        c1, c2 = st.columns(2)
        with c1:
            show_figure(distribution_figure(data, sales_col))
        with c2:
            fig = correlation_figure(data, sales_col)
            if fig is not None:
                show_figure(fig)
            else:
                st.info("Need at least two numeric columns for a heatmap.")

    with tabs[4]:
        evaluation = st.session_state["evaluation"]
        if evaluation is None:
            st.info("Evaluate the models first on the **🏆 Model Performance** page to see this chart.")
        else:
            names = list(evaluation["results"])
            choice = st.selectbox("Model", names, index=names.index(evaluation["best_model"]), key="ava_model")
            st.plotly_chart(actual_vs_predicted_chart(evaluation["results"], [choice]))


# ---------------------------------------------------------------- Prediction page
def page_prediction() -> None:
    render_header("🔮 Sales Prediction", "Choose a model and a horizon, then generate a future sales forecast.")
    if not require_data():
        return
    df, cfg, symbol = st.session_state["clean_df"], st.session_state["cfg"], st.session_state["currency"]
    evaluation = st.session_state["evaluation"]
    models = list(get_models())

    if evaluation:
        st.info(f"💡 In your last evaluation, **{evaluation['best_model']}** had the lowest error.")
    c1, c2, c3 = st.columns(3)
    model_name = c1.selectbox("Prediction model", models,
                              index=models.index(evaluation["best_model"]) if evaluation else 1)
    freq = c2.radio("Forecast frequency", list(FREQ_CONFIG), horizontal=True)
    fc = FREQ_CONFIG[freq]
    horizon = c3.slider(f"How many {fc['unit']} to predict?", 1, fc["max_horizon"], fc["default_horizon"],
                        key=f"horizon_{freq}")

    if st.button("🚀 Generate forecast", type="primary"):
        try:
            with st.spinner("Training model and forecasting…"):
                series = aggregate_sales(df, cfg["date"], cfg["sales"], freq)
                forecast = forecast_future(model_name, series, freq, horizon)
            st.session_state["forecast"] = forecast
            st.session_state["forecast_meta"] = {"model": model_name, "freq": freq, "horizon": horizon,
                                                 "generated": datetime.now().strftime("%Y-%m-%d %H:%M")}
            st.success(f"✅ Forecast generated for the next {horizon} {fc['unit']} using {model_name}.")
        except Exception as exc:
            st.error(f"❌ {exc}")

    forecast, meta = st.session_state["forecast"], st.session_state["forecast_meta"]
    if forecast is None:
        st.info("Pick your settings and click **Generate forecast**.")
        return

    # Summary cards
    peak = forecast.loc[forecast["predicted_sales"].idxmax()]
    k1, k2, k3, k4 = st.columns(4)
    kpi_card(k1, "Total predicted", fmt_money(forecast["predicted_sales"].sum(), symbol), "kpi-1")
    kpi_card(k2, "Average per period", fmt_money(forecast["predicted_sales"].mean(), symbol), "kpi-2")
    kpi_card(k3, "Peak period", f"{peak['date']:%d %b %Y}", "kpi-3")
    kpi_card(k4, "Peak value", fmt_money(peak["predicted_sales"], symbol), "kpi-4")

    # Chart (history uses the same frequency as the forecast)
    series = aggregate_sales(df, cfg["date"], cfg["sales"], meta["freq"])
    history = series.tail(180 if meta["freq"] == "Daily" else 36)
    st.plotly_chart(forecast_chart(history, forecast, meta["model"]))

    # Table + download
    table = forecast.copy()
    table["date"] = table["date"].dt.strftime("%Y-%m-%d" if meta["freq"] == "Daily" else "%Y-%m")
    if meta["freq"] == "Daily":
        table.insert(1, "weekday", pd.to_datetime(table["date"]).dt.day_name())
    table = table.rename(columns={"date": "Date", "weekday": "Weekday", "predicted_sales": "Predicted Sales"})
    st.subheader("Predicted sales")
    st.dataframe(table, hide_index=True)
    st.download_button("⬇️ Download predictions (CSV)", forecast.to_csv(index=False).encode("utf-8"),
                       file_name="predicted_sales.csv", mime="text/csv")


# ---------------------------------------------------------------- Model performance page
def page_performance() -> None:
    render_header("🏆 Model Performance", "Train three models on the past and test them on the most recent data.")
    if not require_data():
        return
    df, cfg = st.session_state["clean_df"], st.session_state["cfg"]

    c1, c2 = st.columns(2)
    freq = c1.radio("Evaluate on", list(FREQ_CONFIG), horizontal=True,
                    help="Daily or monthly totals. Monthly needs at least 18 months of data.")
    test_pct = c2.slider("Test set size (% of the latest data)", 10, 40, 20, step=5)
    st.caption("⏱️ The split is **chronological**: models learn from earlier periods and are tested on the latest "
               "ones, so they never see the future (no data leakage).")

    if st.button("▶️ Train & evaluate models", type="primary"):
        try:
            with st.spinner("Training Linear Regression, Random Forest and Decision Tree…"):
                series = aggregate_sales(df, cfg["date"], cfg["sales"], freq)
                st.session_state["evaluation"] = train_and_evaluate(series, freq, test_pct / 100)
            st.success("✅ All models trained and evaluated.")
        except Exception as exc:
            st.error(f"❌ {exc}")

    ev = st.session_state["evaluation"]
    if ev is None:
        st.info("Click **Train & evaluate models** to see the results.")
        return

    st.markdown(f"**Training set:** {ev['train_rows']} {ev['freq'].lower()} periods  |  "
                f"**Test set:** {ev['test_rows']} periods (from {ev['split_date']:%d %b %Y})")
    table = metrics_table(ev)
    st.success(f"🏆 Best model: **{ev['best_model']}** (lowest RMSE)")
    st.dataframe(table.round({"MAE": 2, "MSE": 2, "RMSE": 2, "R²": 4}), hide_index=True)
    b = ev["baseline"]
    st.caption(f"Reference – naive baseline ('same as previous period'): MAE {b['MAE']:,.2f}, "
               f"RMSE {b['RMSE']:,.2f}, R² {b['R²']:.3f}. A useful model should beat this.")
    st.caption("MAE = average absolute error · MSE = average squared error · RMSE = square root of MSE (same unit "
               "as sales) · R² = share of variance explained (1.0 is perfect; lower is worse).")

    st.plotly_chart(metrics_chart(table))

    names = list(ev["results"])
    choice = st.selectbox("Show predictions for", ["All models"] + names, index=names.index(ev["best_model"]) + 1)
    st.plotly_chart(actual_vs_predicted_chart(ev["results"], names if choice == "All models" else [choice]))

    with st.expander("🔍 Which features matter most? (tree-based models)"):
        tree_name = st.selectbox("Model", ["Random Forest", "Decision Tree"])
        st.plotly_chart(importance_chart(ev["results"][tree_name]["model"], ev["feature_names"]))
        st.caption("lag_N = sales N periods ago · roll_mean = recent average · t = time index (long-term trend).")


# ---------------------------------------------------------------- Reports page
def page_reports() -> None:
    render_header("📥 Download Reports", "Export predictions, the cleaned dataset and a written analysis report.")
    if not require_data():
        return
    df, cfg, log = st.session_state["clean_df"], st.session_state["cfg"], st.session_state["clean_log"]
    forecast, meta, ev = st.session_state["forecast"], st.session_state["forecast_meta"], st.session_state["evaluation"]
    symbol = st.session_state["currency"]
    stamp = datetime.now().strftime("%Y%m%d")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("##### 🔮 Predicted sales")
        if forecast is not None:
            st.download_button("⬇️ Predictions (CSV)", forecast.to_csv(index=False).encode("utf-8"),
                               file_name=f"predicted_sales_{stamp}.csv", mime="text/csv")
        else:
            st.info("Generate a forecast first.")
    with c2:
        st.markdown("##### 🧹 Processed dataset")
        st.download_button("⬇️ Processed data (CSV)", df.to_csv(index=False).encode("utf-8"),
                           file_name=f"processed_sales_data_{stamp}.csv", mime="text/csv")
    with c3:
        st.markdown("##### 🗓️ Monthly summary")
        monthly = aggregate_sales(df, cfg["date"], cfg["sales"], "Monthly")
        monthly["date"] = monthly["date"].dt.strftime("%Y-%m")
        st.download_button("⬇️ Monthly sales (CSV)", monthly.to_csv(index=False).encode("utf-8"),
                           file_name=f"monthly_sales_{stamp}.csv", mime="text/csv")

    st.subheader("Sales analysis report")
    report = generate_report(df, cfg, log, ev, forecast, meta, symbol)
    st.download_button("⬇️ Download full report (Markdown)", report.encode("utf-8"),
                       file_name=f"sales_analysis_report_{stamp}.md", mime="text/markdown", type="primary")
    if ev is None or forecast is None:
        st.caption("Tip: evaluate the models and generate a forecast first for a more complete report.")
    with st.expander("Preview report", expanded=True):
        st.markdown(esc(report))


# =============================================================================
# 7. MAIN
# =============================================================================
def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("## 📈 Sales Predictor")
        st.caption("AI-powered sales forecasting")
        page = st.radio("Navigation", PAGES, key="nav", label_visibility="collapsed")
        st.divider()
        if st.session_state["clean_df"] is not None:
            st.success(f"✅ {st.session_state['file_name']}\n\n{len(st.session_state['clean_df']):,} clean rows")
        elif st.session_state["raw_df"] is not None:
            st.warning("Dataset loaded – not cleaned yet")
        else:
            st.info("No dataset loaded")
        st.text_input("Currency symbol", key="currency", max_chars=4)
        st.caption("Built with Streamlit · scikit-learn · Plotly")
    return page


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()
    pages = {PAGE_HOME: page_home, PAGE_UPLOAD: page_upload, PAGE_ANALYSIS: page_analysis,
             PAGE_PREDICT: page_prediction, PAGE_PERFORMANCE: page_performance, PAGE_REPORTS: page_reports}
    page = render_sidebar()
    try:
        pages[page]()
    except Exception as exc:  # last line of defence: show a friendly message instead of a crash
        st.error(f"Something went wrong: {exc}")
        with st.expander("Technical details"):
            st.exception(exc)


if __name__ == "__main__":
    main()
