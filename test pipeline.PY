"""
Automated tests for the data-processing and machine-learning functions.
Run with:   pip install pytest   then   pytest -q
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import app

SAMPLE = Path(__file__).parent / "sample_sales_data.csv"


@pytest.fixture(scope="module")
def raw():
    return app.load_dataset(SAMPLE.read_bytes(), SAMPLE.name)


@pytest.fixture(scope="module")
def clean(raw):
    return app.preprocess_data(raw, "Date", "Sales")


@pytest.fixture(scope="module")
def daily(clean):
    return app.aggregate_sales(clean[0], "Date", "Sales", "Daily")


# ------------------------------------------------------------------ loading
def test_load_csv_and_excel(raw):
    assert raw.shape[1] == 9 and len(raw) > 1000
    buffer = io.BytesIO()
    raw.head(50).to_excel(buffer, index=False)
    assert app.load_dataset(buffer.getvalue(), "x.xlsx").shape == (50, 9)


@pytest.mark.parametrize("name,data", [("x.txt", b"hello"), ("x.csv", b""), ("x.csv", b"a,b\n")])
def test_bad_files_raise_friendly_error(name, data):
    with pytest.raises(ValueError):
        app.load_dataset(data, name)


def test_column_detection(raw):
    found = app.detect_columns(raw)
    assert (found["date"], found["sales"]) == ("Date", "Sales")
    assert (found["product"], found["category"], found["region"]) == ("Product", "Category", "Region")


def test_day_first_dates_are_detected():
    dates = pd.Series(["15/03/2024", "01/04/2024", "28/02/2024", "03/05/2024"] * 5)
    frame = pd.DataFrame({"Order Date": dates, "Revenue": range(20)})
    assert app.detect_columns(frame)["date"] == "Order Date"
    assert app.suggest_dayfirst(dates) is True
    assert app.suggest_dayfirst(pd.Series(["2024-03-15", "2024-04-01"])) is False
    assert app.suggest_dayfirst(pd.Series(["03/25/2024", "04/01/2024"])) is False  # US style
    # first value is ambiguous (01/03) -> still detected and parsed correctly when day-first is ticked
    shuffled = pd.Series(["01/03/2024", "15/03/2024", "28/03/2024"] * 10)
    assert app.suggest_dayfirst(shuffled) is True
    parsed = app.parse_dates(shuffled, dayfirst=True)
    assert parsed.notna().all() and parsed.iloc[0] == pd.Timestamp(2024, 3, 1)


# ------------------------------------------------------------ preprocessing
def test_cleaning_removes_duplicates_and_fills_missing(raw, clean):
    df, log = clean
    assert log["duplicates_removed"] == 12
    assert df.duplicated().sum() == 0
    assert df.isna().sum().sum() == 0
    assert {"Year", "Month", "Day", "Weekday"} <= set(df.columns)
    assert df["Date"].is_monotonic_increasing


def test_messy_values_are_handled():
    messy = pd.DataFrame({"d": ["31/01/2024", "15/02/2024", "garbage", None],
                          "s": ["$1,200.50", "900", "5", "abc"]})
    df, log = app.preprocess_data(messy, "d", "s", dayfirst=True)
    assert list(df["s"]) == [1200.5, 900.0]
    assert df["d"].iloc[0] == pd.Timestamp(2024, 1, 31)  # day-first parsing worked
    assert log["invalid_dates_removed"] == 2


# -------------------------------------------------------------- ML pipeline
def test_aggregation_keeps_total(clean, daily):
    assert np.isclose(daily["sales"].sum(), clean[0]["Sales"].sum())


def test_lag_features_only_use_the_past(daily):
    frame = app.build_training_frame(daily, "Daily")
    day = frame.index[100]
    pos = daily.index[daily["date"] == day][0]
    assert frame.loc[day, "lag_1"] == daily["sales"][pos - 1]
    assert frame.loc[day, "lag_7"] == daily["sales"][pos - 7]
    assert np.isclose(frame.loc[day, "roll_mean_7"], daily["sales"][pos - 7:pos].mean())


@pytest.mark.parametrize("freq", ["Daily", "Monthly"])
def test_chronological_split_and_metrics(clean, freq):
    series = app.aggregate_sales(clean[0], "Date", "Sales", freq)
    ev = app.train_and_evaluate(series, freq, 0.2)
    assert set(ev["results"]) == {"Linear Regression", "Random Forest", "Decision Tree"}
    for res in ev["results"].values():
        assert set(res["metrics"]) == {"MAE", "MSE", "RMSE", "R²"}
        assert np.isclose(res["metrics"]["RMSE"] ** 2, res["metrics"]["MSE"])
        assert res["predictions"]["date"].min() >= ev["split_date"]  # test data is the latest data
    assert ev["train_rows"] > ev["test_rows"]


def test_too_little_history_is_rejected(daily):
    with pytest.raises(ValueError):
        app.train_and_evaluate(daily.head(20), "Daily")


@pytest.mark.parametrize("model", ["Linear Regression", "Random Forest", "Decision Tree"])
def test_future_forecast(daily, model):
    forecast = app.forecast_future(model, daily, "Daily", 14)
    assert len(forecast) == 14
    assert forecast["date"].iloc[0] == daily["date"].iloc[-1] + pd.Timedelta(days=1)
    assert (forecast["predicted_sales"] >= 0).all()


def test_monthly_forecast_dates(clean):
    monthly = app.aggregate_sales(clean[0], "Date", "Sales", "Monthly")
    forecast = app.forecast_future("Random Forest", monthly, "Monthly", 6)
    assert (forecast["date"].dt.day == 1).all() and len(forecast) == 6


def test_report_and_csv_export(clean, daily):
    df, log = clean
    cfg = {"date": "Date", "sales": "Sales", "product": "Product", "category": "Category", "region": "Region"}
    report = app.generate_report(df, cfg, log)
    assert "Sales Analysis Report" in report and "Total sales" in report
    csv = app.forecast_future("Linear Regression", daily, "Daily", 5).to_csv(index=False)
    assert csv.splitlines()[0] == "date,predicted_sales" and len(csv.splitlines()) == 6
