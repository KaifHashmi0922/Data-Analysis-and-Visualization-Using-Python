"""Basic tests for the Summer Training data-analysis project."""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import CODE.data_analysis_visualization as project


def test_generation():
    df = project.generate_dataset(seed=42, n_orders=5000)
    assert len(df) == 5060
    assert len(df.columns) == 13
    assert df.duplicated().sum() == 52
    assert int(df.isna().sum().sum()) == 220


def test_cleaning():
    raw = project.load_data()
    clean, quality = project.clean_data(raw)
    assert len(clean) == 5008
    assert clean.isna().sum().sum() == 0
    assert quality["duplicates_removed"] == 52
    assert quality["quantity_outliers_capped"] == 25


def test_features_and_analysis():
    raw = project.load_data()
    inspection = project.inspect_data(raw)
    clean, quality = project.clean_data(raw)
    clean = project.add_features(clean)
    metrics = project.analyse(clean, inspection, quality)

    required = {"gross_amount", "revenue", "month", "month_num", "weekday", "is_weekend", "age_group"}
    assert required.issubset(clean.columns)
    assert metrics["kpis"]["total_orders"] == 5008
    assert len(metrics["monthly_revenue"]) == 12
    assert len(metrics["regression"]["forecast_revenue"]) == 3


def test_charts():
    raw = project.load_data()
    inspection = project.inspect_data(raw)
    clean, quality = project.clean_data(raw)
    clean = project.add_features(clean)
    metrics = project.analyse(clean, inspection, quality)
    paths = project.make_charts(clean, metrics)

    assert len(paths) == 13
    assert all(Path(path).exists() for path in paths)


if __name__ == "__main__":
    # Allows: python tests/test_project.py
    test_generation()
    test_cleaning()
    test_features_and_analysis()
    test_charts()
    print("All tests passed.")
