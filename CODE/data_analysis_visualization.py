"""Summer Training Project: Data Analysis and Visualization Using Python.

This program demonstrates a complete data-analysis workflow:
1. Generate a simulated retail-sales dataset.
2. Load and inspect the raw CSV file.
3. Clean duplicates, text, missing values and quantity outliers.
4. Create useful analysis features such as revenue and age groups.
5. Calculate KPIs, grouped summaries, correlations and statistical tests.
6. Build a simple linear trend and 3-month forecast.
7. Save charts, cleaned data and metrics.

Dataset: simulated 2024 retail sales data.
Random seed: 42 (for reproducible results).
"""

# -----------------------------
# 1. IMPORT LIBRARIES
# -----------------------------
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import f_oneway, linregress, pearsonr, ttest_ind

warnings.filterwarnings("ignore")

# -----------------------------
# 2. PROJECT PATHS
# -----------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
CHART_DIR = OUTPUT_DIR / "charts"
RAW_CSV = DATA_DIR / "retail_sales_2024_raw.csv"
CLEAN_CSV = DATA_DIR / "retail_sales_2024_cleaned.csv"
METRICS_JSON = OUTPUT_DIR / "metrics.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# 3. DATA GENERATION
# -----------------------------
def generate_dataset(seed=42, n_orders=5000):
    """Generate approximately 5,000 simulated retail orders.

    Deliberate data-quality problems are added so that the cleaning stage
    has realistic work to perform.
    """
    rng = np.random.default_rng(seed)

    # Dates and categorical values.
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    regions = np.array(["North", "South", "East", "West"])
    categories = np.array(["Electronics", "Home", "Fashion", "Grocery"])

    products = {
        "Electronics": ["Laptop", "Headphones", "Smartphone", "Monitor"],
        "Home": ["Mixer", "Chair", "Lamp", "Cookware"],
        "Fashion": ["Shirt", "Jeans", "Shoes", "Jacket"],
        "Grocery": ["Rice", "Oil", "Snacks", "Beverages"],
    }

    payment_modes = np.array(["UPI", "Card", "Cash", "Net Banking"])

    # Generate the main fields.
    category = rng.choice(categories, n_orders, p=[0.28, 0.22, 0.28, 0.22])
    product = [rng.choice(products[c]) for c in category]

    base_price = {
        "Electronics": 42000,
        "Home": 2500,
        "Fashion": 1800,
        "Grocery": 650,
    }

    unit_price = np.array(
        [max(80, rng.normal(base_price[c], base_price[c] * 0.25)) for c in category]
    ).round(2)

    quantity = rng.integers(1, 6, n_orders)
    discount = np.clip(rng.beta(2, 10, n_orders), 0, 0.35).round(3)
    customer_age = np.clip(rng.normal(34, 10, n_orders).round(), 18, 70).astype(float)
    delivery_days = np.clip(rng.normal(3.5, 1.6, n_orders).round(), 1, 10).astype(float)
    rating = np.clip(rng.normal(4.0, 0.65, n_orders), 1, 5).round(1)
    region = rng.choice(regions, n_orders)
    payment_mode = rng.choice(payment_modes, n_orders)

    df = pd.DataFrame(
        {
            "order_id": [f"ORD{100000 + i}" for i in range(n_orders)],
            "order_date": rng.choice(dates, n_orders),
            "customer_id": [f"CUST{1000 + rng.integers(0, 1800)}" for _ in range(n_orders)],
            "customer_age": customer_age,
            "region": region,
            "category": category,
            "product": product,
            "quantity": quantity,
            "unit_price": unit_price,
            "discount": discount,
            "payment_mode": payment_mode,
            "delivery_days": delivery_days,
            "rating": rating,
        }
    )

    # Add duplicate rows.
    duplicate_rows = df.sample(60, random_state=seed).copy()
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    # Add missing values.
    missing_rating = rng.choice(df.index, 85, replace=False)
    missing_age = rng.choice(df.index.difference(missing_rating), 70, replace=False)
    missing_delivery = rng.choice(
        df.index.difference(missing_rating).difference(missing_age),
        65,
        replace=False,
    )

    df.loc[missing_rating, "rating"] = np.nan
    df.loc[missing_age, "customer_age"] = np.nan
    df.loc[missing_delivery, "delivery_days"] = np.nan

    # Add extreme quantity values.
    outlier_idx = rng.choice(df.index, 25, replace=False)
    df.loc[outlier_idx, "quantity"] = rng.integers(15, 30, len(outlier_idx))

    # Add text inconsistencies.
    lower_idx = rng.choice(df.index, 120, replace=False)
    df.loc[lower_idx, "region"] = df.loc[lower_idx, "region"].str.lower()

    pay_idx = rng.choice(df.index, 140, replace=False)
    df.loc[pay_idx, "payment_mode"] = df.loc[pay_idx, "payment_mode"] + "  "

    # Save the raw dataset.
    df.to_csv(RAW_CSV, index=False)
    return df


# -----------------------------
# 4. LOAD DATA
# -----------------------------
def load_data(path=RAW_CSV):
    """Load the raw CSV file and convert order_date to datetime."""
    return pd.read_csv(path, parse_dates=["order_date"])


# -----------------------------
# 5. INSPECT DATA
# -----------------------------
def inspect_data(df):
    """Return basic data-quality information."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_by_column": {k: int(v) for k, v in df.isna().sum().items()},
        "missing_cells": int(df.isna().sum().sum()),
    }


# -----------------------------
# 6. CLEAN DATA
# -----------------------------
def clean_data(df):
    """Clean text, remove duplicates, fill missing values and cap outliers."""
    out = df.copy()

    # Remove exact duplicate rows.
    before = len(out)
    out = out.drop_duplicates().copy()
    duplicates_removed = before - len(out)

    # Remove extra spaces and standardize text.
    for col in ["region", "category", "product", "payment_mode"]:
        out[col] = out[col].astype(str).str.strip()

    out["region"] = out["region"].str.title()
    out["payment_mode"] = out["payment_mode"].str.title()

    # Fill missing numeric values with the median.
    for col in ["customer_age", "delivery_days", "rating"]:
        out[col] = out[col].fillna(out[col].median())

    # Detect and cap quantity outliers using the IQR rule.
    q1, q3 = out["quantity"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outlier_mask = (out["quantity"] < lower) | (out["quantity"] > upper)
    out["quantity"] = out["quantity"].clip(lower, upper)

    quality = {
        "duplicates_removed": int(duplicates_removed),
        "remaining_rows": int(len(out)),
        "missing_cells_after": int(out.isna().sum().sum()),
        "quantity_iqr_lower": float(lower),
        "quantity_iqr_upper": float(upper),
        "quantity_outliers_capped": int(outlier_mask.sum()),
    }

    return out, quality


# -----------------------------
# 7. FEATURE ENGINEERING
# -----------------------------
def add_features(df):
    """Create financial, date, weekend and customer-age features."""
    out = df.copy()

    # Gross amount before discount.
    out["gross_amount"] = out["quantity"] * out["unit_price"]

    # Revenue after discount.
    out["revenue"] = out["gross_amount"] * (1 - out["discount"])

    # Date-based features.
    out["month"] = out["order_date"].dt.strftime("%b")
    out["month_num"] = out["order_date"].dt.month
    out["weekday"] = out["order_date"].dt.day_name()
    out["is_weekend"] = out["order_date"].dt.dayofweek >= 5

    # Business-friendly age groups.
    out["age_group"] = pd.cut(
        out["customer_age"],
        bins=[0, 24, 34, 44, 54, 100],
        labels=["18-24", "25-34", "35-44", "45-54", "55+"],
    )

    return out


# -----------------------------
# 8. ANALYSIS
# -----------------------------
def analyse(df, inspection, quality):
    """Calculate KPIs, summaries, correlations, tests and forecast."""
    month_order = list(
        pd.date_range("2024-01-01", "2024-12-01", freq="MS").strftime("%b")
    )

    # Monthly revenue.
    monthly = (
        df.groupby("month_num")["revenue"]
        .agg(["sum", "mean", "count"])
        .reset_index()
    )
    monthly["month"] = pd.to_datetime(monthly["month_num"], format="%m").dt.strftime("%b")
    monthly = monthly.sort_values("month_num")

    # Revenue by business dimensions.
    category_revenue = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
    region_revenue = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
    age_revenue = df.groupby("age_group", observed=False)["revenue"].sum()
    payment_revenue = df.groupby("payment_mode")["revenue"].sum().sort_values(ascending=False)

    # Rating distribution and correlations.
    rating_counts = df["rating"].round(1).value_counts().sort_index()
    correlation = df[
        ["quantity", "unit_price", "discount", "delivery_days", "rating", "revenue"]
    ].corr()

    # Welch t-test: weekend vs weekday revenue.
    weekend = df.loc[df["is_weekend"], "revenue"]
    weekday = df.loc[~df["is_weekend"], "revenue"]
    t_stat, t_p = ttest_ind(weekend, weekday, equal_var=False)

    # One-way ANOVA: ratings across regions.
    groups = [group["rating"].values for _, group in df.groupby("region")]
    anova_stat, anova_p = f_oneway(*groups)

    # Pearson correlation: delivery days vs rating.
    pearson_r, pearson_p = pearsonr(df["delivery_days"], df["rating"])

    # Simple linear regression on monthly revenue.
    x = monthly["month_num"].to_numpy(dtype=float)
    y = monthly["sum"].to_numpy(dtype=float)
    regression = linregress(x, y)

    # Forecast the next three months.
    forecast_x = np.arange(13, 16)
    forecast_y = regression.intercept + regression.slope * forecast_x

    monthly_table = monthly.set_index("month")["sum"].to_dict()

    metrics = {
        "dataset": {
            "seed": 42,
            "raw_rows": inspection["rows"],
            "raw_columns": inspection["columns"],
            "clean_rows": int(len(df)),
            "year": 2024,
            "simulated": True,
        },
        "quality": {
            **quality,
            "raw_missing_cells": inspection["missing_cells"],
            "raw_duplicate_rows": inspection["duplicate_rows"],
            "missing_by_column": inspection["missing_by_column"],
        },
        "kpis": {
            "total_revenue": float(df["revenue"].sum()),
            "gross_sales": float(df["gross_amount"].sum()),
            "total_orders": int(len(df)),
            "unique_customers": int(df["customer_id"].nunique()),
            "average_order_value": float(df["revenue"].mean()),
            "average_rating": float(df["rating"].mean()),
            "average_delivery_days": float(df["delivery_days"].mean()),
            "average_discount_pct": float(df["discount"].mean() * 100),
            "total_quantity": float(df["quantity"].sum()),
        },
        "monthly_revenue": {k: float(monthly_table[k]) for k in month_order},
        "category_revenue": {str(k): float(v) for k, v in category_revenue.items()},
        "region_revenue": {str(k): float(v) for k, v in region_revenue.items()},
        "age_group_revenue": {str(k): float(v) for k, v in age_revenue.items()},
        "payment_mode_revenue": {str(k): float(v) for k, v in payment_revenue.items()},
        "rating_distribution": {str(k): int(v) for k, v in rating_counts.items()},
        "correlation_matrix": correlation.round(4).to_dict(),
        "tests": {
            "welch_t_test": {
                "t_stat": float(t_stat),
                "p_value": float(t_p),
                "alpha": 0.05,
                "significant": bool(t_p < 0.05),
            },
            "anova": {
                "f_stat": float(anova_stat),
                "p_value": float(anova_p),
                "alpha": 0.05,
                "significant": bool(anova_p < 0.05),
            },
            "pearson": {
                "r": float(pearson_r),
                "p_value": float(pearson_p),
                "alpha": 0.05,
                "significant": bool(pearson_p < 0.05),
            },
        },
        "regression": {
            "slope": float(regression.slope),
            "intercept": float(regression.intercept),
            "r_squared": float(regression.rvalue**2),
            "p_value": float(regression.pvalue),
            "forecast_months": ["Jan", "Feb", "Mar"],
            "forecast_revenue": [float(v) for v in forecast_y],
        },
        "descriptive": {
            "revenue": {k: float(v) for k, v in df["revenue"].describe().items()},
            "quantity": {k: float(v) for k, v in df["quantity"].describe().items()},
            "rating": {k: float(v) for k, v in df["rating"].describe().items()},
        },
    }

    return metrics


# -----------------------------
# 9. CHART CREATION
# -----------------------------
def make_charts(df, metrics):
    """Create the 13 charts used by the project."""
    sns.set_theme(style="whitegrid")
    chart_paths = []

    def save_chart(fig, filename):
        """Save one chart and close its figure."""
        path = CHART_DIR / filename
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        chart_paths.append(str(path))

    # 1. Missing values.
    missing = pd.Series(metrics["quality"]["missing_by_column"]).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    missing[missing > 0].plot(kind="bar", ax=ax)
    ax.set_title("Missing Values by Column")
    ax.set_xlabel("Column")
    ax.set_ylabel("Missing cells")
    save_chart(fig, "01_missing_values.png")

    # 2. Monthly revenue.
    monthly = pd.Series(metrics["monthly_revenue"])
    fig, ax = plt.subplots(figsize=(9, 5))
    monthly.plot(marker="o", ax=ax)
    ax.set_title("Monthly Revenue - 2024")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue")
    save_chart(fig, "02_monthly_revenue.png")

    # 3. Category revenue.
    category = pd.Series(metrics["category_revenue"]).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    category.plot(kind="barh", ax=ax)
    ax.set_title("Revenue by Category")
    ax.set_xlabel("Revenue")
    ax.set_ylabel("Category")
    save_chart(fig, "03_category_revenue.png")

    # 4. Region revenue.
    region = pd.Series(metrics["region_revenue"]).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    region.plot(kind="bar", ax=ax)
    ax.set_title("Revenue by Region")
    ax.set_xlabel("Region")
    ax.set_ylabel("Revenue")
    save_chart(fig, "04_region_revenue.png")

    # 5. Correlation heatmap.
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        df[["quantity", "unit_price", "discount", "delivery_days", "rating", "revenue"]].corr(),
        annot=True,
        fmt=".2f",
        ax=ax,
    )
    ax.set_title("Correlation Matrix")
    save_chart(fig, "05_correlation_heatmap.png")

    # 6. Revenue distribution.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["revenue"], bins=35)
    ax.set_title("Order Value Distribution")
    ax.set_xlabel("Order revenue")
    ax.set_ylabel("Orders")
    save_chart(fig, "06_order_value_histogram.png")

    # 7. Box plot by category.
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x="category", y="revenue", ax=ax)
    ax.set_title("Order Revenue by Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Revenue")
    ax.tick_params(axis="x", rotation=20)
    save_chart(fig, "07_boxplot_by_category.png")

    # 8. Discount vs quantity.
    sample = df.sample(min(1500, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(data=sample, x="discount", y="quantity", alpha=0.5, ax=ax)
    ax.set_title("Discount vs Quantity")
    ax.set_xlabel("Discount")
    ax.set_ylabel("Quantity")
    save_chart(fig, "08_discount_vs_quantity.png")

    # 9. Rating distribution.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["rating"], bins=np.arange(0.75, 5.26, 0.25))
    ax.set_title("Rating Distribution")
    ax.set_xlabel("Rating (1-5)")
    ax.set_ylabel("Orders")
    save_chart(fig, "09_rating_distribution.png")

    # 10. Payment-mode revenue share.
    payment = pd.Series(metrics["payment_mode_revenue"])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(payment.values, labels=payment.index, autopct="%1.1f%%", startangle=90)
    ax.set_title("Revenue Share by Payment Mode")
    save_chart(fig, "10_payment_mode_pie.png")

    # 11. Region x month heatmap.
    pivot = df.pivot_table(
        index="region", columns="month_num", values="revenue", aggfunc="sum"
    )
    pivot.columns = pd.to_datetime(pivot.columns, format="%m").strftime("%b")
    month_labels = pd.date_range("2024-01-01", "2024-12-01", freq="MS").strftime("%b")
    pivot = pivot.reindex(columns=month_labels)

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.heatmap(pivot, ax=ax)
    ax.set_title("Region x Month Revenue Heatmap")
    ax.set_xlabel("Month")
    ax.set_ylabel("Region")
    save_chart(fig, "11_region_month_heatmap.png")

    # 12. Age-group revenue.
    age = pd.Series(metrics["age_group_revenue"])
    fig, ax = plt.subplots(figsize=(9, 5))
    age.plot(kind="bar", ax=ax)
    ax.set_title("Revenue by Age Group")
    ax.set_xlabel("Age group")
    ax.set_ylabel("Revenue")
    save_chart(fig, "12_age_group_revenue.png")

    # 13. Trend and forecast.
    x = np.arange(1, 13)
    y = np.array(list(metrics["monthly_revenue"].values()))
    fx = np.arange(1, 16)
    slope = metrics["regression"]["slope"]
    intercept = metrics["regression"]["intercept"]
    fy = intercept + slope * fx

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, marker="o", label="Actual 2024")
    ax.plot(fx, fy, linestyle="--", label="Linear trend + forecast")
    ax.axvline(12.5, linestyle=":")
    ax.set_title("Monthly Revenue Trend and 3-Month Forecast")
    ax.set_xlabel("Month number")
    ax.set_ylabel("Revenue")
    ax.legend()
    save_chart(fig, "13_trend_forecast.png")

    return chart_paths


# -----------------------------
# 10. MAIN PROGRAM
# -----------------------------
def main():
    """Run the complete project from start to finish."""

    print("Starting Data Analysis and Visualization project...\n")

    # Step 1: Generate the raw simulated dataset.
    generate_dataset(seed=42, n_orders=5000)

    # Step 2: Load and inspect raw data.
    raw = load_data()
    inspection = inspect_data(raw)

    # Step 3: Clean the data.
    clean, quality = clean_data(raw)

    # Step 4: Create analysis features.
    clean = add_features(clean)

    # Step 5: Save cleaned data.
    clean.to_csv(CLEAN_CSV, index=False)

    # Step 6: Calculate all analysis results.
    metrics = analyse(clean, inspection, quality)

    # Step 7: Create all charts.
    chart_paths = make_charts(clean, metrics)

    # Step 8: Add file information to metrics.
    metrics["files"] = {
        "raw_csv": str(RAW_CSV.relative_to(ROOT)),
        "cleaned_csv": str(CLEAN_CSV.relative_to(ROOT)),
        "charts_dir": str(CHART_DIR.relative_to(ROOT)),
    }
    metrics["chart_files"] = [str(Path(p).relative_to(ROOT)) for p in chart_paths]

    # Step 9: Save metrics as JSON.
    METRICS_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Step 10: Print a simple final summary.
    print(json.dumps(
        {
            "status": "success",
            "raw_rows": inspection["rows"],
            "clean_rows": len(clean),
            "raw_columns": inspection["columns"],
            "charts_created": len(chart_paths),
            "cleaned_csv": str(CLEAN_CSV),
            "metrics_file": str(METRICS_JSON),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
