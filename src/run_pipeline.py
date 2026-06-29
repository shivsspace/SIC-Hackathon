import json
import logging
import pandas as pd
from src.pipeline.config import (
    DASHBOARD_JSON_FILENAME,
    HOUR_BINS,
    HOUR_LABELS,
)
from src.pipeline.etl import generate_raw_data, ETLPipeline
from src.pipeline.heap import build_inventory_priority_queue
from src.pipeline.visualization import build_dashboard
from src.pipeline.mapping import build_regional_map

# Setup logger for CLI run
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def export_dashboard_json(clean_df: pd.DataFrame, priority_list: list[tuple[float, str]]) -> None:
    """
    Serialise all dashboard data to dashboard_data.json and dashboard_data.js.
    """
    log.info("--- Exporting Dashboard JSON ---")

    # Per-product summary
    product_summary = (
        clean_df.groupby("product")
        .agg(
            total_revenue=("revenue", "sum"),
            total_qty=("qty", "sum"),
            avg_unit_price=("unit_price", "mean"),
            fraud_count=("is_fraud", "sum"),
            order_count=("order_id", "count"),
        )
        .reset_index()
        .round(2)
    )

    # Per-region summary
    region_summary = (
        clean_df.groupby("region")
        .agg(
            total_revenue=("revenue", "sum"),
            order_count=("order_id", "count"),
            fraud_count=("is_fraud", "sum"),
        )
        .reset_index()
        .round(2)
    )

    # Monthly revenue pivot
    monthly = (
        clean_df.groupby(["order_month", "product"])["revenue"]
        .sum()
        .reset_index()
        .round(2)
    )
    monthly_list = monthly.to_dict(orient="records")

    # Daily revenue trend
    clean_df["order_day"] = clean_df["order_date"].dt.day
    daily = (
        clean_df.groupby(["order_day", "product"])["revenue"]
        .sum()
        .reset_index()
        .round(2)
    )
    daily_list = daily.to_dict(orient="records")

    # Hourly heatmap (product x hour_bin)
    clean_df["hour_bin"] = pd.cut(
        clean_df["order_hour"], bins=HOUR_BINS, labels=HOUR_LABELS, right=False
    ).astype(str)
    heatmap = (
        clean_df.groupby(["product", "hour_bin"])["revenue"]
        .mean()
        .reset_index()
        .round(2)
    )

    # Fraud orders
    fraud_orders = (
        clean_df[clean_df["is_fraud"]]
        [["order_id", "product", "region", "revenue", "order_date"]]
        .copy()
    )
    fraud_orders["order_date"] = fraud_orders["order_date"].dt.strftime("%Y-%m-%d %H:%M")
    fraud_orders = fraud_orders.round(2)

    # All cleaned orders (for the orders table)
    orders_export = clean_df[
        ["order_id", "product", "category", "region", "qty",
         "unit_price", "discount", "revenue", "is_fraud", "order_month", "order_hour"]
    ].copy()
    orders_export["is_fraud"] = orders_export["is_fraud"].astype(bool)
    orders_export = orders_export.round(2)

    # AI Insight Generator
    insights = []
    recommendations = []

    total_rev = float(clean_df["revenue"].sum())

    # Top product
    top_p = clean_df.groupby("product")["revenue"].sum().idxmax()
    insights.append(f"{top_p} generated the highest revenue.")

    # Highest Revenue Region
    top_reg = clean_df.groupby("region")["revenue"].sum().idxmax()
    top_reg_rev = clean_df.groupby("region")["revenue"].sum().max()
    top_reg_pct = (top_reg_rev / total_rev) * 100 if total_rev > 0 else 0
    insights.append(f"{top_reg} region contributed {top_reg_pct:.0f}% of total sales.")

    # Fraud Analysis
    fraud_df = clean_df[clean_df["is_fraud"]]
    if len(fraud_df) > 0:
        worst_reg = fraud_df.groupby("region").size().idxmax()
        insights.append(f"Most fraud occurred in {worst_reg} region.")
        recommendations.append(f"Investigate fraudulent transactions in {worst_reg}.")
    else:
        insights.append("No fraudulent transactions were detected.")
        recommendations.append("Continue regular fraud audit schedules.")

    # Peak Shopping Hour
    peak_h = clean_df.groupby("order_hour")["revenue"].sum().idxmax()
    insights.append(f"Peak shopping hour was {peak_h:02d}:00.")

    # Best Month
    best_m = clean_df.groupby("order_month")["revenue"].sum().idxmax()
    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December"
    }
    best_m_name = month_names.get(best_m, f"Month {best_m}")
    insights.append(f"{best_m_name} recorded the highest monthly revenue.")

    # Recommendations
    sales_series = clean_df.groupby("product")["revenue"].sum()
    worst_p = sales_series.idxmin()
    recommendations.append(f"Boost marketing for {worst_p}.")

    best_p = sales_series.idxmax()
    recommendations.append(f"Increase inventory for {best_p}.")

    disc_series = clean_df.groupby("product")["discount"].mean()
    high_disc_p = disc_series.idxmax()
    p_plural = high_disc_p
    if not p_plural.endswith('s'):
        p_plural = p_plural + "s"
    recommendations.append(f"Reduce discounts on {p_plural}.")

    rev_series = clean_df.groupby("region")["revenue"].sum()
    low_reg = rev_series.idxmin()
    recommendations.append(f"Run promotions in {low_reg}.")

    payload = {
        "meta": {
            "total_orders": int(len(clean_df)),
            "total_revenue": round(float(clean_df["revenue"].sum()), 2),
            "total_fraud": int(clean_df["is_fraud"].sum()),
            "products": sorted(clean_df["product"].unique().tolist()),
            "regions": sorted(clean_df["region"].unique().tolist()),
        },
        "product_summary": product_summary.to_dict(orient="records"),
        "region_summary": region_summary.to_dict(orient="records"),
        "monthly_revenue": monthly_list,
        "daily_revenue": daily_list,
        "heatmap": heatmap.to_dict(orient="records"),
        "fraud_orders": fraud_orders.to_dict(orient="records"),
        "inventory_priority": [
            {"rank": i + 1, "product": p, "restock_score": round(s, 6)}
            for i, (s, p) in enumerate(priority_list)
        ],
        "ai_insights": {
            "insights": insights,
            "recommendations": recommendations
        },
        "orders": orders_export.to_dict(orient="records"),
    }

    with open(DASHBOARD_JSON_FILENAME, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # Export dashboard_data.js as well for offline file:// loading support
    dashboard_js_filename = DASHBOARD_JSON_FILENAME.replace(".json", ".js")
    with open(dashboard_js_filename, "w", encoding="utf-8") as f:
        f.write("window.DASHBOARD_DATA = ")
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write(";\n")

    log.info("Dashboard JSON exported -> %s and %s", DASHBOARD_JSON_FILENAME, dashboard_js_filename)


def main() -> None:
    """Orchestrate all tasks end-to-end."""
    log.info("============================================")
    log.info("  E-Commerce Sales Intelligence - Pipeline  ")
    log.info("============================================")

    # Task 1: ETL
    raw_df = generate_raw_data()
    pipeline = ETLPipeline(raw_df)
    clean_df, product_summary = pipeline.run()

    # Task 2: Min-Heap Priority Queue
    priority_list = build_inventory_priority_queue(clean_df)

    # Task 3: Visualizations
    build_dashboard(clean_df)

    # Task 4: Folium Mapping
    build_regional_map(clean_df)

    # Web Dashboard JSON
    export_dashboard_json(clean_df, priority_list)

    log.info("============================================")
    log.info("  PIPELINE EXECUTION COMPLETE")
    log.info("============================================")


if __name__ == "__main__":
    main()
