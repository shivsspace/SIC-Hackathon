"""
E-Commerce Sales Intelligence — Hackathon Set B
================================================
Production-grade solution covering all 4 tasks:
  Task 1: ETL Pipeline (Pandas)
  Task 2: DSA — Min-Heap Priority Queue (from scratch)
  Task 3: Revenue Visualization Dashboard (Matplotlib)
  Task 4: Regional Revenue Map (Folium)

Run: python solution.py
Outputs:
  - cleaned_orders.csv
  - product_summary.csv
  - sales_dashboard.png
  - regional_sales_map.html
"""

from __future__ import annotations
import json
from folium.plugins import MarkerCluster, MiniMap
import folium
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings("ignore")

# LOGGING

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# CONSTANTS
RANDOM_SEED: int = 7
N_RECORDS: int = 600
FRAUD_STD_MULTIPLIER: float = 3.0
DASHBOARD_DPI: int = 150
DASHBOARD_FILENAME: str = "sales_dashboard.png"
PRODUCT_SUMMARY_FILENAME: str = "product_summary.csv"
MAP_FILENAME: str = "regional_sales_map.html"
DASHBOARD_JSON_FILENAME: str = "dashboard_data.json"
HOUR_BINS: list[int] = [0, 9, 17, 24]
HOUR_LABELS: list[str] = ["0\u20138", "9\u201316", "17\u201323"]
MARKER_RADIUS_DIVISOR: float = 500_000.0
MAP_CENTER: list[float] = [20.5937, 78.9629]
MAP_ZOOM: int = 5
JITTER_RANGE: float = 1.0

REGION_COORDS: dict[str, list[float]] = {
    "North": [28.6139, 77.2090],
    "South": [13.0827, 80.2707],
    "East": [22.5726, 88.3639],
    "West": [19.0760, 72.8777],
    "Other": [20.5937, 78.9629],
}

PRODUCT_COLORS: dict[str, str] = {
    "Laptop":  "#2196F3",
    "Phone":   "#4CAF50",
    "Tablet":  "#FF9800",
    "Watch":   "#9C27B0",
    "Earbuds": "#F44336",
}


# TASK 1 — ETL PIPELINE

def generate_raw_data() -> pd.DataFrame:
    """
    Reproduce the exact noisy dataset from the hackathon starter code.
    Returns the raw DataFrame with all injected data quality issues intact.
    """
    log.info("Generating raw dataset with injected noise ...")
    np.random.seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    unit_price: np.ndarray = (
        np.random.uniform(500, 80_000, N_RECORDS).round(2).astype(object)
    )
    qty: np.ndarray = np.random.randint(1, 10, N_RECORDS).astype(object)
    product: list[str] = list(
        np.random.choice(["Laptop", "Phone", "Tablet",
                         "Watch", "Earbuds"], N_RECORDS)
    )
    region: list[str] = list(
        np.random.choice(["North", "South", "East", "West"], N_RECORDS)
    )

    #  Noise injection (must NOT be modified)
    idx = rng.choice(N_RECORDS, size=60, replace=False)
    for i in idx[:12]:
        unit_price[i] = np.nan
    for i in idx[12:22]:
        unit_price[i] = f"\u20b9{float(unit_price[i]):.2f}"   # ₹ prefix
    for i in idx[22:32]:
        qty[i] = 0                                              # zero qty
    for i in idx[32:42]:
        product[i] = product[i].upper()                        # ALL CAPS

    disc: np.ndarray = np.random.choice(
        [0, 5, 10, 15, 20, np.nan], N_RECORDS
    ).astype(object)
    for i in idx[42:50]:
        # impossible discount
        disc[i] = 150
    for i in idx[50:55]:
        region[i] = "UNKNOWN"                                   # bad region

    df = pd.DataFrame(
        {
            "order_id": [f"ORD{i:05d}" for i in range(N_RECORDS)],
            "product": product,
            "category": np.random.choice(["Electronics", "Accessories"], N_RECORDS),
            "qty": qty,
            "unit_price": unit_price,
            "discount": disc,
            "region": region,
            "order_date": pd.date_range("2023-01-01", periods=N_RECORDS, freq="h"),
        }
    )

    dup_idx = rng.choice(N_RECORDS, 10)
    df = pd.concat([df, df.iloc[dup_idx]],
                   ignore_index=True)  # 10 duplicate rows
    log.info("Raw dataset shape: %s", df.shape)
    return df


class ETLPipeline:
    """
    Production ETL pipeline for e-commerce order data.

    Responsibilities
    ----------------
    1. Duplicate removal
    2. Invalid quantity removal
    3. Currency string cleaning + dtype coercion
    4. NaN imputation via product-wise median
    5. Discount cleaning
    6. Product / Region normalisation
    7. Derived feature engineering
    8. Fraud detection
    9. Export
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df: pd.DataFrame = df.copy()
        self._validate_required_columns()

    #  Validation

    def _validate_required_columns(self) -> None:
        required = {
            "order_id", "product", "category", "qty",
            "unit_price", "discount", "region", "order_date",
        }
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    #  Subtask 6: Drop duplicates & zero-qty rows

    def drop_duplicates(self) -> "ETLPipeline":
        """Remove fully-duplicate rows."""
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        removed = before - len(self.df)
        log.info("Duplicates removed: %d (rows remaining: %d)",
                 removed, len(self.df))
        assert removed >= 10, f"Expected ≥10 duplicates, found {removed}"
        return self

    def remove_zero_qty(self) -> "ETLPipeline":
        """Remove orders with qty = 0 (invalid)."""
        self.df["qty"] = pd.to_numeric(self.df["qty"], errors="coerce")
        before = len(self.df)
        self.df = self.df[self.df["qty"] > 0].copy()
        self.df.reset_index(drop=True, inplace=True)
        removed = before - len(self.df)
        log.info("Zero-qty rows removed: %d", removed)
        assert removed >= 10, f"Expected ≥10 zero-qty rows, found {removed}"
        return self

    #  Subtask 7: Clean unit_price

    def clean_unit_price(self) -> "ETLPipeline":
        """Strip ₹ prefix, coerce to float, impute NaN with product-wise median."""
        self.df["unit_price"] = (
            self.df["unit_price"]
            .astype(str)
            .str.replace("₹", "", regex=False)
            .str.strip()
        )
        self.df["unit_price"] = pd.to_numeric(
            self.df["unit_price"], errors="coerce")

        nan_count = self.df["unit_price"].isna().sum()
        log.info("unit_price NaNs before imputation: %d", nan_count)

        # Product-wise median imputation
        product_median = (
            self.df.groupby("product")["unit_price"]
            .transform("median")
        )
        self.df["unit_price"] = self.df["unit_price"].fillna(product_median)

        # Fallback: global median for any remaining NaN
        global_median = self.df["unit_price"].median()
        self.df["unit_price"] = self.df["unit_price"].fillna(global_median)

        assert self.df["unit_price"].isna().sum(
        ) == 0, "NaNs remain in unit_price"
        assert (self.df["unit_price"] > 0).all(
        ), "Non-positive unit_price detected"
        log.info("unit_price cleaned *")
        return self

    #  Subtask 8: Clean discount

    def clean_discount(self) -> "ETLPipeline":
        """Replace impossible discount values (>100) with NaN, then product-median impute."""
        self.df["discount"] = pd.to_numeric(
            self.df["discount"], errors="coerce")
        invalid_count = (self.df["discount"] > 100).sum()
        log.info("Impossible discount values (>100): %d", invalid_count)

        self.df.loc[self.df["discount"] > 100, "discount"] = np.nan

        product_median = (
            self.df.groupby("product")["discount"]
            .transform("median")
        )
        self.df["discount"] = self.df["discount"].fillna(product_median)
        self.df["discount"] = self.df["discount"].fillna(0.0)  # fallback

        assert (self.df["discount"] <= 100).all(
        ), "Discount > 100 still present"
        assert self.df["discount"].isna().sum() == 0, "NaNs remain in discount"
        log.info("discount cleaned *")
        return self

    #  Subtask 9: Normalise product & region

    def normalise_product(self) -> "ETLPipeline":
        """Convert ALL-CAPS product names to Title Case."""
        before_unique = set(self.df["product"].unique())
        self.df["product"] = self.df["product"].str.title()
        after_unique = set(self.df["product"].unique())
        log.info(
            "Product normalisation: %d -> %d unique values",
            len(before_unique), len(after_unique),
        )
        return self

    def normalise_region(self) -> "ETLPipeline":
        """Replace 'UNKNOWN' region with 'Other'."""
        unknown_count = (self.df["region"] == "UNKNOWN").sum()
        log.info("UNKNOWN region rows: %d", unknown_count)
        self.df["region"] = self.df["region"].replace("UNKNOWN", "Other")
        assert "UNKNOWN" not in self.df["region"].values, "UNKNOWN still present in region"
        log.info("region normalised *")
        return self

    #  Subtask 10: Derived features, fraud detection, export

    def compute_derived_features(self) -> "ETLPipeline":
        """
        Compute:
          - revenue = qty × unit_price × (1 - discount / 100)
          - is_fraud: revenue > mean + 3σ
          - order_month: integer month
          - order_hour: hour-of-day
        """
        self.df["revenue"] = (
            self.df["qty"].astype(float)
            * self.df["unit_price"]
            * (1.0 - self.df["discount"] / 100.0)
        ).round(2)

        rev_mean = self.df["revenue"].mean()
        rev_std = self.df["revenue"].std()
        fraud_threshold = rev_mean + FRAUD_STD_MULTIPLIER * rev_std
        self.df["is_fraud"] = self.df["revenue"] > fraud_threshold

        self.df["order_month"] = self.df["order_date"].dt.month
        self.df["order_hour"] = self.df["order_date"].dt.hour

        fraud_count = self.df["is_fraud"].sum()
        log.info(
            "Revenue stats - mean: INR %.2f  std: INR %.2f  fraud threshold: INR %.2f",
            rev_mean, rev_std, fraud_threshold,
        )
        log.info("Fraud orders detected: %d", fraud_count)
        return self

    def export_product_summary(self) -> pd.DataFrame:
        """
        Aggregate per-product summary and export to product_summary.csv.

        Columns: product, total_revenue, total_qty, avg_unit_price,
                 fraud_count, order_count
        """
        summary = (
            self.df.groupby("product")
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
        summary.to_csv(PRODUCT_SUMMARY_FILENAME, index=False)
        log.info("Exported -> %s", PRODUCT_SUMMARY_FILENAME)
        return summary

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Execute the full ETL pipeline and return (cleaned_df, product_summary)."""
        log.info("--- Starting ETL Pipeline ---")
        (
            self
            .drop_duplicates()
            .remove_zero_qty()
            .normalise_product()
            .normalise_region()
            .clean_unit_price()
            .clean_discount()
            .compute_derived_features()
        )
        summary = self.export_product_summary()
        log.info("ETL complete - final shape: %s", self.df.shape)
        return self.df, summary

# TASK 2 — DSA: MIN-HEAP FROM SCRATCH


class MinHeap:
    """
    Binary Min-Heap implemented from scratch (no heapq).

    Each item is a tuple: (priority: float, label: str)
    The heap guarantees: parent.priority ≤ children.priority at all times.

    Complexity
    ----------
    insert       : O(log n)
    extract_min  : O(log n)
    peek         : O(1)
    size         : O(1)
    """

    def __init__(self) -> None:
        self._data: list[tuple[float, str]] = []

    #  Public API

    def insert(self, item: tuple[float, str]) -> None:
        """Insert (priority, label) and restore heap property."""
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(f"Expected (float, str) tuple, got {type(item)}")
        self._data.append(item)
        self._sift_up(len(self._data) - 1)
        self._assert_heap_property()   # Subtask 10: validate after every insert

    def extract_min(self) -> tuple[float, str]:
        """Remove and return the item with the smallest priority."""
        if self.size() == 0:
            raise IndexError("extract_min called on empty heap")
        # Swap root with last element, pop last, then sift root down
        self._data[0], self._data[-1] = self._data[-1], self._data[0]
        minimum = self._data.pop()
        if self._data:
            self._sift_down(0)
        return minimum

    def peek(self) -> tuple[float, str]:
        """Return (but don't remove) the minimum-priority item."""
        if self.size() == 0:
            raise IndexError("peek called on empty heap")
        return self._data[0]

    def size(self) -> int:
        """Return number of elements in the heap."""
        return len(self._data)

    #  Internal helpers

    @staticmethod
    def _parent(i: int) -> int:
        return (i - 1) // 2

    @staticmethod
    def _left(i: int) -> int:
        return 2 * i + 1

    @staticmethod
    def _right(i: int) -> int:
        return 2 * i + 2

    def _sift_up(self, i: int) -> None:
        """Bubble element at index i upward until heap property holds."""
        while i > 0:
            parent = self._parent(i)
            if self._data[i][0] < self._data[parent][0]:
                self._data[i], self._data[parent] = self._data[parent], self._data[i]
                i = parent
            else:
                break

    def _sift_down(self, i: int) -> None:
        """Push element at index i downward until heap property holds."""
        n = self.size()
        while True:
            smallest = i
            left = self._left(i)
            right = self._right(i)

            if left < n and self._data[left][0] < self._data[smallest][0]:
                smallest = left
            if right < n and self._data[right][0] < self._data[smallest][0]:
                smallest = right

            if smallest != i:
                self._data[i], self._data[smallest] = self._data[smallest], self._data[i]
                i = smallest
            else:
                break

    def _assert_heap_property(self) -> None:
        """Assert parent ≤ both children for every node (Subtask 10)."""
        n = self.size()
        for i in range(n):
            left = self._left(i)
            right = self._right(i)
            if left < n:
                assert self._data[i][0] <= self._data[left][0], (
                    f"Heap violation: node[{i}]={self._data[i]} > left[{left}]={self._data[left]}"
                )
            if right < n:
                assert self._data[i][0] <= self._data[right][0], (
                    f"Heap violation: node[{i}]={self._data[i]} > right[{right}]={self._data[right]}"
                )

    def __repr__(self) -> str:
        return f"MinHeap(size={self.size()}, min={self._data[0] if self._data else None})"


def build_inventory_priority_queue(
    df: pd.DataFrame,
) -> list[tuple[float, str]]:
    """
    Task 2 — full workflow:
      1. Compute restock_score = total_qty_sold / avg_unit_price per product
      2. Insert all (score, product) tuples into MinHeap
      3. Extract in ascending order → restocking priority list (lowest score = restock first)
      4. Print priority order
    Returns the sorted extraction list.
    """
    log.info("--- Starting Inventory Priority Queue ---")

    # Subtask 7: compute restock scores
    product_stats = (
        df.groupby("product")
        .agg(total_qty=("qty", "sum"), avg_price=("unit_price", "mean"))
        .reset_index()
    )
    product_stats["restock_score"] = (
        product_stats["total_qty"] / product_stats["avg_price"]
    ).round(6)

    log.info("Restock scores:\n%s", product_stats[[
             "product", "restock_score"]].to_string(index=False))

    # Subtask 8: insert into MinHeap
    heap = MinHeap()
    for _, row in product_stats.iterrows():
        heap.insert((row["restock_score"], row["product"]))

    log.info("Heap size after all insertions: %d", heap.size())

    # Subtask 9: extract in priority order
    priority_list: list[tuple[float, str]] = []
    while heap.size() > 0:
        priority_list.append(heap.extract_min())

    print("\n" + "=" * 50)
    print("  INVENTORY RESTOCKING PRIORITY (lowest score first)")
    print("=" * 50)
    for rank, (score, product) in enumerate(priority_list, start=1):
        print(f"  {rank}. {product:<10}  restock_score = {score:.6f}")
    print("=" * 50 + "\n")

    return priority_list

# TASK 3 — REVENUE VISUALIZATION DASHBOARD


def build_dashboard(df: pd.DataFrame) -> None:
    """
    Task 3 — 3-panel executive dashboard saved as sales_dashboard.png.

    Panel 1: Stacked area chart — monthly revenue per product
    Panel 2: Box plot — revenue distribution per region + fraud overlay
    Panel 3: Heatmap — avg revenue by product × hour-bin, annotated
    """
    log.info("--- Building Visualization Dashboard ---")

    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    fig.patch.set_facecolor("#F8F9FA")
    plt.suptitle(
        "E-Commerce Sales Intelligence Dashboard — Q1 2023",
        fontsize=16, fontweight="bold", y=1.01, color="#1A1A2E",
    )

    #  Panel 1: Stacked area chart
    ax1 = axes[0]
    ax1.set_facecolor("#FAFAFA")

    monthly_pivot = (
        df.pivot_table(
            values="revenue",
            index="order_month",
            columns="product",
            aggfunc="sum",
            fill_value=0,
        )
    )

    months = monthly_pivot.index.tolist()
    products = monthly_pivot.columns.tolist()
    colors = [PRODUCT_COLORS.get(p, "#999999") for p in products]
    ax1.stackplot(
        months,
        [monthly_pivot[p] for p in products],
        labels=products,
        colors=colors,
        alpha=0.85,
    )
    ax1.set_title("Monthly Revenue by Product",
                  fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Month", fontsize=10)
    ax1.set_ylabel("Revenue (₹)", fontsize=10)
    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"₹{x/1e6:.1f}M")
    )
    ax1.set_xticks(months)
    ax1.set_xticklabels([f"M{m}" for m in months])
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.8)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    #  Panel 2: Box plot + fraud overlay
    ax2 = axes[1]
    ax2.set_facecolor("#FAFAFA")

    regions = sorted(df["region"].unique())
    region_data = [df.loc[df["region"] == r,
                          "revenue"].values for r in regions]

    bp = ax2.boxplot(
        region_data,
        labels=regions,
        patch_artist=True,
        notch=False,
        widths=0.5,
    )
    region_palette = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    for patch, color in zip(bp["boxes"], region_palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Overlay fraud points as red crosses
    for idx_r, region in enumerate(regions):
        fraud_mask = (df["region"] == region) & df["is_fraud"]
        fraud_rev = df.loc[fraud_mask, "revenue"].values
        if len(fraud_rev) > 0:
            ax2.scatter(
                [idx_r + 1] * len(fraud_rev),
                fraud_rev,
                marker="x",
                color="red",
                zorder=5,
                s=80,
                linewidths=1.8,
                label="Fraud" if idx_r == 0 else "",
            )

    fraud_patch = mpatches.Patch(color="red", label="Fraud orders (×)")
    ax2.legend(handles=[fraud_patch], fontsize=8, loc="upper right")
    ax2.set_title("Revenue Distribution by Region",
                  fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Region", fontsize=10)
    ax2.set_ylabel("Revenue (₹)", fontsize=10)
    ax2.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"₹{x/1e3:.0f}K")
    )
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    #  Panel 3: Heatmap
    ax3 = axes[2]

    df["hour_bin"] = pd.cut(
        df["order_hour"],
        bins=HOUR_BINS,
        labels=HOUR_LABELS,
        right=False,
    )

    heatmap_data = (
        df.pivot_table(
            values="revenue",
            index="product",
            columns="hour_bin",
            aggfunc="mean",
            fill_value=0,
        )
    )
    # Ensure all hour-bin columns are present and ordered
    for label in HOUR_LABELS:
        if label not in heatmap_data.columns:
            heatmap_data[label] = 0.0
    heatmap_data = heatmap_data[HOUR_LABELS]

    matrix = heatmap_data.values
    im = ax3.imshow(matrix, aspect="auto", cmap="YlOrRd")

    ax3.set_xticks(range(len(HOUR_LABELS)))
    ax3.set_xticklabels(HOUR_LABELS, fontsize=9)
    ax3.set_yticks(range(len(heatmap_data.index)))
    ax3.set_yticklabels(heatmap_data.index.tolist(), fontsize=9)

    # Annotating each cell
    for row_i in range(matrix.shape[0]):
        for col_j in range(matrix.shape[1]):
            val = matrix[row_i, col_j]
            text_color = "white" if val > matrix.max() * 0.6 else "black"
            ax3.text(
                col_j, row_i,
                f"₹{val/1e3:.1f}K",
                ha="center", va="center",
                fontsize=8, color=text_color, fontweight="bold",
            )

    cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label("Avg Revenue (₹)", fontsize=9)
    cbar.ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"₹{x/1e3:.0f}K")
    )

    ax3.set_title("Avg Revenue: Product × Hour Bin",
                  fontsize=12, fontweight="bold", pad=10)
    ax3.set_xlabel("Hour Bin", fontsize=10)
    ax3.set_ylabel("Product", fontsize=10)

    #  Final layout & save
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(DASHBOARD_FILENAME, dpi=DASHBOARD_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Dashboard saved -> %s at %d DPI",
             DASHBOARD_FILENAME, DASHBOARD_DPI)

# TASK 4 — REGIONAL REVENUE MAP (FOLIUM)


def build_regional_map(df: pd.DataFrame) -> None:
    """
    Task 4 — Interactive Folium map with:
      - CircleMarkers per region (radius ∝ total revenue, colour = quartile)
      - Tooltips (Region, Total Revenue ₹, Top Product, Fraud Count)
      - MarkerCluster for individual order locations (jittered)
      - MiniMap plugin
    Saved as regional_sales_map.html.
    """
    log.info("--- Building Regional Revenue Map ---")

    #  Aggregate regional stats
    region_stats = (
        df.groupby("region")
        .agg(
            total_revenue=("revenue", "sum"),
            fraud_count=("is_fraud", "sum"),
        )
        .reset_index()
    )

    top_product = (
        df.groupby(["region", "product"])["revenue"]
        .sum()
        .reset_index()
        .sort_values("revenue", ascending=False)
        .drop_duplicates("region")
        .rename(columns={"product": "top_product"})
        [["region", "top_product"]]
    )

    region_stats = region_stats.merge(top_product, on="region", how="left")

    # Quartile colour logic
    q25 = region_stats["total_revenue"].quantile(0.25)
    q75 = region_stats["total_revenue"].quantile(0.75)

    def _marker_color(revenue: float) -> str:
        if revenue <= q25:
            return "green"
        elif revenue >= q75:
            return "red"
        return "orange"

    # Build map
    fmap = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM,
                      tiles="CartoDB positron")

    # Subtask 7 & 8: CircleMarkers with tooltip
    for _, row in region_stats.iterrows():
        region = row["region"]
        revenue = row["total_revenue"]
        coords = REGION_COORDS.get(region, MAP_CENTER)
        radius = max(revenue / MARKER_RADIUS_DIVISOR,
                     5.0)   # zoom start at 5px
        color = _marker_color(revenue)

        tooltip_text = (
            f"<b>{region}</b><br>"
            f"Total Revenue: ₹{revenue:,.0f}<br>"
            f"Top Product: {row['top_product']}<br>"
            f"Fraud Orders: {int(row['fraud_count'])}"
        )

        folium.CircleMarker(
            location=coords,
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.55,
            weight=2,
            tooltip=folium.Tooltip(tooltip_text),
            popup=folium.Popup(tooltip_text, max_width=250),
        ).add_to(fmap)

    # Subtask 9: MarkerCluster for individual orders (jittered)
    rng_jitter = np.random.default_rng(RANDOM_SEED)
    cluster = MarkerCluster(name="Individual Orders").add_to(fmap)

    for _, order in df.iterrows():
        region = order["region"]
        base = REGION_COORDS.get(region, MAP_CENTER)
        lat = base[0] + rng_jitter.uniform(-JITTER_RANGE, JITTER_RANGE)
        lon = base[1] + rng_jitter.uniform(-JITTER_RANGE, JITTER_RANGE)

        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color="blue" if not order["is_fraud"] else "red",
            fill=True,
            fill_opacity=0.4,
            tooltip=f"Order: {order['order_id']} | ₹{order['revenue']:,.0f}",
        ).add_to(cluster)

    # Subtask 10: MiniMap plugin
    MiniMap(toggle_display=True, position="bottomright").add_to(fmap)

    # Layer control for cluster toggle
    folium.LayerControl().add_to(fmap)

    fmap.save(MAP_FILENAME)
    log.info("Map saved -> %s", MAP_FILENAME)

# JSON EXPORT FOR WEB DASHBOARD


def export_dashboard_json(clean_df: pd.DataFrame, priority_list: list[tuple[float, str]]) -> None:
    """
    Serialise all dashboard data to dashboard_data.json so the
    salesiq-dashboard.html can consume real ETL output without a backend.
    """
    log.info("--- Exporting Dashboard JSON ---")

    #  Per-product summary
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

    #  Daily revenue trend
    clean_df["order_day"] = clean_df["order_date"].dt.day
    daily = (
        clean_df.groupby(["order_day", "product"])["revenue"]
        .sum()
        .reset_index()
        .round(2)
    )
    daily_list = daily.to_dict(orient="records")

    #  Hourly heatmap (product x hour_bin)
    clean_df["hour_bin"] = pd.cut(
        clean_df["order_hour"], bins=HOUR_BINS, labels=HOUR_LABELS, right=False
    ).astype(str)
    heatmap = (
        clean_df.groupby(["product", "hour_bin"])["revenue"]
        .mean()
        .reset_index()
        .round(2)
    )

    #  Fraud orders
    fraud_orders = (
        clean_df[clean_df["is_fraud"]]
        [["order_id", "product", "region", "revenue", "order_date"]]
        .copy()
    )
    fraud_orders["order_date"] = fraud_orders["order_date"].dt.strftime(
        "%Y-%m-%d %H:%M")
    fraud_orders = fraud_orders.round(2)

    #  All cleaned orders (for the orders table)
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
    insights.append(
        f"{top_reg} region contributed {top_reg_pct:.0f}% of total sales.")

    # Fraud Analysis
    fraud_df = clean_df[clean_df["is_fraud"]]
    if len(fraud_df) > 0:
        worst_reg = fraud_df.groupby("region").size().idxmax()
        insights.append(f"Most fraud occurred in {worst_reg} region.")
        recommendations.append(
            f"Investigate fraudulent transactions in {worst_reg}.")
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
    # pluralize product name nicely for Table -> Tablets, Earbuds -> Earbuds, Laptop -> Laptops, etc.
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

    log.info("Dashboard JSON exported -> %s and %s",
             DASHBOARD_JSON_FILENAME, dashboard_js_filename)


# ENTRY POINT
def main() -> None:
    """Orchestrate all four hackathon tasks end-to-end."""
    log.info("============================================")
    log.info("  E-Commerce Sales Intelligence - Set B     ")
    log.info("============================================")

    #  Task 1: ETL
    raw_df = generate_raw_data()
    pipeline = ETLPipeline(raw_df)
    clean_df, product_summary = pipeline.run()

    #  Task 2: Min-Heap
    priority_list = build_inventory_priority_queue(clean_df)

    #  Task 3: Dashboard
    build_dashboard(clean_df)

    #  Task 4: Map
    build_regional_map(clean_df)

    # Web Dashboard JSON
    export_dashboard_json(clean_df, priority_list)

    log.info("============================================")
    log.info("  ALL TASKS COMPLETE")
    log.info("  Outputs:")
    log.info("    * %s", PRODUCT_SUMMARY_FILENAME)
    log.info("    * %s", DASHBOARD_FILENAME)
    log.info("    * %s", MAP_FILENAME)
    log.info("    * %s", DASHBOARD_JSON_FILENAME)
    log.info("============================================")


if __name__ == "__main__":
    main()
