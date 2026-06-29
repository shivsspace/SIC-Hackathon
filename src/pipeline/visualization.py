import logging
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from src.pipeline.config import (
    PRODUCT_COLORS,
    DASHBOARD_FILENAME,
    DASHBOARD_DPI,
    HOUR_BINS,
    HOUR_LABELS,
)

log = logging.getLogger(__name__)

def build_dashboard(df: pd.DataFrame) -> None:
    """
    Task 3 — 3-panel executive dashboard saved as sales_dashboard.png.
    """
    log.info("--- Building Visualization Dashboard ---")

    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    fig.patch.set_facecolor("#F8F9FA")
    plt.suptitle(
        "E-Commerce Sales Intelligence Dashboard — Q1 2023",
        fontsize=16, fontweight="bold", y=1.01, color="#1A1A2E",
    )

    # Panel 1: Stacked area chart
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
    ax1.set_title("Monthly Revenue by Product", fontsize=12, fontweight="bold", pad=10)
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

    # Panel 2: Box plot + fraud overlay
    ax2 = axes[1]
    ax2.set_facecolor("#FAFAFA")

    regions = sorted(df["region"].unique())
    region_data = [df.loc[df["region"] == r, "revenue"].values for r in regions]

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
    ax2.set_title("Revenue Distribution by Region", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Region", fontsize=10)
    ax2.set_ylabel("Revenue (₹)", fontsize=10)
    ax2.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"₹{x/1e3:.0f}K")
    )
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Panel 3: Heatmap
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

    ax3.set_title("Avg Revenue: Product × Hour Bin", fontsize=12, fontweight="bold", pad=10)
    ax3.set_xlabel("Hour Bin", fontsize=10)
    ax3.set_ylabel("Product", fontsize=10)

    # Final layout & save
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(DASHBOARD_FILENAME, dpi=DASHBOARD_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Dashboard saved -> %s at %d DPI", DASHBOARD_FILENAME, DASHBOARD_DPI)
