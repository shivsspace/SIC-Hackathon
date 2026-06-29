import logging
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster, MiniMap
from src.pipeline.config import (
    REGION_COORDS,
    MAP_CENTER,
    MAP_ZOOM,
    MARKER_RADIUS_DIVISOR,
    JITTER_RANGE,
    RANDOM_SEED,
    MAP_FILENAME,
)

log = logging.getLogger(__name__)

def build_regional_map(df: pd.DataFrame) -> None:
    """
    Task 4 — Interactive Folium map with regional markers and order clustering.
    """
    log.info("--- Building Regional Revenue Map ---")

    # Aggregate regional stats
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
    fmap = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM, tiles="CartoDB positron")

    # CircleMarkers with tooltip
    for _, row in region_stats.iterrows():
        region = row["region"]
        revenue = row["total_revenue"]
        coords = REGION_COORDS.get(region, MAP_CENTER)
        radius = max(revenue / MARKER_RADIUS_DIVISOR, 5.0)  # min 5px
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

    # MarkerCluster for individual orders (jittered)
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

    # MiniMap plugin
    MiniMap(toggle_display=True, position="bottomright").add_to(fmap)

    # Layer control for cluster toggle
    folium.LayerControl().add_to(fmap)

    fmap.save(MAP_FILENAME)
    log.info("Map saved -> %s", MAP_FILENAME)
