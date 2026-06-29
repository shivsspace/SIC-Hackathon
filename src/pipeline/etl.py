from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from src.pipeline.config import (
    RANDOM_SEED,
    N_RECORDS,
    FRAUD_STD_MULTIPLIER,
    PRODUCT_SUMMARY_FILENAME,
)

log = logging.getLogger(__name__)

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

    # Noise injection (must NOT be modified)
    idx = rng.choice(N_RECORDS, size=60, replace=False)
    for i in idx[:12]:
        unit_price[i] = np.nan
    for i in idx[12:22]:
        unit_price[i] = f"₹{float(unit_price[i]):.2f}"   # ₹ prefix
    for i in idx[22:32]:
        qty[i] = 0                                              # zero qty
    for i in idx[32:42]:
        product[i] = product[i].upper()                        # ALL CAPS

    disc: np.ndarray = np.random.choice(
        [0, 5, 10, 15, 20, np.nan], N_RECORDS
    ).astype(object)
    for i in idx[42:50]:
        disc[i] = 150                                           # impossible discount
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
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)  # 10 duplicate rows
    log.info("Raw dataset shape: %s", df.shape)
    return df


class ETLPipeline:
    """
    Production ETL pipeline for e-commerce order data.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df: pd.DataFrame = df.copy()
        self._validate_required_columns()

    def _validate_required_columns(self) -> None:
        required = {
            "order_id", "product", "category", "qty",
            "unit_price", "discount", "region", "order_date",
        }
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def drop_duplicates(self) -> ETLPipeline:
        """Remove fully-duplicate rows."""
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        removed = before - len(self.df)
        log.info("Duplicates removed: %d (rows remaining: %d)", removed, len(self.df))
        assert removed >= 10, f"Expected ≥10 duplicates, found {removed}"
        return self

    def remove_zero_qty(self) -> ETLPipeline:
        """Remove orders with qty = 0 (invalid)."""
        self.df["qty"] = pd.to_numeric(self.df["qty"], errors="coerce")
        before = len(self.df)
        self.df = self.df[self.df["qty"] > 0].copy()
        self.df.reset_index(drop=True, inplace=True)
        removed = before - len(self.df)
        log.info("Zero-qty rows removed: %d", removed)
        assert removed >= 10, f"Expected ≥10 zero-qty rows, found {removed}"
        return self

    def clean_unit_price(self) -> ETLPipeline:
        """Strip ₹ prefix, coerce to float, impute NaN with product-wise median."""
        self.df["unit_price"] = (
            self.df["unit_price"]
            .astype(str)
            .str.replace("₹", "", regex=False)
            .str.strip()
        )
        self.df["unit_price"] = pd.to_numeric(self.df["unit_price"], errors="coerce")

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

        assert self.df["unit_price"].isna().sum() == 0, "NaNs remain in unit_price"
        assert (self.df["unit_price"] > 0).all(), "Non-positive unit_price detected"
        log.info("unit_price cleaned *")
        return self

    def clean_discount(self) -> ETLPipeline:
        """Replace impossible discount values (>100) with NaN, then product-median impute."""
        self.df["discount"] = pd.to_numeric(self.df["discount"], errors="coerce")
        invalid_count = (self.df["discount"] > 100).sum()
        log.info("Impossible discount values (>100): %d", invalid_count)

        self.df.loc[self.df["discount"] > 100, "discount"] = np.nan

        product_median = (
            self.df.groupby("product")["discount"]
            .transform("median")
        )
        self.df["discount"] = self.df["discount"].fillna(product_median)
        self.df["discount"] = self.df["discount"].fillna(0.0)  # fallback

        assert (self.df["discount"] <= 100).all(), "Discount > 100 still present"
        assert self.df["discount"].isna().sum() == 0, "NaNs remain in discount"
        log.info("discount cleaned *")
        return self

    def normalise_product(self) -> ETLPipeline:
        """Convert ALL-CAPS product names to Title Case."""
        before_unique = set(self.df["product"].unique())
        self.df["product"] = self.df["product"].str.title()
        after_unique = set(self.df["product"].unique())
        log.info("Product normalisation: %d -> %d unique values", len(before_unique), len(after_unique))
        return self

    def normalise_region(self) -> ETLPipeline:
        """Replace 'UNKNOWN' region with 'Other'."""
        unknown_count = (self.df["region"] == "UNKNOWN").sum()
        log.info("UNKNOWN region rows: %d", unknown_count)
        self.df["region"] = self.df["region"].replace("UNKNOWN", "Other")
        assert "UNKNOWN" not in self.df["region"].values, "UNKNOWN still present in region"
        log.info("region normalised *")
        return self

    def compute_derived_features(self) -> ETLPipeline:
        """
        Compute derived features and identify fraud records.
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
