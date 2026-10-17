"""Load, validate, clean, and enrich the Parcl client and property data."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd


CLIENT_REQUIRED_COLUMNS = {
    "client_id",
    "client_type",
    "date_of_birth",
    "gender",
    "country",
    "region",
    "acquisition_purpose",
    "satisfaction_score",
    "loan_applied",
    "referral_channel",
}
PROPERTY_REQUIRED_COLUMNS = {
    "listing_id",
    "transaction_date",
    "unit_category",
    "floor_area_sqft",
    "sale_price",
    "listing_status",
    "client_ref",
}


class PreparedData(NamedTuple):
    """Clean source tables and the one-row-per-client modeling table."""

    clients: pd.DataFrame
    properties: pd.DataFrame
    client_features: pd.DataFrame
    numeric_features: list[str]
    categorical_features: list[str]
    as_of_date: pd.Timestamp


def _normalise_text(series: pd.Series) -> pd.Series:
    """Trim text and standardise whitespace without changing business labels."""
    return series.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)


def _parse_currency(series: pd.Series) -> pd.Series:
    """Convert values such as '$300,385.62' to numeric dollars."""
    cleaned = series.astype("string").str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def _age_on_date(dates_of_birth: pd.Series, as_of_date: pd.Timestamp) -> pd.Series:
    """Calculate completed years as at a stated reporting date."""
    age = as_of_date.year - dates_of_birth.dt.year
    had_birthday = (dates_of_birth.dt.month < as_of_date.month) | (
        (dates_of_birth.dt.month == as_of_date.month)
        & (dates_of_birth.dt.day <= as_of_date.day)
    )
    return (age - (~had_birthday).astype("int64")).astype("float64")


def _ensure_columns(frame: pd.DataFrame, required: set[str], file_name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{file_name} is missing required columns: {', '.join(missing)}")


def _validate_values(clients: pd.DataFrame, properties: pd.DataFrame) -> None:
    if clients["client_id"].isna().any() or clients["client_id"].duplicated().any():
        raise ValueError("client_id must be present and unique in clients.csv.")
    if properties["listing_id"].isna().any() or properties["listing_id"].duplicated().any():
        raise ValueError("listing_id must be present and unique in properties.csv.")
    if properties["sale_price"].isna().any() or (properties["sale_price"] <= 0).any():
        raise ValueError("Every property must have a positive, parseable sale_price.")
    if properties["floor_area_sqft"].isna().any() or (properties["floor_area_sqft"] <= 0).any():
        raise ValueError("Every property must have a positive floor_area_sqft.")

    sold = properties[properties["listing_status"].str.casefold().eq("sold")]
    unresolved_sold = ~sold["client_ref"].isin(clients["client_id"])
    if unresolved_sold.any():
        raise ValueError("Every sold property must reference a client in clients.csv.")


def load_and_prepare_data(data_dir: str | Path) -> PreparedData:
    """Return a validated modeling dataset with purchase behaviour joined to clients.

    The source dates are interpreted month-first (MM-DD-YYYY or MM/DD/YYYY), which
    is confirmed by the monthly transaction dates in the supplied property data.
    """
    data_dir = Path(data_dir)
    clients_path = data_dir / "clients.csv"
    properties_path = data_dir / "properties.csv"
    if not clients_path.exists() or not properties_path.exists():
        raise FileNotFoundError("clients.csv and properties.csv must both be in the data directory.")

    clients = pd.read_csv(clients_path)
    properties = pd.read_csv(properties_path)
    _ensure_columns(clients, CLIENT_REQUIRED_COLUMNS, "clients.csv")
    _ensure_columns(properties, PROPERTY_REQUIRED_COLUMNS, "properties.csv")

    # Remove only exact duplicate records; preserve legitimate repeat property purchases.
    clients = clients.drop_duplicates().copy()
    properties = properties.drop_duplicates().copy()
    client_text_columns = [
        column for column in clients.columns if pd.api.types.is_string_dtype(clients[column])
    ]
    property_text_columns = [
        column for column in properties.columns if pd.api.types.is_string_dtype(properties[column])
    ]
    for column in client_text_columns:
        clients[column] = _normalise_text(clients[column])
    for column in property_text_columns:
        properties[column] = _normalise_text(properties[column])

    clients["date_of_birth"] = pd.to_datetime(
        clients["date_of_birth"], format="mixed", dayfirst=False, errors="coerce"
    )
    properties["transaction_date"] = pd.to_datetime(
        properties["transaction_date"], format="mixed", dayfirst=False, errors="coerce"
    )
    properties["sale_price"] = _parse_currency(properties["sale_price"])
    properties["floor_area_sqft"] = pd.to_numeric(properties["floor_area_sqft"], errors="coerce")
    clients["satisfaction_score"] = pd.to_numeric(clients["satisfaction_score"], errors="coerce")
    _validate_values(clients, properties)

    if properties["transaction_date"].notna().any():
        as_of_date = properties["transaction_date"].max().normalize()
    else:
        as_of_date = pd.Timestamp.today().normalize()
    clients["age"] = _age_on_date(clients["date_of_birth"], as_of_date)

    sold_properties = properties.loc[
        properties["listing_status"].str.casefold().eq("sold") & properties["client_ref"].notna()
    ].copy()
    sold_properties["is_office"] = sold_properties["unit_category"].str.casefold().eq("office").astype(int)
    purchase_summary = (
        sold_properties.groupby("client_ref", as_index=False)
        .agg(
            purchase_count=("listing_id", "nunique"),
            total_spend=("sale_price", "sum"),
            average_purchase_price=("sale_price", "mean"),
            average_floor_area_sqft=("floor_area_sqft", "mean"),
            office_purchase_share=("is_office", "mean"),
            latest_purchase_date=("transaction_date", "max"),
        )
        .rename(columns={"client_ref": "client_id"})
    )
    client_features = clients.merge(purchase_summary, on="client_id", how="left", validate="one_to_one")
    purchase_columns = [
        "purchase_count",
        "total_spend",
        "average_purchase_price",
        "average_floor_area_sqft",
        "office_purchase_share",
    ]
    client_features[purchase_columns] = client_features[purchase_columns].fillna(0.0)
    client_features["is_international"] = (
        ~client_features["country"].str.casefold().eq("usa")
    ).astype(int)

    # Gender is intentionally retained for descriptive fairness review only, not used to target clusters.
    numeric_features = [
        "age",
        "satisfaction_score",
        "purchase_count",
        "total_spend",
        "average_purchase_price",
        "average_floor_area_sqft",
        "office_purchase_share",
    ]
    categorical_features = [
        "client_type",
        "country",
        "region",
        "acquisition_purpose",
        "loan_applied",
        "referral_channel",
    ]
    return PreparedData(
        clients=clients,
        properties=properties,
        client_features=client_features,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        as_of_date=as_of_date,
    )
