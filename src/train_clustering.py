"""Train, evaluate, and export the buyer-segmentation model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if __package__ in {None, ""}:
    # Allow the documented `python src/train_clustering.py` command from project root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_preparation import load_and_prepare_data


RANDOM_STATE = 42


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """Create the reproducible encoding and scaling workflow."""
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def evaluate_cluster_counts(features: np.ndarray, maximum_clusters: int = 8) -> pd.DataFrame:
    """Score practical K values using inertia and silhouette score."""
    maximum_clusters = min(maximum_clusters, len(features) - 1)
    if maximum_clusters < 2:
        raise ValueError("At least three client records are required for cluster evaluation.")
    rows: list[dict[str, float | int]] = []
    for cluster_count in range(2, maximum_clusters + 1):
        model = KMeans(n_clusters=cluster_count, n_init=25, random_state=RANDOM_STATE)
        labels = model.fit_predict(features)
        rows.append(
            {
                "cluster_count": cluster_count,
                "inertia": float(model.inertia_),
                "silhouette_score": float(silhouette_score(features, labels)),
            }
        )
    return pd.DataFrame(rows)


def select_cluster_count(scores: pd.DataFrame, requested_count: str) -> int:
    """Use a requested K or the silhouette-optimal K, resolving ties toward simpler segments."""
    if requested_count != "auto":
        requested = int(requested_count)
        if requested not in scores["cluster_count"].tolist():
            available = ", ".join(map(str, scores["cluster_count"].tolist()))
            raise ValueError(f"--n-clusters must be one of {available}, or 'auto'.")
        return requested
    return int(
        scores.sort_values(["silhouette_score", "cluster_count"], ascending=[False, True]).iloc[0]["cluster_count"]
    )


def _mode_or_unknown(series: pd.Series) -> str:
    cleaned = series.dropna()
    return "Unknown" if cleaned.empty else str(cleaned.mode().iat[0])


def create_segment_names(profile: pd.DataFrame) -> dict[int, str]:
    """Give each cluster an evidence-based, non-sensitive business label."""
    overall = {
        "purchase_count": profile["average_purchase_count"].mean(),
        "purchase_price": profile["average_purchase_price"].mean(),
        "floor_area": profile["average_floor_area_sqft"].mean(),
        "office_share": profile["average_office_purchase_share"].mean(),
    }
    names: dict[int, str] = {}
    used: dict[str, int] = {}
    for row in profile.sort_values("cluster_id").itertuples(index=False):
        if row.average_purchase_count >= overall["purchase_count"] * 1.5:
            base = "Multi-property buyers"
        elif row.average_office_purchase_share >= max(0.20, overall["office_share"] + 0.12):
            base = "Mixed-use buyers"
        elif (
            row.average_purchase_price >= overall["purchase_price"] * 1.15
            and row.average_floor_area_sqft >= overall["floor_area"] * 1.15
        ):
            base = "Premium-property buyers"
        elif row.average_office_purchase_share <= 0.02:
            base = "Apartment-focused buyers"
        else:
            base = "Residential buyers"
        used[base] = used.get(base, 0) + 1
        suffix = f" ({used[base]})" if used[base] > 1 else ""
        names[int(row.cluster_id)] = f"C{int(row.cluster_id)} - {base}{suffix}"
    return names


def build_cluster_profile(segmented_clients: pd.DataFrame) -> pd.DataFrame:
    """Summarise the interpretable business traits of each final segment."""
    profile = (
        segmented_clients.groupby("cluster_id", as_index=False)
        .agg(
            clients=("client_id", "nunique"),
            average_age=("age", "mean"),
            average_satisfaction=("satisfaction_score", "mean"),
            average_purchase_count=("purchase_count", "mean"),
            average_total_spend=("total_spend", "mean"),
            average_purchase_price=("average_purchase_price", "mean"),
            average_floor_area_sqft=("average_floor_area_sqft", "mean"),
            average_office_purchase_share=("office_purchase_share", "mean"),
            investment_rate=("is_investment", "mean"),
            loan_rate=("is_loan", "mean"),
            company_rate=("is_company", "mean"),
            international_rate=("is_international", "mean"),
            top_country=("country", _mode_or_unknown),
            top_region=("region", _mode_or_unknown),
            top_referral_channel=("referral_channel", _mode_or_unknown),
        )
    )
    names = create_segment_names(profile)
    profile.insert(1, "segment", profile["cluster_id"].map(names))
    return profile


def build_recommendation(row: pd.Series) -> str:
    """Provide a recommendation grounded in the cluster's measured dominant behaviour."""
    if row["average_purchase_count"] >= 5:
        return "Offer portfolio reviews, multi-unit options, and a relationship-managed sales journey."
    if row["average_office_purchase_share"] >= 0.20:
        return "Show both office and residential inventory, with use-case and yield comparisons."
    if row["average_purchase_price"] >= 375000 and row["average_floor_area_sqft"] >= 1200:
        return "Use a premium-property journey with detailed unit specifications and appointment-led viewings."
    if row["average_office_purchase_share"] <= 0.02:
        return "Prioritise apartment inventory, neighbourhood guidance, and a straightforward home-search journey."
    return "Lead with relevant property options and tailored purchase support."


def write_markdown_report(
    path: Path,
    profile: pd.DataFrame,
    scores: pd.DataFrame,
    chosen_clusters: int,
    hierarchical_silhouette: float,
    adjusted_rand: float,
    as_of_date: pd.Timestamp,
    data_summary: dict[str, float | int],
) -> None:
    """Export a research-ready paper from actual model outputs and EDA statistics."""
    profile_display = profile.copy()
    percentage_columns = [
        "investment_rate",
        "loan_rate",
        "company_rate",
        "international_rate",
        "average_office_purchase_share",
    ]
    for column in percentage_columns:
        profile_display[column] = (profile_display[column] * 100).round(1).astype(str) + "%"
    currency_columns = ["average_total_spend", "average_purchase_price"]
    for column in currency_columns:
        profile_display[column] = profile_display[column].map(lambda value: f"${value:,.0f}")
    profile_display["average_age"] = profile_display["average_age"].round(1)
    profile_display["average_satisfaction"] = profile_display["average_satisfaction"].round(2)
    profile_display["average_purchase_count"] = profile_display["average_purchase_count"].round(2)

    silhouette_best = scores.sort_values(["silhouette_score", "cluster_count"], ascending=[False, True]).iloc[0]
    selection_interpretation = (
        f"The four-group solution was selected to meet the project brief. The highest silhouette score in the "
        f"automatic comparison was {silhouette_best['silhouette_score']:.3f} at K={int(silhouette_best['cluster_count'])}, "
        f"so K=4 should be treated as a business-reporting choice rather than the statistically strongest K."
        if chosen_clusters != int(silhouette_best["cluster_count"])
        else f"K={chosen_clusters} had the highest silhouette score among the candidate solutions."
    )
    report = [
        "# Machine Learning Based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence",
        "",
        "## Abstract",
        "",
        f"This analysis segments {data_summary['clients']:,} Parcl clients using demographic, acquisition, financing, and completed-purchase information. "
        f"The source data contains {data_summary['properties']:,} property records: {data_summary['sold_properties']:,} completed purchases "
        f"and {data_summary['available_listings']:,} available listings. A K-means model produces {chosen_clusters} actionable buyer groups, "
        "supported by an interactive dashboard for geographic and investment-behaviour exploration.",
        "",
        "## 1. Business problem",
        "",
        "Parcl needs to replace broad buyer outreach with evidence-based groups that reflect how clients buy property, finance purchases, and engage across markets. "
        "The objective is to support more relevant property recommendations, improve marketing efficiency, and identify distinct investment behaviours without using protected characteristics for targeting.",
        "",
        "## 2. Dataset and data quality",
        "",
        "The client table was joined to sold properties on `client_id` / `client_ref`. Available listings were not treated as purchases because they do not have a buyer reference.",
        "",
        "| Measure | Result |",
        "|---|---:|",
        f"| Client records modelled | {data_summary['clients']:,} |",
        f"| Property records | {data_summary['properties']:,} |",
        f"| Completed purchases | {data_summary['sold_properties']:,} |",
        f"| Available listings excluded from buyer behaviour | {data_summary['available_listings']:,} |",
        f"| Countries represented | {data_summary['countries']:,} |",
        f"| Regions represented | {data_summary['regions']:,} |",
        "",
        "Cleaning removed exact duplicate rows, trimmed categorical labels, parsed date and currency fields, validated positive property values, and required each completed purchase to resolve to a known client. "
        "The supplied date convention was interpreted as month-first, consistent with the monthly property transaction records.",
        "",
        "## 3. Exploratory data analysis",
        "",
        "| Client and purchase measure | Result |",
        "|---|---:|",
        f"| Individual clients | {data_summary['individual_share']:.1%} |",
        f"| Investment-purpose clients | {data_summary['investment_share']:.1%} |",
        f"| Clients applying for a loan | {data_summary['loan_share']:.1%} |",
        f"| Median age at reporting date | {data_summary['median_age']:.0f} years |",
        f"| Age range | {data_summary['minimum_age']:.0f}–{data_summary['maximum_age']:.0f} years |",
        f"| Mean satisfaction score | {data_summary['average_satisfaction']:.2f} / 5 |",
        f"| Median completed-purchase spend per client | ${data_summary['median_total_spend']:,.0f} |",
        f"| Mean completed-purchase spend per client | ${data_summary['average_total_spend']:,.0f} |",
        "",
        "The purchase measures show meaningful variation in property price, floor area, office share, and number of completed purchases. These behaviours provide the main signal separating the segments; acquisition purpose and loan use are valuable contextual descriptors but less differentiated in this sample.",
        "",
        "## 4. Modeling methodology",
        "",
        "The workflow uses one row per client. Derived purchase features are purchase count, total spend, average purchase price, average floor area, and office-purchase share. Client features are age, satisfaction score, client type, country, region, acquisition purpose, loan status, and referral channel.",
        "",
        "Numeric features are median-imputed and standardised. Categorical features are mode-imputed and one-hot encoded. K-means is evaluated for K=2 through K=8 using inertia for elbow inspection and silhouette score for cohesion and separation. Hierarchical agglomerative clustering is calculated at the final K as a secondary validation check. Names, IDs, raw date of birth, and latest purchase date are excluded from the model. Gender is retained only for descriptive fairness review and is not a cluster input.",
        "",
        selection_interpretation,
        "",
        f"The final K-means silhouette score is **{scores.loc[scores['cluster_count'].eq(chosen_clusters), 'silhouette_score'].iat[0]:.3f}**. "
        f"Hierarchical validation has a silhouette score of **{hierarchical_silhouette:.3f}** and an adjusted Rand index of **{adjusted_rand:.3f}** against K-means. These modest validation values mean the segments should guide exploration and campaign design, not be treated as fixed natural categories.",
        "",
        "### Cluster-selection diagnostics",
        "",
        scores.round(4).to_markdown(index=False),
        "",
        "## 5. Segment profiles",
        "",
        profile_display.to_markdown(index=False),
        "",
        "## 6. Recommendations",
        "",
    ]
    for row in profile.sort_values("cluster_id").iterrows():
        values = row[1]
        report.append(f"- **{values['segment']}**: {build_recommendation(values)}")
    report.extend(
        [
            "",
            "## 7. Implementation and governance",
            "",
            f"The Streamlit dashboard reports results as at {as_of_date:%Y-%m-%d} and allows filtering by country, region, acquisition purpose, and client type. "
            "It includes cluster distribution, investor behaviour, geography, descriptive statistics, and model-quality diagnostics.",
            "",
            "These segments describe patterns in the supplied records. They are decision-support groups, not predictions of an individual buyer's value or eligibility. Re-train the model when material new client or sales data is available, monitor stability of the selected K, and review campaign outcomes for fairness before operational use.",
        ]
    )
    path.write_text("\n".join(report), encoding="utf-8")


def train_and_export(data_dir: Path, n_clusters: str = "auto", max_clusters: int = 8) -> dict[str, object]:
    """Execute the full training workflow and save all dashboard inputs."""
    project_root = Path(__file__).resolve().parents[1]
    outputs_dir = project_root / "outputs"
    models_dir = project_root / "models"
    reports_dir = project_root / "reports"
    outputs_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    prepared = load_and_prepare_data(data_dir)
    feature_columns = prepared.numeric_features + prepared.categorical_features
    raw_features = prepared.client_features[feature_columns].copy()
    preprocessor = build_preprocessor(prepared.numeric_features, prepared.categorical_features)
    transformed_features = preprocessor.fit_transform(raw_features)
    scores = evaluate_cluster_counts(transformed_features, max_clusters)
    chosen_clusters = select_cluster_count(scores, n_clusters)

    kmeans = KMeans(n_clusters=chosen_clusters, n_init=25, random_state=RANDOM_STATE)
    kmeans_labels = kmeans.fit_predict(transformed_features)
    hierarchical = AgglomerativeClustering(n_clusters=chosen_clusters, linkage="ward")
    hierarchical_labels = hierarchical.fit_predict(transformed_features)
    hierarchical_silhouette = float(silhouette_score(transformed_features, hierarchical_labels))
    adjusted_rand = float(adjusted_rand_score(kmeans_labels, hierarchical_labels))

    segmented_clients = prepared.client_features.copy()
    segmented_clients["cluster_id"] = kmeans_labels + 1
    segmented_clients["is_investment"] = segmented_clients["acquisition_purpose"].str.casefold().eq("investment").astype(int)
    segmented_clients["is_loan"] = segmented_clients["loan_applied"].str.casefold().eq("yes").astype(int)
    segmented_clients["is_company"] = segmented_clients["client_type"].str.casefold().eq("company").astype(int)
    profile = build_cluster_profile(segmented_clients)
    segment_names = dict(zip(profile["cluster_id"], profile["segment"]))
    segmented_clients["segment"] = segmented_clients["cluster_id"].map(segment_names)
    segmented_clients = segmented_clients.sort_values(["cluster_id", "client_id"]).reset_index(drop=True)

    scores["selected"] = scores["cluster_count"].eq(chosen_clusters)
    scores.to_csv(outputs_dir / "cluster_selection.csv", index=False)
    segmented_clients.to_csv(outputs_dir / "segmented_clients.csv", index=False)
    profile.to_csv(outputs_dir / "cluster_profiles.csv", index=False)
    metadata = {
        "reporting_date": str(prepared.as_of_date.date()),
        "clients": int(len(segmented_clients)),
        "properties": int(len(prepared.properties)),
        "sold_properties": int(prepared.properties["listing_status"].str.casefold().eq("sold").sum()),
        "available_listings": int(prepared.properties["listing_status"].str.casefold().eq("available").sum()),
        "selected_clusters": chosen_clusters,
        "selection_method": "manual" if n_clusters != "auto" else "highest silhouette score (ties favour fewer clusters)",
        "kmeans_silhouette_score": float(silhouette_score(transformed_features, kmeans_labels)),
        "hierarchical_silhouette_score": hierarchical_silhouette,
        "adjusted_rand_index": adjusted_rand,
        "excluded_from_model": ["client_id", "first_name", "last_name", "date_of_birth", "gender", "latest_purchase_date"],
        "numeric_features": prepared.numeric_features,
        "categorical_features": prepared.categorical_features,
    }
    data_summary = {
        "clients": int(len(segmented_clients)),
        "properties": int(len(prepared.properties)),
        "sold_properties": int(prepared.properties["listing_status"].str.casefold().eq("sold").sum()),
        "available_listings": int(prepared.properties["listing_status"].str.casefold().eq("available").sum()),
        "countries": int(segmented_clients["country"].nunique()),
        "regions": int(segmented_clients["region"].nunique()),
        "individual_share": float(segmented_clients["client_type"].str.casefold().eq("individual").mean()),
        "investment_share": float(segmented_clients["is_investment"].mean()),
        "loan_share": float(segmented_clients["is_loan"].mean()),
        "median_age": float(segmented_clients["age"].median()),
        "minimum_age": float(segmented_clients["age"].min()),
        "maximum_age": float(segmented_clients["age"].max()),
        "average_satisfaction": float(segmented_clients["satisfaction_score"].mean()),
        "median_total_spend": float(segmented_clients["total_spend"].median()),
        "average_total_spend": float(segmented_clients["total_spend"].mean()),
    }
    (outputs_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "model": kmeans,
            "feature_columns": feature_columns,
            "numeric_features": prepared.numeric_features,
            "categorical_features": prepared.categorical_features,
            "segment_names": segment_names,
            "metadata": metadata,
        },
        models_dir / "buyer_segmentation_kmeans.joblib",
    )
    write_markdown_report(
        reports_dir / "research_paper.md",
        profile,
        scores,
        chosen_clusters,
        hierarchical_silhouette,
        adjusted_rand,
        prepared.as_of_date,
        data_summary,
    )
    return metadata


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Parcl's buyer segmentation model.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Folder containing clients.csv and properties.csv (default: project root).",
    )
    parser.add_argument(
        "--n-clusters",
        default="auto",
        help="'auto' for silhouette-based selection, or an integer from 2 to 8.",
    )
    parser.add_argument("--max-clusters", type=int, default=8, help="Largest K to evaluate (default: 8).")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    result = train_and_export(arguments.data_dir, str(arguments.n_clusters), arguments.max_clusters)
    print(
        f"Training complete: {result['clients']:,} clients, "
        f"{result['selected_clusters']} clusters, "
        f"K-means silhouette {result['kmeans_silhouette_score']:.3f}."
    )
