

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
STYLESHEET = PROJECT_ROOT / "assets" / "industrial.css"
FILTER_KEYS = ("filter_country", "filter_region", "filter_purpose", "filter_client_type")
COUNTRY_ISO3 = {
    "Australia": "AUS",
    "Belgium": "BEL",
    "Canada": "CAN",
    "Denmark": "DNK",
    "France": "FRA",
    "Germany": "DEU",
    "Mexico": "MEX",
    "Russia": "RUS",
    "UK": "GBR",
    "USA": "USA",
}
CHART_COLORS = ["#c83343", "#425466", "#8a7354", "#718096", "#586e75", "#a3b1c6"]


st.set_page_config(
    page_title="Real Estate Market Intelligence",
    page_icon=":material/precision_manufacturing:",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Read dashboard data exported by the reproducible clustering workflow."""
    required_files = [
        OUTPUTS_DIR / "segmented_clients.csv",
        OUTPUTS_DIR / "cluster_profiles.csv",
        OUTPUTS_DIR / "cluster_selection.csv",
        OUTPUTS_DIR / "model_metadata.json",
    ]
    missing = [file.name for file in required_files if not file.exists()]
    if missing:
        raise FileNotFoundError(
            "The dashboard needs trained model outputs. Run: "
            "python src/train_clustering.py --data-dir . --n-clusters 4"
        )
    clients = pd.read_csv(
        OUTPUTS_DIR / "segmented_clients.csv",
        parse_dates=["date_of_birth", "latest_purchase_date"],
    )
    profiles = pd.read_csv(OUTPUTS_DIR / "cluster_profiles.csv")
    scores = pd.read_csv(OUTPUTS_DIR / "cluster_selection.csv")
    metadata = json.loads((OUTPUTS_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    return clients, profiles, scores, metadata


def reset_filters() -> None:
    """Return the global filter control bay to its neutral state."""
    for key in FILTER_KEYS:
        st.session_state[key] = "All"


def apply_filters(clients: pd.DataFrame) -> pd.DataFrame:
    """Render global sidebar controls and return the chosen client subset."""
    with st.sidebar:
        st.html(
            """
            <div class="industrial-eyebrow"><span class="status-led"></span>Filter control bay</div>
            <p style="margin:0 0 1.35rem;color:#4a5568;line-height:1.55;">Tune the buyer signal without changing the trained model.</p>
            """
        )
        country = st.selectbox(
            "Country",
            ["All"] + sorted(clients["country"].dropna().unique().tolist()),
            key="filter_country",
        )
        region_source = clients if country == "All" else clients[clients["country"].eq(country)]
        region = st.selectbox(
            "Region",
            ["All"] + sorted(region_source["region"].dropna().unique().tolist()),
            key="filter_region",
        )
        purpose = st.selectbox(
            "Acquisition purpose",
            ["All"] + sorted(clients["acquisition_purpose"].dropna().unique().tolist()),
            key="filter_purpose",
        )
        client_type = st.selectbox(
            "Client type",
            ["All"] + sorted(clients["client_type"].dropna().unique().tolist()),
            key="filter_client_type",
        )
        st.button(
            "Reset controls",
            icon=":material/restart_alt:",
            key="reset_filters",
            on_click=reset_filters,
            width="stretch",
        )
        st.caption("Filter changes update every instrument panel.")

    filtered = clients.copy()
    if country != "All":
        filtered = filtered[filtered["country"].eq(country)]
    if region != "All":
        filtered = filtered[filtered["region"].eq(region)]
    if purpose != "All":
        filtered = filtered[filtered["acquisition_purpose"].eq(purpose)]
    if client_type != "All":
        filtered = filtered[filtered["client_type"].eq(client_type)]
    return filtered


def percentage(value: float) -> str:
    """Format decimal rates consistently across all dashboard surfaces."""
    return f"{value:.1%}" if pd.notna(value) else "—"


def style_chart(figure, *, height: int = 360):
    """Apply the shared industrial chart grammar without modifying the data layer."""
    figure.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f0f2f5",
        colorway=CHART_COLORS,
        font={"family": "Inter, sans-serif", "color": "#2d3436", "size": 13},
        margin={"l": 14, "r": 14, "t": 18, "b": 12},
        legend={"title": None, "orientation": "h", "y": -0.18, "x": 0},
        hoverlabel={"bgcolor": "#2d3436", "font": {"family": "JetBrains Mono, monospace", "color": "#ffffff"}},
    )
    figure.update_xaxes(showgrid=False, zeroline=False, linecolor="#babecc", tickfont={"size": 11})
    figure.update_yaxes(gridcolor="rgba(163,177,198,0.32)", zeroline=False, tickfont={"size": 11})
    return figure


def render_chart(figure, *, height: int = 360) -> None:
    """Render a chart with a consistent, unobtrusive Plotly control bar."""
    st.plotly_chart(
        style_chart(figure, height=height),
        width="stretch",
        config={"displaylogo": False, "responsive": True},
    )


def render_hero(metadata: dict) -> None:
    """Render the dashboard's CSS-built industrial instrument header."""
    st.html(
        f"""
        <section class="industrial-hero" aria-label="Real Estate Market Intelligence">
          <div class="industrial-copy">
            <p class="industrial-eyebrow"><span class="status-led"></span>System operational · data as of {metadata['reporting_date']}</p>
            <h1>Real Estate Market <span>Intelligence</span></h1>
            <p>Inspect customer segments through property behaviour, investment purpose, financing, and geography—mounted on one clear, decision-ready workspace.</p>
            <div class="signal-rack" aria-label="Dataset signal summary">
              <div class="signal-block"><strong>{metadata['clients']:,}</strong><span class="signal-label">Client signals</span></div>
              <div class="signal-block"><strong>{metadata['selected_clusters']}</strong><span class="signal-label">Active segments</span></div>
              <div class="signal-block"><strong>{metadata['kmeans_silhouette_score']:.3f}</strong><span class="signal-label">Model cohesion</span></div>
            </div>
          </div>
          <div class="control-device" aria-hidden="true">
            <div class="device-topline"><span class="device-label"><span class="status-led"></span>PARCL / BI-04</span><span class="device-vents"><i></i><i></i><i></i></span></div>
            <div class="device-screen">
              <p class="screen-heading">SEGMENT SIGNAL</p>
              <div class="screen-line"><span>PREMIUM PROPERTY</span><span>35.4%</span></div><div class="screen-bar"><span style="width:72%"></span></div>
              <div class="screen-line"><span>APARTMENT FOCUS</span><span>34.1%</span></div><div class="screen-bar"><span style="width:69%"></span></div>
              <div class="screen-line"><span>MIXED USE</span><span>28.1%</span></div><div class="screen-bar"><span style="width:57%"></span></div>
              <div class="screen-line"><span>PORTFOLIO SCALE</span><span>2.5%</span></div><div class="screen-bar"><span style="width:18%"></span></div>
            </div>
            <div class="device-controls"><span class="device-button"></span><span class="device-button"></span><span class="device-button accent"></span></div>
          </div>
        </section>
        """
    )


def render_overview(filtered: pd.DataFrame, metadata: dict) -> None:
    """Show the segment mix as two complementary instrument panels."""
    counts = filtered.groupby(["cluster_id", "segment"], as_index=False).size().sort_values("cluster_id")
    counts = counts.rename(columns={"size": "buyers"})
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True, key="overview_distribution_panel"):
            st.subheader("Segment distribution", icon=":material/stacked_bar_chart:")
            st.caption("Client count by current filter selection")
            figure = px.bar(
                counts,
                x="segment",
                y="buyers",
                color="segment",
                text="buyers",
                labels={"segment": "Segment", "buyers": "Clients"},
            )
            figure.update_layout(showlegend=False)
            figure.update_xaxes(tickangle=-18)
            render_chart(figure)
    with right:
        with st.container(border=True, key="overview_share_panel"):
            st.subheader("Signal share", icon=":material/donut_small:")
            st.caption("Relative weight of every active segment")
            figure = px.pie(
                counts,
                values="buyers",
                names="segment",
                hole=0.66,
                color_discrete_sequence=CHART_COLORS,
            )
            figure.update_traces(textinfo="percent", textfont={"family": "JetBrains Mono, monospace"})
            render_chart(figure)
    st.caption(
        f"Reporting date {metadata['reporting_date']} · "
        f"cluster selection {metadata['selection_method']} · "
        f"K-means silhouette {metadata['kmeans_silhouette_score']:.3f}"
    )


def render_investor_behaviour(filtered: pd.DataFrame) -> None:
    """Compare purpose, financing, and portfolio scale without changing the cluster model."""
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True, key="investment_purpose_panel"):
            st.subheader("Acquisition mode", icon=":material/home_work:")
            st.caption("Home and investment intent by segment")
            purpose = (
                filtered.groupby(["segment", "acquisition_purpose"], as_index=False)
                .size()
                .rename(columns={"size": "buyers"})
            )
            chart = px.bar(
                purpose,
                x="segment",
                y="buyers",
                color="acquisition_purpose",
                barmode="stack",
                labels={"segment": "Segment", "buyers": "Clients", "acquisition_purpose": "Purpose"},
            )
            chart.update_xaxes(tickangle=-18)
            render_chart(chart)
    with right:
        with st.container(border=True, key="financing_panel"):
            st.subheader("Financing signal", icon=":material/account_balance:")
            st.caption("Loan application behaviour by segment")
            loan = (
                filtered.groupby(["segment", "loan_applied"], as_index=False)
                .size()
                .rename(columns={"size": "buyers"})
            )
            chart = px.bar(
                loan,
                x="segment",
                y="buyers",
                color="loan_applied",
                barmode="stack",
                labels={"segment": "Segment", "buyers": "Clients", "loan_applied": "Loan applied"},
            )
            chart.update_xaxes(tickangle=-18)
            render_chart(chart)

    with st.container(border=True, key="portfolio_scale_panel"):
        st.subheader("Portfolio scale", icon=":material/bubble_chart:")
        st.caption("Average completed-purchase count against average total spend")
        spend = filtered.groupby("segment", as_index=False).agg(
            average_total_spend=("total_spend", "mean"),
            average_purchase_count=("purchase_count", "mean"),
        )
        chart = px.scatter(
            spend,
            x="average_purchase_count",
            y="average_total_spend",
            size="average_total_spend",
            color="segment",
            hover_name="segment",
            labels={
                "average_purchase_count": "Average completed purchases",
                "average_total_spend": "Average total spend (USD)",
            },
            size_max=56,
        )
        render_chart(chart, height=400)


def render_geography(filtered: pd.DataFrame) -> None:
    """Map the segment signal by country and regional distribution."""
    country_summary = (
        filtered.groupby("country", as_index=False)
        .agg(clients=("client_id", "nunique"), dominant_segment=("segment", lambda values: values.mode().iat[0]))
    )
    country_summary["iso3"] = country_summary["country"].map(COUNTRY_ISO3)
    valid_map_data = country_summary.dropna(subset=["iso3"])
    with st.container(border=True, key="geography_map_panel"):
        st.subheader("Global signal map", icon=":material/public:")
        st.caption("Dominant segment in each represented country")
        if valid_map_data.empty:
            st.caption("No country-level signal is available for the current filters.")
        else:
            map_figure = px.choropleth(
                valid_map_data,
                locations="iso3",
                color="dominant_segment",
                hover_name="country",
                hover_data={"clients": True, "iso3": False},
                projection="natural earth",
                labels={"dominant_segment": "Dominant segment"},
                color_discrete_sequence=CHART_COLORS,
            )
            map_figure.update_geos(
                showcoastlines=True,
                coastlinecolor="#a3b1c6",
                showframe=False,
                bgcolor="rgba(0,0,0,0)",
                landcolor="#d1d9e6",
            )
            render_chart(map_figure, height=420)

    with st.container(border=True, key="geography_region_panel"):
        st.subheader("Regional mounting", icon=":material/map:")
        st.caption("Buyer-segment distribution across registered regions")
        by_region = (
            filtered.groupby(["region", "segment"], as_index=False)
            .size()
            .rename(columns={"size": "clients"})
        )
        regional_chart = px.bar(
            by_region,
            x="region",
            y="clients",
            color="segment",
            barmode="stack",
            labels={"region": "Region", "clients": "Clients", "segment": "Segment"},
        )
        regional_chart.update_xaxes(tickangle=-32)
        render_chart(regional_chart, height=430)


def render_segment_insights(filtered: pd.DataFrame, profiles: pd.DataFrame) -> None:
    """Show filter-aware segment measurements with expandable full population context."""
    filtered_profile = (
        filtered.groupby(["cluster_id", "segment"], as_index=False)
        .agg(
            clients=("client_id", "nunique"),
            average_age=("age", "mean"),
            investment_rate=("is_investment", "mean"),
            loan_rate=("is_loan", "mean"),
            average_total_spend=("total_spend", "mean"),
            average_purchase_count=("purchase_count", "mean"),
            average_satisfaction=("satisfaction_score", "mean"),
            office_purchase_share=("office_purchase_share", "mean"),
        )
        .sort_values("cluster_id")
    )
    display = filtered_profile.copy()
    display["investment_rate"] = display["investment_rate"].map(percentage)
    display["loan_rate"] = display["loan_rate"].map(percentage)
    display["office_purchase_share"] = display["office_purchase_share"].map(percentage)
    display["average_total_spend"] = display["average_total_spend"].map(lambda value: f"${value:,.0f}")
    for column in ["average_age", "average_purchase_count", "average_satisfaction"]:
        display[column] = display[column].round(2)

    with st.container(border=True, key="segment_table_panel"):
        st.subheader("Segment readout", icon=":material/monitoring:")
        st.caption("Measurements update to the current filter selection")
        st.dataframe(display, width="stretch", hide_index=True)

    with st.expander("Full-population profiles and recommended actions", icon=":material/fact_check:"):
        full_profile = profiles.copy()
        for column in [
            "investment_rate",
            "loan_rate",
            "company_rate",
            "international_rate",
            "average_office_purchase_share",
        ]:
            full_profile[column] = full_profile[column].map(percentage)
        for column in ["average_total_spend", "average_purchase_price"]:
            full_profile[column] = full_profile[column].map(lambda value: f"${value:,.0f}")
        st.dataframe(full_profile, width="stretch", hide_index=True)
        st.caption("See reports/research_paper.md for the EDA, methodology, recommendations, and interpretation cautions.")


def render_data_quality(metadata: dict, scores: pd.DataFrame) -> None:
    """Expose source coverage and selection diagnostics as a transparent model panel."""
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True, key="quality_coverage_panel"):
            st.subheader("Source coverage", icon=":material/database:")
            st.caption("Record flow into the current buyer model")
            st.metric("Clients modelled", f"{metadata['clients']:,}", border=True)
            st.metric("Completed purchases", f"{metadata['sold_properties']:,}", border=True)
            st.metric("Available listings excluded", f"{metadata['available_listings']:,}", border=True)
            st.caption("Gender remains descriptive-only and is excluded from the model features.")
    with right:
        with st.container(border=True, key="quality_diagnostics_panel"):
            st.subheader("K-selection diagnostics", icon=":material/show_chart:")
            st.caption("Silhouette score across evaluated cluster counts")
            chart = px.line(
                scores,
                x="cluster_count",
                y="silhouette_score",
                markers=True,
                labels={"cluster_count": "Number of clusters (K)", "silhouette_score": "Silhouette score"},
            )
            chart.update_traces(line={"width": 3}, marker={"size": 9})
            render_chart(chart, height=370)


def main() -> None:
    """Compose the Streamlit dashboard from native, reusable visual modules."""
    st.html(STYLESHEET)
    try:
        clients, profiles, scores, metadata = load_outputs()
    except FileNotFoundError as error:
        st.error(str(error), icon=":material/error:")
        st.stop()

    render_hero(metadata)
    filtered = apply_filters(clients)
    if filtered.empty:
        st.warning(
            "No clients match the selected filters. Adjust one or more controls to continue.",
            icon=":material/filter_alt_off:",
        )
        st.stop()

    st.html('<p class="industrial-eyebrow"><span class="status-led"></span>Live buyer signal</p>')
    metrics = st.columns(4, gap="medium", vertical_alignment="center")
    with metrics[0]:
        st.metric("Clients in view", f"{filtered['client_id'].nunique():,}", border=True)
    with metrics[1]:
        st.metric("Investment buyers", percentage(filtered["is_investment"].mean()), border=True)
    with metrics[2]:
        st.metric("Loan applicants", percentage(filtered["is_loan"].mean()), border=True)
    with metrics[3]:
        st.metric("Average total spend", f"${filtered['total_spend'].mean():,.0f}", border=True)

    st.space("medium")
    overview, investment, geography, insights, quality = st.tabs(
        [
            "01 Overview",
            "02 Investor behaviour",
            "03 Geographic analysis",
            "04 Segment insights",
            "05 Model quality",
        ]
    )
    with overview:
        render_overview(filtered, metadata)
    with investment:
        render_investor_behaviour(filtered)
    with geography:
        render_geography(filtered)
    with insights:
        render_segment_insights(filtered, profiles)
    with quality:
        render_data_quality(metadata, scores)


if __name__ == "__main__":
    main()
