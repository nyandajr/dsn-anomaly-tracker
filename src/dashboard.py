"""Streamlit dashboard for DSN operations and anomaly monitoring."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

HISTORY_PATH = Path("data/history.csv")
BASELINE_PATH = Path("models/baselines.json")
REPORT_PATH = Path("outputs/signal_report.png")
PAGE_TITLE = "DSN Operations Console"
RECENT_ROWS = 200
LOOKBACK_OPTIONS = {
    "Last 24 hours": 24,
    "Last 7 days": 24 * 7,
    "Last 30 days": 24 * 30,
    "All available history": None,
}
ANOMALY_THRESHOLD = 3.0
SEVERITY_MEDIUM = 3.0
SEVERITY_HIGH = 5.0
SEVERITY_CRITICAL = 8.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def apply_dashboard_theme() -> None:
    """Inject custom CSS for a denser operations-console layout.

    Args:
        None.

    Returns:
        None.
    """
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

        :root {
            --bg-primary: #07111f;
            --bg-panel: rgba(12, 25, 42, 0.88);
            --bg-panel-strong: rgba(20, 37, 61, 0.96);
            --border-color: rgba(130, 176, 255, 0.18);
            --text-primary: #f3f7ff;
            --text-muted: #8ca4c6;
            --accent-cyan: #4fd1ff;
            --accent-gold: #f4b860;
            --accent-red: #ff6b6b;
            --accent-green: #46d68d;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(79, 209, 255, 0.16), transparent 28%),
                radial-gradient(circle at top right, rgba(244, 184, 96, 0.12), transparent 22%),
                linear-gradient(180deg, #030814 0%, #07111f 100%);
            color: var(--text-primary);
            font-family: 'Space Grotesk', 'Segoe UI', sans-serif;
        }

        [data-testid="stSidebar"] {
            background: rgba(5, 14, 26, 0.96);
            border-right: 1px solid var(--border-color);
        }

        .dsn-shell {
            padding: 0.8rem 0 1.4rem 0;
        }

        .dsn-hero {
            background: linear-gradient(135deg, rgba(11, 29, 52, 0.95), rgba(13, 46, 77, 0.88));
            border: 1px solid var(--border-color);
            border-radius: 22px;
            padding: 1.2rem 1.35rem;
            box-shadow: 0 18px 60px rgba(2, 6, 13, 0.32);
            margin-bottom: 1rem;
        }

        .dsn-hero h1 {
            margin: 0;
            font-size: 2rem;
            letter-spacing: 0.02em;
        }

        .dsn-hero p {
            margin: 0.35rem 0 0 0;
            color: var(--text-muted);
        }

        .dsn-panel {
            background: var(--bg-panel);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 1rem 1rem 0.85rem 1rem;
            box-shadow: 0 16px 50px rgba(2, 6, 13, 0.24);
            margin-bottom: 1rem;
        }

        .dsn-panel h3 {
            margin-top: 0;
        }

        .dsn-kpi {
            background: linear-gradient(180deg, rgba(11, 24, 42, 0.98), rgba(6, 18, 31, 0.96));
            border: 1px solid var(--border-color);
            border-radius: 18px;
            padding: 0.85rem 1rem;
            min-height: 110px;
        }

        .dsn-kpi-label {
            color: var(--text-muted);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .dsn-kpi-value {
            color: var(--text-primary);
            font-size: 1.8rem;
            font-weight: 700;
            margin-top: 0.25rem;
        }

        .dsn-kpi-sub {
            color: var(--text-muted);
            font-size: 0.9rem;
            margin-top: 0.35rem;
        }

        .dsn-complex {
            background: linear-gradient(180deg, rgba(16, 34, 58, 0.98), rgba(9, 20, 35, 0.95));
            border: 1px solid var(--border-color);
            border-radius: 18px;
            padding: 1rem;
            min-height: 180px;
        }

        .dsn-badge {
            display: inline-block;
            padding: 0.18rem 0.52rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        .dsn-badge-ok {
            background: rgba(70, 214, 141, 0.16);
            color: var(--accent-green);
        }

        .dsn-badge-alert {
            background: rgba(255, 107, 107, 0.16);
            color: var(--accent-red);
        }

        .dsn-badge-warn {
            background: rgba(244, 184, 96, 0.18);
            color: var(--accent-gold);
        }

        .dsn-metric-line {
            color: var(--text-muted);
            margin-top: 0.45rem;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False, ttl=120)
def load_history(history_path: str) -> pd.DataFrame:
    """Load DSN history from CSV and coerce dashboard-ready types.

    Args:
        history_path: String path to the historical CSV file.

    Returns:
        pd.DataFrame: Parsed historical telemetry records.
    """
    path = Path(history_path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    frame = pd.read_csv(path)
    if frame.empty:
        return frame

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    numeric_columns = ["azimuth", "elevation", "downlink_rate", "uplink_rate", "rate_log", "z_score"]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

    if "is_active" in frame.columns:
        frame["is_active"] = frame["is_active"].astype(str).str.lower().eq("true")
    if "is_anomaly" in frame.columns:
        frame["is_anomaly"] = frame["is_anomaly"].astype(str).str.lower().eq("true")

    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable")
    return frame


@st.cache_data(show_spinner=False, ttl=120)
def load_baselines(baseline_path: str) -> pd.DataFrame:
    """Load persisted per-spacecraft baseline statistics.

    Args:
        baseline_path: String path to the baselines JSON file.

    Returns:
        pd.DataFrame: Baselines in tabular form.
    """
    path = Path(baseline_path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["spacecraft_name", "mean", "std", "n"])

    with path.open("r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)

    if not data:
        return pd.DataFrame(columns=["spacecraft_name", "mean", "std", "n"])

    baseline_frame = pd.DataFrame.from_dict(data, orient="index").reset_index()
    baseline_frame = baseline_frame.rename(columns={"index": "spacecraft_name"})
    return baseline_frame.sort_values("spacecraft_name", kind="stable")


def apply_filters(
    history_df: pd.DataFrame,
    lookback_hours: int | None,
    complexes: list[str],
    signal_types: list[str],
    selected_spacecraft: str,
    active_only: bool,
    anomalies_only: bool,
) -> pd.DataFrame:
    """Apply sidebar filters to dashboard history.

    Args:
        history_df: Full historical DataFrame.
        lookback_hours: Optional time-window length in hours.
        complexes: Selected complexes.
        signal_types: Selected signal types.
        selected_spacecraft: Selected spacecraft or All spacecraft.
        active_only: Whether to keep active rows only.
        anomalies_only: Whether to keep anomalous rows only.

    Returns:
        pd.DataFrame: Filtered DataFrame for dashboard rendering.
    """
    if history_df.empty:
        return history_df

    filtered = history_df.copy()

    if lookback_hours is not None:
        latest_timestamp = filtered["timestamp"].max()
        window_start = latest_timestamp - pd.Timedelta(hours=lookback_hours)
        filtered = filtered[filtered["timestamp"] >= window_start]

    if complexes:
        filtered = filtered[filtered["complex_id"].isin(complexes)]
    if signal_types:
        filtered = filtered[filtered["signal_type"].isin(signal_types)]
    if selected_spacecraft != "All spacecraft":
        filtered = filtered[filtered["spacecraft_name"] == selected_spacecraft]
    if active_only:
        filtered = filtered[filtered["is_active"]]
    if anomalies_only:
        filtered = filtered[filtered["is_anomaly"]]

    return filtered.sort_values("timestamp", kind="stable")


def latest_snapshot(history_df: pd.DataFrame) -> pd.DataFrame:
    """Extract the most recent timestamp slice from history.

    Args:
        history_df: Historical DataFrame.

    Returns:
        pd.DataFrame: Latest snapshot rows.
    """
    if history_df.empty:
        return history_df
    latest_timestamp = history_df["timestamp"].max()
    return history_df[history_df["timestamp"] == latest_timestamp].copy()


def freshness_status(latest_timestamp: pd.Timestamp | None) -> tuple[str, str, str]:
    """Classify pipeline freshness from latest timestamp.

    Args:
        latest_timestamp: Most recent timestamp in the dataset.

    Returns:
        tuple[str, str, str]: Status label, Streamlit state color, and detail text.
    """
    if latest_timestamp is None:
        return "UNKNOWN", "error", "No historical telemetry available yet."

    stale_minutes = (pd.Timestamp.now(tz="UTC") - latest_timestamp).total_seconds() / 60
    if stale_minutes <= 20:
        return "HEALTHY", "success", f"Data is fresh. Last update {stale_minutes:.1f} minutes ago."
    if stale_minutes <= 60:
        return "DEGRADED", "warning", f"Data is aging. Last update {stale_minutes:.1f} minutes ago."
    return "STALE", "error", f"Data is stale. Last update {stale_minutes:.1f} minutes ago."


def render_pipeline_banner(history_df: pd.DataFrame) -> None:
    """Render top-of-page pipeline freshness banner.

    Args:
        history_df: Full historical telemetry DataFrame.

    Returns:
        None.
    """
    latest_timestamp = history_df["timestamp"].max() if not history_df.empty else None
    label, tone, detail = freshness_status(latest_timestamp)
    message = f"Pipeline status: {label} | {detail}"
    if tone == "success":
        st.success(message)
    elif tone == "warning":
        st.warning(message)
    else:
        st.error(message)


def render_header(snapshot_df: pd.DataFrame, filtered_df: pd.DataFrame) -> None:
    """Render page header and KPI strip.

    Args:
        snapshot_df: Most recent snapshot DataFrame.
        filtered_df: Filtered historical DataFrame.

    Returns:
        None.
    """
    latest_timestamp = snapshot_df["timestamp"].max() if not snapshot_df.empty else None
    last_update = latest_timestamp.strftime("%Y-%m-%d %H:%M:%S UTC") if latest_timestamp is not None else "Unavailable"
    total_downlink_mbps = snapshot_df["downlink_rate"].sum() / 1_000_000 if not snapshot_df.empty else 0.0
    anomaly_count = int(snapshot_df["is_anomaly"].sum()) if not snapshot_df.empty else 0
    active_contacts = int(snapshot_df["is_active"].sum()) if not snapshot_df.empty else 0
    active_spacecraft = int(snapshot_df.loc[snapshot_df["is_active"], "spacecraft_name"].nunique()) if not snapshot_df.empty else 0
    tracked_antennas = int(snapshot_df["antenna_id"].nunique()) if not snapshot_df.empty else 0

    st.markdown("<div class='dsn-shell'>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class='dsn-hero'>
            <h1>{PAGE_TITLE}</h1>
            <p>Real-time view of DSN contacts, anomalies, and pipeline health. Last update: {last_update}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    columns = st.columns(5)
    kpis = [
        ("Active Contacts", active_contacts, f"{tracked_antennas} antennas in latest snapshot"),
        ("Active Spacecraft", active_spacecraft, f"{snapshot_df['spacecraft_name'].nunique() if not snapshot_df.empty else 0} total visible craft"),
        ("Current Anomalies", anomaly_count, f"Threshold: abs(z) > {ANOMALY_THRESHOLD:.1f}"),
        ("Network Downlink", f"{total_downlink_mbps:,.2f} Mbps", "Aggregate current throughput"),
        ("Filtered Rows", len(filtered_df), "Rows after dashboard filters"),
    ]
    for column, (label, value, subtext) in zip(columns, kpis, strict=True):
        column.markdown(
            f"""
            <div class='dsn-kpi'>
                <div class='dsn-kpi-label'>{label}</div>
                <div class='dsn-kpi-value'>{value}</div>
                <div class='dsn-kpi-sub'>{subtext}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_complex_status(snapshot_df: pd.DataFrame) -> None:
    """Render complex-level status cards.

    Args:
        snapshot_df: Most recent snapshot DataFrame.

    Returns:
        None.
    """
    st.markdown("<div class='dsn-panel'><h3>Complex Status</h3></div>", unsafe_allow_html=True)
    if snapshot_df.empty:
        st.info("No current snapshot available for complex status.")
        return

    complexes = ["Goldstone", "Madrid", "Canberra"]
    columns = st.columns(3)
    for column, complex_name in zip(columns, complexes, strict=True):
        subset = snapshot_df[snapshot_df["complex_id"] == complex_name]
        anomaly_count = int(subset["is_anomaly"].sum()) if not subset.empty else 0
        active_count = int(subset["is_active"].sum()) if not subset.empty else 0
        status_class = "dsn-badge-ok" if anomaly_count == 0 else "dsn-badge-alert"
        status_text = "stable" if anomaly_count == 0 else "attention"
        downlink_mbps = subset["downlink_rate"].sum() / 1_000_000 if not subset.empty else 0.0
        spacecraft_count = subset["spacecraft_name"].nunique() if not subset.empty else 0
        antenna_count = subset["antenna_id"].nunique() if not subset.empty else 0
        column.markdown(
            f"""
            <div class='dsn-complex'>
                <div class='dsn-badge {status_class}'>{status_text}</div>
                <h3>{complex_name}</h3>
                <div class='dsn-metric-line'>Active contacts: <strong>{active_count}</strong></div>
                <div class='dsn-metric-line'>Visible spacecraft: <strong>{spacecraft_count}</strong></div>
                <div class='dsn-metric-line'>Antennas online: <strong>{antenna_count}</strong></div>
                <div class='dsn-metric-line'>Downlink throughput: <strong>{downlink_mbps:,.2f} Mbps</strong></div>
                <div class='dsn-metric-line'>Anomalies in latest run: <strong>{anomaly_count}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_signal_trends(filtered_df: pd.DataFrame) -> None:
    """Render downlink and anomaly trend charts.

    Args:
        filtered_df: Filtered historical DataFrame.

    Returns:
        None.
    """
    st.markdown("<div class='dsn-panel'><h3>Signal Intelligence</h3></div>", unsafe_allow_html=True)
    if filtered_df.empty:
        st.info("No data available for the selected filters.")
        return

    line_df = filtered_df.tail(RECENT_ROWS)
    trend_columns = st.columns(2)

    downlink_fig = px.line(
        line_df,
        x="timestamp",
        y="downlink_rate",
        color="spacecraft_name",
        line_group="spacecraft_name",
        template="plotly_dark",
        title="Downlink Rate by Spacecraft",
    )
    downlink_fig.update_layout(legend_title_text="Spacecraft", margin=dict(l=10, r=10, t=48, b=10), height=380)
    trend_columns[0].plotly_chart(downlink_fig, use_container_width=True)

    zscore_fig = px.scatter(
        line_df,
        x="timestamp",
        y="z_score",
        color="anomaly_label",
        symbol="complex_id",
        hover_data=["spacecraft_name", "antenna_id", "downlink_rate"],
        color_discrete_map={"NORMAL": "#4fd1ff", "HIGH": "#ff6b6b", "LOW": "#f4b860"},
        template="plotly_dark",
        title="Anomaly Score Timeline",
    )
    zscore_fig.add_hline(y=ANOMALY_THRESHOLD, line_dash="dash", line_color="#ff6b6b")
    zscore_fig.add_hline(y=-ANOMALY_THRESHOLD, line_dash="dash", line_color="#f4b860")
    zscore_fig.update_layout(margin=dict(l=10, r=10, t=48, b=10), height=380)
    trend_columns[1].plotly_chart(zscore_fig, use_container_width=True)


def render_anomaly_center(snapshot_df: pd.DataFrame, filtered_df: pd.DataFrame) -> None:
    """Render latest anomalies and repeat offender summaries.

    Args:
        snapshot_df: Most recent snapshot DataFrame.
        filtered_df: Filtered historical DataFrame.

    Returns:
        None.
    """
    st.markdown("<div class='dsn-panel'><h3>Anomaly Center</h3></div>", unsafe_allow_html=True)
    if not snapshot_df.empty:
        abs_z = snapshot_df["z_score"].abs() if "z_score" in snapshot_df.columns else pd.Series(dtype=float)
        medium_count = int(((abs_z > SEVERITY_MEDIUM) & (abs_z <= SEVERITY_HIGH)).sum())
        high_count = int(((abs_z > SEVERITY_HIGH) & (abs_z <= SEVERITY_CRITICAL)).sum())
        critical_count = int((abs_z > SEVERITY_CRITICAL).sum())
        sev_cols = st.columns(3)
        sev_cols[0].metric("Medium severity (|z| 3-5)", medium_count)
        sev_cols[1].metric("High severity (|z| 5-8)", high_count)
        sev_cols[2].metric("Critical severity (|z| > 8)", critical_count)

    left_col, right_col = st.columns([1.4, 1])

    anomaly_df = snapshot_df[snapshot_df["is_anomaly"]].copy() if not snapshot_df.empty else pd.DataFrame()
    if anomaly_df.empty:
        left_col.success("No anomalies flagged in the latest run.")
    else:
        anomaly_df["abs_z_score"] = anomaly_df["z_score"].abs()
        anomaly_df = anomaly_df.sort_values("abs_z_score", ascending=False)
        left_col.dataframe(
            anomaly_df[
                [
                    "spacecraft_name",
                    "antenna_id",
                    "complex_id",
                    "downlink_rate",
                    "z_score",
                    "anomaly_label",
                    "timestamp",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    if filtered_df.empty:
        right_col.info("Not enough filtered history to compute anomaly trends.")
        return

    recurring = (
        filtered_df[filtered_df["is_anomaly"]]
        .groupby("spacecraft_name")
        .size()
        .reset_index(name="anomaly_count")
        .sort_values("anomaly_count", ascending=False)
        .head(8)
    )
    if recurring.empty:
        right_col.info("No anomalous spacecraft in the selected window.")
    else:
        repeat_fig = px.bar(
            recurring,
            x="anomaly_count",
            y="spacecraft_name",
            orientation="h",
            template="plotly_dark",
            color="anomaly_count",
            color_continuous_scale="Sunset",
            title="Most Recurrent Anomalies",
        )
        repeat_fig.update_layout(margin=dict(l=10, r=10, t=48, b=10), coloraxis_showscale=False, height=340)
        right_col.plotly_chart(repeat_fig, use_container_width=True)


def render_spacecraft_explorer(filtered_df: pd.DataFrame, baseline_df: pd.DataFrame) -> None:
    """Render spacecraft-specific drill-down charts and baseline comparison.

    Args:
        filtered_df: Filtered historical DataFrame.
        baseline_df: Baselines DataFrame.

    Returns:
        None.
    """
    st.markdown("<div class='dsn-panel'><h3>Spacecraft Explorer</h3></div>", unsafe_allow_html=True)
    if filtered_df.empty:
        st.info("No spacecraft history matches the current filters.")
        return

    spacecraft_options = sorted(filtered_df["spacecraft_name"].dropna().unique().tolist())
    default_craft = spacecraft_options[0] if spacecraft_options else None
    selected_craft = st.selectbox("Inspect spacecraft", spacecraft_options, index=0 if default_craft else None)
    craft_df = filtered_df[filtered_df["spacecraft_name"] == selected_craft].tail(RECENT_ROWS)

    baseline_row = baseline_df[baseline_df["spacecraft_name"] == selected_craft]
    baseline_mean = float(baseline_row["mean"].iloc[0]) if not baseline_row.empty else 0.0
    baseline_std = float(baseline_row["std"].iloc[0]) if not baseline_row.empty else 0.0
    baseline_n = int(baseline_row["n"].iloc[0]) if not baseline_row.empty else 0
    current_rate_log = float(craft_df["rate_log"].iloc[-1]) if not craft_df.empty else 0.0

    explorer_cols = st.columns([1.3, 1])
    craft_trend = px.line(
        craft_df,
        x="timestamp",
        y=["downlink_rate", "uplink_rate"],
        template="plotly_dark",
        title=f"{selected_craft} Signal History",
    )
    craft_trend.update_layout(margin=dict(l=10, r=10, t=48, b=10), height=380, legend_title_text="Metric")
    explorer_cols[0].plotly_chart(craft_trend, use_container_width=True)

    baseline_fig = go.Figure()
    baseline_fig.add_trace(
        go.Bar(
            x=["Baseline mean", "Current rate_log"],
            y=[baseline_mean, current_rate_log],
            marker_color=["#4fd1ff", "#f4b860"],
        )
    )
    if baseline_std > 0:
        upper_bound = baseline_mean + (ANOMALY_THRESHOLD * baseline_std)
        lower_bound = max(baseline_mean - (ANOMALY_THRESHOLD * baseline_std), 0.0)
        baseline_fig.add_hrect(y0=lower_bound, y1=upper_bound, fillcolor="rgba(79, 209, 255, 0.16)", line_width=0)
    baseline_fig.update_layout(
        template="plotly_dark",
        title=f"{selected_craft} Baseline Context",
        margin=dict(l=10, r=10, t=48, b=10),
        height=380,
    )
    explorer_cols[1].plotly_chart(baseline_fig, use_container_width=True)

    st.caption(
        f"Baseline samples: {baseline_n} | mean(rate_log): {baseline_mean:.3f} | std(rate_log): {baseline_std:.3f}"
    )


def render_contact_table(snapshot_df: pd.DataFrame) -> None:
    """Render latest contact table for operators.

    Args:
        snapshot_df: Most recent snapshot DataFrame.

    Returns:
        None.
    """
    st.markdown("<div class='dsn-panel'><h3>Live Contact Table</h3></div>", unsafe_allow_html=True)
    if snapshot_df.empty:
        st.info("No current snapshot available.")
        return

    display_columns = [
        "spacecraft_name",
        "antenna_id",
        "complex_id",
        "signal_type",
        "elevation",
        "downlink_rate",
        "uplink_rate",
        "z_score",
        "anomaly_label",
        "is_active",
        "timestamp",
    ]
    sort_columns = [column for column in ["is_anomaly", "downlink_rate"] if column in snapshot_df.columns]
    if sort_columns:
        ascending = [False] * len(sort_columns)
        sorted_snapshot = snapshot_df.sort_values(sort_columns, ascending=ascending)
    else:
        sorted_snapshot = snapshot_df
    contact_table = sorted_snapshot[display_columns]
    st.dataframe(contact_table, use_container_width=True, hide_index=True)


def render_pipeline_health(history_df: pd.DataFrame, baseline_df: pd.DataFrame) -> None:
    """Render pipeline and data-quality observability metrics.

    Args:
        history_df: Full historical DataFrame.
        baseline_df: Baselines DataFrame.

    Returns:
        None.
    """
    st.markdown("<div class='dsn-panel'><h3>Pipeline Health</h3></div>", unsafe_allow_html=True)
    history_rows = len(history_df)
    run_count = history_df["timestamp"].nunique() if not history_df.empty else 0
    latest_timestamp = history_df["timestamp"].max() if not history_df.empty else None
    latest_update = latest_timestamp.strftime("%Y-%m-%d %H:%M:%S UTC") if latest_timestamp is not None else "Unavailable"
    baseline_count = len(baseline_df)
    output_status = "available" if REPORT_PATH.exists() else "missing"
    freshness, _, _ = freshness_status(latest_timestamp)

    health_cols = st.columns(4)
    health_cols[0].metric("History rows", f"{history_rows:,}")
    health_cols[1].metric("Captured runs", f"{run_count:,}")
    health_cols[2].metric("Spacecraft baselines", baseline_count)
    health_cols[3].metric("Chart artifact", output_status)
    st.caption(f"Latest successful telemetry snapshot: {latest_update} | Feed freshness: {freshness}")

    if REPORT_PATH.exists():
        with st.expander("Latest generated signal report"):
            st.image(str(REPORT_PATH), use_container_width=True)


def sidebar_controls(history_df: pd.DataFrame) -> tuple[int | None, list[str], list[str], str, bool, bool]:
    """Render sidebar controls and return current filter state.

    Args:
        history_df: Full historical DataFrame.

    Returns:
        tuple[int | None, list[str], list[str], str, bool, bool]: Active filter values.
    """
    st.sidebar.title("Operations Filters")
    lookback_label = st.sidebar.selectbox("Time window", list(LOOKBACK_OPTIONS.keys()), index=0)
    lookback_hours = LOOKBACK_OPTIONS[lookback_label]

    complexes = sorted(history_df["complex_id"].dropna().unique().tolist()) if not history_df.empty else []
    selected_complexes = st.sidebar.multiselect("Complexes", complexes, default=complexes)

    signal_types = sorted(history_df["signal_type"].dropna().unique().tolist()) if not history_df.empty else []
    selected_signal_types = st.sidebar.multiselect("Signal types", signal_types, default=signal_types)

    spacecraft_options = ["All spacecraft"]
    if not history_df.empty:
        spacecraft_options.extend(sorted(history_df["spacecraft_name"].dropna().unique().tolist()))
    selected_spacecraft = st.sidebar.selectbox("Spacecraft", spacecraft_options, index=0)

    active_only = st.sidebar.toggle("Active contacts only", value=False)
    anomalies_only = st.sidebar.toggle("Anomalies only", value=False)
    if st.sidebar.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.caption("Data is cached for 120 seconds. Use Refresh data to pull the latest local files immediately.")
    return lookback_hours, selected_complexes, selected_signal_types, selected_spacecraft, active_only, anomalies_only


def main() -> None:
    """Run the Streamlit DSN operations dashboard.

    Args:
        None.

    Returns:
        None.
    """
    st.set_page_config(page_title=PAGE_TITLE, layout="wide", initial_sidebar_state="expanded")
    apply_dashboard_theme()

    history_df = load_history(str(HISTORY_PATH))
    baseline_df = load_baselines(str(BASELINE_PATH))

    if history_df.empty:
        st.error("No DSN history is available yet. Run the pipeline first to populate data/history.csv.")
        return

    controls = sidebar_controls(history_df)
    filtered_df = apply_filters(history_df, *controls)
    snapshot_df = latest_snapshot(filtered_df if not filtered_df.empty else history_df)

    render_pipeline_banner(history_df)
    render_header(snapshot_df, filtered_df)
    render_complex_status(snapshot_df)
    render_signal_trends(filtered_df)
    render_anomaly_center(snapshot_df, filtered_df)
    render_spacecraft_explorer(filtered_df if not filtered_df.empty else history_df, baseline_df)
    render_contact_table(snapshot_df)
    render_pipeline_health(history_df, baseline_df)