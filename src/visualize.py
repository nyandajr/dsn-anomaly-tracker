"""Visualization utilities for DSN anomaly monitoring reports."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_PATH = Path("outputs/signal_report.png")
LOOKBACK_ROWS = 144
ANOMALY_COLOR = "red"
NORMAL_COLOR = "steelblue"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def generate_report(history_df: pd.DataFrame, output_path: str | Path = OUTPUT_PATH) -> None:
    """Generate and save a two-panel DSN signal monitoring report.

    Args:
        history_df: Full historical DSN DataFrame.
        output_path: Destination PNG path for the rendered report.

    Returns:
        None.
    """
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("dark_background")

    if history_df is None or history_df.empty:
        LOGGER.warning("History DataFrame is empty; generating placeholder report")
        figure, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 8), sharex=True)
        axes[0].text(0.5, 0.5, "No DSN history available", ha="center", va="center")
        axes[1].text(0.5, 0.5, "No anomaly points available", ha="center", va="center")
        for axis in axes:
            axis.set_axis_off()
        title_date = datetime.now(timezone.utc).date().isoformat()
        figure.suptitle(f"DSN Signal Report — {title_date}")
        figure.tight_layout()
        figure.savefig(target_path, dpi=150)
        plt.close(figure)
        return

    plot_df = history_df.copy().tail(LOOKBACK_ROWS)

    if "timestamp" in plot_df.columns:
        plot_df["timestamp"] = pd.to_datetime(plot_df["timestamp"], errors="coerce", utc=True)
    else:
        plot_df["timestamp"] = pd.NaT

    plot_df = plot_df.sort_values("timestamp", kind="stable")

    if "spacecraft_name" not in plot_df.columns:
        plot_df["spacecraft_name"] = "UNKNOWN"
    if "downlink_rate" not in plot_df.columns:
        plot_df["downlink_rate"] = 0.0
    if "z_score" not in plot_df.columns:
        plot_df["z_score"] = 0.0
    if "is_anomaly" not in plot_df.columns:
        plot_df["is_anomaly"] = False

    plot_df["downlink_rate"] = pd.to_numeric(plot_df["downlink_rate"], errors="coerce").fillna(0.0)
    plot_df["z_score"] = pd.to_numeric(plot_df["z_score"], errors="coerce").fillna(0.0)
    plot_df["is_anomaly"] = plot_df["is_anomaly"].fillna(False).astype(bool)

    figure, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 8), sharex=True)

    for spacecraft, group in plot_df.groupby("spacecraft_name"):
        axes[0].plot(group["timestamp"], group["downlink_rate"], label=str(spacecraft), linewidth=1.5)

    anomaly_colors = plot_df["is_anomaly"].map({True: ANOMALY_COLOR, False: NORMAL_COLOR})
    axes[1].scatter(plot_df["timestamp"], plot_df["z_score"], c=anomaly_colors, s=24, alpha=0.85)

    axes[0].set_ylabel("Downlink Rate (bps)")
    axes[0].set_title("Downlink Rate by Spacecraft")
    axes[0].grid(alpha=0.25)

    axes[1].set_ylabel("Z-Score")
    axes[1].set_xlabel("Timestamp (UTC)")
    axes[1].set_title("Anomaly Scores")
    axes[1].grid(alpha=0.25)

    if plot_df["spacecraft_name"].nunique() <= 12:
        axes[0].legend(loc="upper left", fontsize=8, frameon=False)

    title_date = datetime.now(timezone.utc).date().isoformat()
    figure.suptitle(f"DSN Signal Report — {title_date}")
    figure.tight_layout()
    figure.savefig(target_path, dpi=150)
    plt.close(figure)
