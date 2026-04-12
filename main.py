"""Main orchestration entrypoint for DSN anomaly tracking pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.anomaly import AnomalyDetector
from src.fetch import fetch_dsn_data
from src.features import engineer_features
from src.visualize import generate_report

HISTORY_PATH = Path("data/history.csv")
BASELINE_PATH = Path("models/baselines.json")
REPORT_PATH = Path("outputs/signal_report.png")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def _load_history(history_path: Path) -> pd.DataFrame:
    """Load historical DSN records from CSV when available.

    Args:
        history_path: Path to the historical CSV store.

    Returns:
        pd.DataFrame: Existing historical records or an empty DataFrame.
    """
    if not history_path.exists() or history_path.stat().st_size == 0:
        return pd.DataFrame()

    try:
        return pd.read_csv(history_path)
    except (OSError, pd.errors.EmptyDataError) as exc:
        LOGGER.warning("Could not read history file %s: %s", history_path, exc)
        return pd.DataFrame()


def main() -> int:
    """Run the end-to-end DSN anomaly pipeline.

    Args:
        None.

    Returns:
        int: Process exit code.
    """
    raw_df = fetch_dsn_data()

    if raw_df.empty:
        LOGGER.info("No DSN data available")
        return 0

    feat_df = engineer_features(raw_df)

    detector = AnomalyDetector(baseline_path=BASELINE_PATH)
    detector.update_baseline(feat_df)
    scored_df = detector.score(feat_df)

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not HISTORY_PATH.exists()) or HISTORY_PATH.stat().st_size == 0
    scored_df.to_csv(HISTORY_PATH, mode="a", index=False, header=write_header)

    history_df = _load_history(HISTORY_PATH)
    generate_report(history_df, output_path=REPORT_PATH)

    anomaly_count = int(scored_df["is_anomaly"].sum()) if "is_anomaly" in scored_df.columns else 0
    print(f"Processed {len(scored_df)} records; flagged {anomaly_count} anomalies in this run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
