"""Statistical anomaly detection for DSN telemetry signals."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BASELINE_PATH = Path("models/baselines.json")
MIN_HISTORY_POINTS = 10
Z_SCORE_THRESHOLD = 3.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


class AnomalyDetector:
    """Detect anomalies using per-spacecraft z-score baselines."""

    def __init__(self, baseline_path: str | Path = BASELINE_PATH) -> None:
        """Initialize detector state from JSON baseline file.

        Args:
            baseline_path: Path to persisted spacecraft baseline stats.

        Returns:
            None.
        """
        self.baseline_path = Path(baseline_path)
        self.baselines: dict[str, dict[str, float | int]] = self._load_baselines()

    def _load_baselines(self) -> dict[str, dict[str, float | int]]:
        """Load baseline statistics from disk if available.

        Args:
            None.

        Returns:
            dict[str, dict[str, float | int]]: Baseline mapping by spacecraft.
        """
        if not self.baseline_path.exists():
            return {}

        try:
            with self.baseline_path.open("r", encoding="utf-8") as file_obj:
                loaded = json.load(file_obj)
            if isinstance(loaded, dict):
                return loaded
            LOGGER.warning("Baselines file has invalid format, starting fresh")
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("Could not load baselines from %s: %s", self.baseline_path, exc)
        return {}

    def _save_baselines(self) -> None:
        """Persist baseline statistics to JSON on disk.

        Args:
            None.

        Returns:
            None.
        """
        self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with self.baseline_path.open("w", encoding="utf-8") as file_obj:
            json.dump(self.baselines, file_obj, indent=2)

    @staticmethod
    def _combine_group_stats(
        old_mean: float,
        old_std: float,
        old_n: int,
        new_mean: float,
        new_std: float,
        new_n: int,
    ) -> tuple[float, float, int]:
        """Combine two population-stat groups into one aggregate mean/std.

        Args:
            old_mean: Previous aggregate mean.
            old_std: Previous aggregate population std.
            old_n: Previous aggregate count.
            new_mean: New batch mean.
            new_std: New batch population std.
            new_n: New batch count.

        Returns:
            tuple[float, float, int]: Combined mean, std, and count.
        """
        if old_n <= 0:
            return new_mean, new_std, new_n
        if new_n <= 0:
            return old_mean, old_std, old_n

        total_n = old_n + new_n
        combined_mean = ((old_mean * old_n) + (new_mean * new_n)) / total_n

        old_var = old_std**2
        new_var = new_std**2

        # Parallel variance merge with correction for mean offsets between groups.
        ss_old = old_n * (old_var + (old_mean - combined_mean) ** 2)
        ss_new = new_n * (new_var + (new_mean - combined_mean) ** 2)
        combined_var = (ss_old + ss_new) / total_n

        return float(combined_mean), float(np.sqrt(max(combined_var, 0.0))), int(total_n)

    def update_baseline(self, df: pd.DataFrame) -> None:
        """Update per-spacecraft rolling mean/std baselines from provided history.

        Args:
            df: DataFrame containing at least spacecraft_name and rate_log columns.

        Returns:
            None.
        """
        if df is None or df.empty:
            LOGGER.info("No rows provided for baseline update")
            self._save_baselines()
            return

        required = {"spacecraft_name", "rate_log"}
        if not required.issubset(df.columns):
            missing = sorted(required.difference(df.columns))
            LOGGER.warning("Missing columns for baseline update: %s", ", ".join(missing))
            self._save_baselines()
            return

        working = df[["spacecraft_name", "rate_log"]].copy()
        working["spacecraft_name"] = working["spacecraft_name"].fillna("UNKNOWN").astype(str)
        working["rate_log"] = pd.to_numeric(working["rate_log"], errors="coerce").fillna(0.0)

        grouped = working.groupby("spacecraft_name")["rate_log"]
        means = grouped.mean()
        stds = grouped.std(ddof=0).fillna(0.0)
        counts = grouped.count()

        for spacecraft_name in means.index:
            key = str(spacecraft_name)
            new_mean = float(means.loc[spacecraft_name])
            new_std = float(stds.loc[spacecraft_name])
            new_n = int(counts.loc[spacecraft_name])

            prior = self.baselines.get(key, {})
            old_mean = float(prior.get("mean", 0.0) or 0.0)
            old_std = float(prior.get("std", 0.0) or 0.0)
            old_n = int(prior.get("n", 0) or 0)

            merged_mean, merged_std, merged_n = self._combine_group_stats(
                old_mean=old_mean,
                old_std=old_std,
                old_n=old_n,
                new_mean=new_mean,
                new_std=new_std,
                new_n=new_n,
            )

            self.baselines[key] = {
                "mean": merged_mean,
                "std": merged_std,
                "n": merged_n,
            }

        self._save_baselines()

    def _row_score(self, row: pd.Series) -> float:
        """Compute z-score for one telemetry row using loaded baselines.

        Args:
            row: DataFrame row with spacecraft_name and rate_log.

        Returns:
            float: z-score value.
        """
        spacecraft = str(row.get("spacecraft_name", "UNKNOWN"))
        stats: dict[str, Any] = self.baselines.get(spacecraft, {})

        n_obs = int(stats.get("n", 0) or 0)
        if n_obs < MIN_HISTORY_POINTS:
            return 0.0

        baseline_std = float(stats.get("std", 0.0) or 0.0)
        if baseline_std <= 0.0:
            return 0.0

        baseline_mean = float(stats.get("mean", 0.0) or 0.0)
        rate_log = float(row.get("rate_log", 0.0) or 0.0)
        return (rate_log - baseline_mean) / baseline_std

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add anomaly scores and labels to incoming DSN records.

        Args:
            df: Feature DataFrame with spacecraft_name and rate_log.

        Returns:
            pd.DataFrame: DataFrame with z_score, is_anomaly, and anomaly_label.
        """
        if df is None or df.empty:
            LOGGER.info("No rows provided for anomaly scoring")
            empty = df.copy() if df is not None else pd.DataFrame()
            empty["z_score"] = pd.Series(dtype=float)
            empty["is_anomaly"] = pd.Series(dtype=bool)
            empty["anomaly_label"] = pd.Series(dtype=str)
            return empty

        scored = df.copy()
        if "spacecraft_name" not in scored.columns:
            scored["spacecraft_name"] = "UNKNOWN"
        if "rate_log" not in scored.columns:
            scored["rate_log"] = 0.0

        scored["rate_log"] = pd.to_numeric(scored["rate_log"], errors="coerce").fillna(0.0)
        scored["z_score"] = scored.apply(self._row_score, axis=1).astype(float)
        scored["is_anomaly"] = scored["z_score"].abs() > Z_SCORE_THRESHOLD

        conditions = [scored["z_score"] > Z_SCORE_THRESHOLD, scored["z_score"] < -Z_SCORE_THRESHOLD]
        choices = ["HIGH", "LOW"]
        scored["anomaly_label"] = np.select(conditions, choices, default="NORMAL")

        return scored
