"""Feature engineering utilities for DSN telemetry records."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

NUMERIC_COLUMNS = ["azimuth", "elevation", "downlink_rate", "uplink_rate"]
COMPLEX_ENCODING = {"Goldstone": 0, "Madrid": 1, "Canberra": 2}
DEFAULT_COMPLEX_ENCODING = -1

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def _elevation_to_band(elevation: float) -> str:
    """Convert elevation in degrees to a categorical band.

    Args:
        elevation: Elevation angle in degrees.

    Returns:
        str: Elevation band label LOW, MID, or HIGH.
    """
    if elevation < 15.0:
        return "LOW"
    if elevation <= 45.0:
        return "MID"
    return "HIGH"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready features from DSN telemetry records.

    Args:
        df: Input DataFrame containing DSN records.

    Returns:
        pd.DataFrame: DataFrame with engineered feature columns.
    """
    if df is None or df.empty:
        LOGGER.info("Received empty DataFrame in engineer_features")
        empty = df.copy() if df is not None else pd.DataFrame()
        empty["is_active"] = pd.Series(dtype=bool)
        empty["rate_log"] = pd.Series(dtype=float)
        empty["elevation_band"] = pd.Series(dtype=str)
        empty["complex_id_encoded"] = pd.Series(dtype=int)
        return empty

    features_df = df.copy()

    for column in NUMERIC_COLUMNS:
        if column not in features_df.columns:
            features_df[column] = 0.0
        features_df[column] = pd.to_numeric(features_df[column], errors="coerce").fillna(0.0)

    features_df["is_active"] = features_df["downlink_rate"] > 0.0
    features_df["rate_log"] = np.log1p(features_df["downlink_rate"].clip(lower=0.0))
    features_df["elevation_band"] = features_df["elevation"].apply(_elevation_to_band)
    complex_series = (
        features_df["complex_id"]
        if "complex_id" in features_df.columns
        else pd.Series(["" for _ in range(len(features_df))], index=features_df.index)
    )
    features_df["complex_id_encoded"] = (
        complex_series.map(COMPLEX_ENCODING).fillna(DEFAULT_COMPLEX_ENCODING).astype(int)
    )

    return features_df
