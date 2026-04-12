"""Fetch and parse live DSN XML data into a pandas DataFrame."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
import xml.etree.ElementTree as ET

import pandas as pd
import requests

DSN_XML_URL = "https://eyes.nasa.gov/dsn/data/dsn.xml"
REQUEST_TIMEOUT_SECONDS = 10
DEFAULT_COMPLEX_ID = "UNKNOWN"
OUTPUT_COLUMNS = [
    "antenna_id",
    "spacecraft_name",
    "azimuth",
    "elevation",
    "downlink_rate",
    "uplink_rate",
    "signal_type",
    "timestamp",
    "complex_id",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def _empty_frame() -> pd.DataFrame:
    """Return an empty DataFrame with canonical DSN columns.

    Args:
        None.

    Returns:
        pd.DataFrame: Empty DataFrame with expected columns.
    """
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def _to_float(value: Any) -> float:
    """Convert a value to float with safe fallback.

    Args:
        value: Any scalar value from XML attributes.

    Returns:
        float: Parsed float value or 0.0 on failure.
    """
    try:
        if value in (None, ""):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _normalize_antenna_id(raw_name: str | None) -> str:
    """Normalize DSN dish names to the DSS-## style when possible.

    Args:
        raw_name: Raw `dish` name attribute (for example DSS14).

    Returns:
        str: Normalized antenna id string.
    """
    if not raw_name:
        return "UNKNOWN"

    upper_name = raw_name.upper()
    if upper_name.startswith("DSS") and "-" not in upper_name:
        suffix = upper_name[3:]
        if suffix.isdigit():
            return f"DSS-{suffix}"
    return upper_name


def fetch_dsn_data() -> pd.DataFrame:
    """Fetch DSN XML feed and parse key telemetry rows.

    Args:
        None.

    Returns:
        pd.DataFrame: Parsed DSN records or an empty canonical DataFrame.
    """
    try:
        response = requests.get(DSN_XML_URL, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        xml_text = response.text.strip()
        if not xml_text:
            LOGGER.warning("DSN feed returned empty response body")
            return _empty_frame()
        root = ET.fromstring(xml_text)
    except requests.RequestException as exc:
        LOGGER.warning("Failed to fetch DSN XML feed: %s", exc)
        return _empty_frame()
    except ET.ParseError as exc:
        LOGGER.warning("Failed to parse DSN XML feed: %s", exc)
        return _empty_frame()

    parsed_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    current_complex = DEFAULT_COMPLEX_ID

    for element in list(root):
        if element.tag == "station":
            friendly_name = (element.attrib.get("friendlyName") or "").strip()
            name_attr = (element.attrib.get("name") or "").strip()
            current_complex = friendly_name or name_attr or DEFAULT_COMPLEX_ID
            continue

        if element.tag != "dish":
            continue

        up_signal = element.find("upSignal")
        down_signal = element.find("downSignal")
        target = element.find("target")

        spacecraft_name = "UNKNOWN"
        if down_signal is not None:
            spacecraft_name = down_signal.attrib.get("spacecraft", spacecraft_name)
        if (not spacecraft_name or spacecraft_name == "UNKNOWN") and up_signal is not None:
            spacecraft_name = up_signal.attrib.get("spacecraft", spacecraft_name)
        if (not spacecraft_name or spacecraft_name == "UNKNOWN") and target is not None:
            spacecraft_name = target.attrib.get("name", spacecraft_name)

        signal_type = "none"
        if down_signal is not None:
            signal_type = down_signal.attrib.get("signalType", signal_type)
        elif up_signal is not None:
            signal_type = up_signal.attrib.get("signalType", signal_type)

        downlink_rate = _to_float(down_signal.attrib.get("dataRate") if down_signal is not None else 0.0)
        uplink_rate = _to_float(up_signal.attrib.get("dataRate") if up_signal is not None else 0.0)

        rows.append(
            {
                "antenna_id": _normalize_antenna_id(element.attrib.get("name")),
                "spacecraft_name": (spacecraft_name or "UNKNOWN").strip(),
                "azimuth": _to_float(element.attrib.get("azimuthAngle")),
                "elevation": _to_float(element.attrib.get("elevationAngle")),
                "downlink_rate": downlink_rate,
                "uplink_rate": uplink_rate,
                "signal_type": (signal_type or "none").strip().lower(),
                "timestamp": parsed_at,
                "complex_id": current_complex,
            }
        )

    if not rows:
        LOGGER.warning("DSN XML parsed successfully but no dish records were found")
        return _empty_frame()

    frame = pd.DataFrame(rows)
    return frame.reindex(columns=OUTPUT_COLUMNS)
