from __future__ import annotations

from io import BytesIO
import re
from pathlib import Path

import numpy as np
import pandas as pd


STANDARD_COLUMN_ORDER = ("date", "open", "high", "low", "close", "adj_close", "volume")


def normalize_column_name(column_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", column_name.strip().lower()).strip("_")


def infer_asset_name(source_name: str) -> str:
    return normalize_column_name(Path(source_name).stem)


def clean_market_frame(frame: pd.DataFrame, asset_name: str) -> pd.DataFrame:
    renamed = {column: normalize_column_name(column) for column in frame.columns}
    cleaned = frame.rename(columns=renamed).copy()

    if "date" not in cleaned.columns:
        raise ValueError(f"{asset_name} is missing a date column.")
    if "close" not in cleaned.columns and "adj_close" not in cleaned.columns:
        raise ValueError(f"{asset_name} is missing a close column.")

    if "close" not in cleaned.columns and "adj_close" in cleaned.columns:
        cleaned["close"] = cleaned["adj_close"]

    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")

    numeric_columns = [column for column in cleaned.columns if column != "date"]
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    if "volume" not in cleaned.columns:
        cleaned["volume"] = np.nan

    available_columns = [column for column in STANDARD_COLUMN_ORDER if column in cleaned.columns]
    cleaned = cleaned[available_columns]
    cleaned = cleaned.dropna(subset=["date", "close"])
    cleaned = cleaned.sort_values("date")
    cleaned = cleaned.drop_duplicates(subset=["date"], keep="last")
    cleaned = cleaned.reset_index(drop=True)

    return cleaned


def load_asset_csv(csv_path: Path) -> pd.DataFrame:
    asset_name = csv_path.stem.lower()
    frame = pd.read_csv(csv_path)
    return clean_market_frame(frame, asset_name=asset_name)


def load_uploaded_csv(upload_name: str, payload: bytes, asset_name: str | None = None) -> tuple[str, pd.DataFrame]:
    resolved_asset_name = asset_name or infer_asset_name(upload_name)
    frame = pd.read_csv(BytesIO(payload))
    return resolved_asset_name, clean_market_frame(frame, asset_name=resolved_asset_name)


def load_asset_directory(directory: Path) -> dict[str, pd.DataFrame]:
    files = sorted(directory.glob("*.csv"))
    return {file.stem.lower(): load_asset_csv(file) for file in files}


def discover_csv_assets(directory: Path) -> list[str]:
    return sorted(file.stem.lower() for file in directory.glob("*.csv"))
