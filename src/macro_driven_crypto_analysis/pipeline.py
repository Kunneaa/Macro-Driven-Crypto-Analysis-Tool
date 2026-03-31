from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .config import PipelineConfig
from .io import discover_csv_assets, load_asset_directory
from .lstm_model import LSTMTrainingResult, train_lstm_model
from .scoring import infer_indicator_spec


@dataclass
class AnalysisResult:
    dataset: pd.DataFrame
    summary: dict
    gap_purge_summary: dict
    lstm_result: LSTMTrainingResult
    core_assets: list[str]
    macro_assets: list[str]
    selected_macro_assets: list[str]
    core_asset: str
    config: PipelineConfig


def discover_assets(project_root: Path) -> tuple[list[str], list[str]]:
    core_assets = discover_csv_assets(project_root / "data" / "core")
    macro_assets = discover_csv_assets(project_root / "data" / "macro")
    return core_assets, macro_assets


def _rename_close_columns(frame: pd.DataFrame, asset_name: str) -> pd.DataFrame:
    renamed = frame.copy()
    column_map = {
        "close": f"{asset_name}_close",
        "open": f"{asset_name}_open",
        "high": f"{asset_name}_high",
        "low": f"{asset_name}_low",
        "volume": f"{asset_name}_volume",
    }
    return renamed.rename(columns={source: target for source, target in column_map.items() if source in renamed.columns})


def _choose_scaler(scaler_name: str):
    if scaler_name == "standard":
        return StandardScaler()
    if scaler_name == "minmax":
        return MinMaxScaler()
    return None


def _scale_to_unit(series: pd.Series, lower: float, upper: float) -> pd.Series:
    clipped = series.clip(lower=lower, upper=upper)
    return (clipped - lower) / (upper - lower)


def _classify_signal(row: pd.Series, drawdown_column: str, trend_column: str, distance_column: str) -> str:
    if row["accumulation_score"] >= 72 and row[trend_column] > 0 and row[drawdown_column] <= -0.20:
        return "High Conviction Accumulation"
    if row["accumulation_score"] >= 60 and row[drawdown_column] <= -0.12:
        return "Layered Accumulation"
    if row["accumulation_score"] <= 35 and row[trend_column] < 0:
        return "Macro Headwind"
    if row["accumulation_score"] < 45 and row[distance_column] > 0.12:
        return "Extended / Chase Risk"
    return "Neutral"


def _assemble_dataset(
    core_asset: str,
    core_frame: pd.DataFrame,
    macro_frames: dict[str, pd.DataFrame],
    config: PipelineConfig,
) -> tuple[pd.DataFrame, dict]:
    dataset = _rename_close_columns(core_frame, core_asset)
    keep_columns = [
        column
        for column in dataset.columns
        if column
        in {
            "date",
            f"{core_asset}_open",
            f"{core_asset}_high",
            f"{core_asset}_low",
            f"{core_asset}_close",
            f"{core_asset}_volume",
        }
    ]
    dataset = dataset[keep_columns].copy().sort_values("date").reset_index(drop=True)
    rows_before_purge = len(dataset)
    per_asset_purged_rows: dict[str, int] = {}

    for asset_name, macro_frame in macro_frames.items():
        series = _rename_close_columns(macro_frame[["date", "close"]], asset_name)
        observed_date_column = f"{asset_name}_observed_date"
        series[observed_date_column] = series["date"]
        dataset = dataset.merge(series[["date", f"{asset_name}_close", observed_date_column]], on="date", how="left")
        dataset[f"{asset_name}_close"] = dataset[f"{asset_name}_close"].ffill()
        dataset[observed_date_column] = dataset[observed_date_column].ffill()
        staleness_column = f"{asset_name}_staleness_days"
        dataset[staleness_column] = (dataset["date"] - dataset[observed_date_column]).dt.days
        per_asset_purged_rows[asset_name] = 0

    close_columns = [column for column in dataset.columns if column.endswith("_close")]
    staleness_columns = [column for column in dataset.columns if column.endswith("_staleness_days")]
    for staleness_column in staleness_columns:
        asset_name = staleness_column.removesuffix("_staleness_days")
        stale_asset_mask = dataset[staleness_column] > config.macro_gap_purge_days
        per_asset_purged_rows[asset_name] = int(stale_asset_mask.fillna(False).sum())
        dataset.loc[stale_asset_mask, f"{asset_name}_close"] = np.nan
    dataset = dataset.dropna(subset=close_columns)
    rows_after_purge = len(dataset)

    gap_purge_summary = {
        "macro_gap_purge_days": config.macro_gap_purge_days,
        "rows_before_purge": rows_before_purge,
        "rows_after_purge": rows_after_purge,
        "rows_removed_by_gap_purge": int(rows_before_purge - rows_after_purge),
        "per_asset_purged_rows": per_asset_purged_rows,
    }

    return dataset, gap_purge_summary


def _engineer_return_features(dataset: pd.DataFrame, core_asset: str) -> tuple[pd.DataFrame, list[str]]:
    feature_columns: list[str] = []
    close_columns = [column for column in dataset.columns if column.endswith("_close")]

    for close_column in close_columns:
        asset_name = close_column.removesuffix("_close")
        return_column = f"{asset_name}_return_1d"
        dataset[return_column] = dataset[close_column].pct_change()
        feature_columns.append(return_column)

    volume_column = f"{core_asset}_volume"
    if volume_column in dataset.columns:
        volume_change_column = f"{core_asset}_volume_change_1d"
        dataset[volume_change_column] = dataset[volume_column].pct_change()
        feature_columns.append(volume_change_column)

    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset = dataset.dropna(subset=feature_columns).copy()

    return dataset, feature_columns


def _assign_dataset_splits(dataset: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    dataset = dataset.sort_values("date").reset_index(drop=True).copy()
    row_count = len(dataset)
    if row_count < 3:
        raise ValueError("At least 3 rows are required to create train/validation/test splits.")

    train_end = int(np.floor(row_count * config.train_ratio))
    validation_end = train_end + int(np.floor(row_count * config.validation_ratio))

    train_end = max(1, train_end)
    validation_end = max(train_end + 1, validation_end)
    validation_end = min(validation_end, row_count - 1)
    train_end = min(train_end, validation_end - 1)

    split_labels = np.full(row_count, "test", dtype=object)
    split_labels[:train_end] = "train"
    split_labels[train_end:validation_end] = "validation"
    split_labels[validation_end:] = "test"
    dataset["dataset_split"] = split_labels
    return dataset


def _scale_feature_columns(dataset: pd.DataFrame, feature_columns: list[str], scaler_name: str) -> tuple[pd.DataFrame, list[str]]:
    scaler = _choose_scaler(scaler_name)
    scaled_feature_columns = [f"{column}_scaled" for column in feature_columns]

    if scaler is None:
        dataset[scaled_feature_columns] = dataset[feature_columns]
        return dataset, scaled_feature_columns

    train_mask = dataset["dataset_split"] == "train"
    scaler.fit(dataset.loc[train_mask, feature_columns])
    dataset[scaled_feature_columns] = scaler.transform(dataset[feature_columns])
    return dataset, scaled_feature_columns


def _add_supervised_target_columns(dataset: pd.DataFrame, core_asset: str, forecast_horizon: int) -> tuple[pd.DataFrame, str]:
    close_column = f"{core_asset}_close"
    target_column = f"{core_asset}_forward_return_{forecast_horizon}p"
    target_date_column = f"{core_asset}_target_date_{forecast_horizon}p"
    target_split_column = f"{core_asset}_target_split_{forecast_horizon}p"

    dataset[target_column] = dataset[close_column].shift(-forecast_horizon) / dataset[close_column] - 1
    dataset[target_date_column] = dataset["date"].shift(-forecast_horizon)
    dataset[target_split_column] = dataset["dataset_split"].shift(-forecast_horizon)
    return dataset, target_column


def _add_macro_and_signal_columns(
    dataset: pd.DataFrame,
    core_asset: str,
    macro_assets: list[str],
    config: PipelineConfig,
) -> pd.DataFrame:
    contribution_columns: list[str] = []
    total_weight = 0.0

    for asset in macro_assets:
        spec = infer_indicator_spec(asset)
        scaled_return_column = f"{asset}_return_1d_scaled"
        contribution_column = f"{asset}_macro_contribution"
        dataset[contribution_column] = dataset[scaled_return_column] * spec.direction * spec.weight
        contribution_columns.append(contribution_column)
        total_weight += abs(spec.weight)

    if contribution_columns:
        dataset["macro_support_score"] = dataset[contribution_columns].sum(axis=1) / total_weight
    else:
        dataset["macro_support_score"] = 0.0

    trend_column = f"macro_support_trend_{config.macro_trend_window}d"
    impulse_column = f"macro_support_impulse_{config.macro_trend_window}d"
    dataset[trend_column] = dataset["macro_support_score"].rolling(config.macro_trend_window).mean()
    dataset[impulse_column] = dataset[trend_column].diff(config.macro_trend_window)

    core_close_column = f"{core_asset}_close"
    drawdown_column = f"{core_asset}_drawdown"
    distance_column = f"{core_asset}_distance_to_{config.long_window}d"
    return_30d_column = f"{core_asset}_return_30d"
    return_90d_column = f"{core_asset}_return_90d"
    volume_change_mean_column = f"{core_asset}_volume_change_{config.volume_window}d_mean"
    volume_change_column = f"{core_asset}_volume_change_1d"

    dataset[drawdown_column] = dataset[core_close_column] / dataset[core_close_column].cummax() - 1
    dataset[distance_column] = dataset[core_close_column] / dataset[core_close_column].rolling(config.long_window).mean() - 1
    dataset[return_30d_column] = dataset[core_close_column].pct_change(30)
    dataset[return_90d_column] = dataset[core_close_column].pct_change(90)
    dataset[volume_change_mean_column] = dataset[volume_change_column].rolling(config.volume_window).mean()
    dataset[f"{core_asset}_realized_vol_30d"] = dataset[f"{core_asset}_return_1d"].rolling(30).std() * np.sqrt(365)

    dataset["valuation_component"] = (
        0.45 * _scale_to_unit(-dataset[drawdown_column], 0.0, 0.85)
        + 0.35 * _scale_to_unit(-dataset[distance_column], 0.0, 0.60)
        + 0.20 * _scale_to_unit(-dataset[return_30d_column], 0.0, 0.50)
    )
    dataset["macro_component"] = (
        0.70 * _scale_to_unit(dataset[trend_column], -1.50, 1.50)
        + 0.30 * _scale_to_unit(dataset[impulse_column], -0.60, 0.60)
    )
    dataset["participation_component"] = _scale_to_unit(dataset[volume_change_mean_column], -0.50, 1.00)

    dataset["accumulation_score"] = (
        100
        * (
            0.50 * dataset["valuation_component"]
            + 0.35 * dataset["macro_component"]
            + 0.15 * dataset["participation_component"]
        )
    )

    dataset["signal_zone"] = dataset.apply(
        _classify_signal,
        axis=1,
        drawdown_column=drawdown_column,
        trend_column=trend_column,
        distance_column=distance_column,
    )

    return dataset


def _add_rolling_correlations(
    dataset: pd.DataFrame,
    core_asset: str,
    macro_assets: list[str],
    correlation_window: int,
) -> pd.DataFrame:
    core_return_column = f"{core_asset}_return_1d"
    for asset in macro_assets:
        correlation_column = f"{asset}_corr_{correlation_window}d"
        dataset[correlation_column] = dataset[core_return_column].rolling(correlation_window).corr(dataset[f"{asset}_return_1d"])
    return dataset


def _add_buy_signal_columns(
    dataset: pd.DataFrame,
    core_asset: str,
    forecast_horizon: int,
) -> pd.DataFrame:
    prediction_column = f"{core_asset}_lstm_pred_forward_return_{forecast_horizon}p"
    zone_gate = dataset["signal_zone"].isin({"High Conviction Accumulation", "Layered Accumulation"})
    score_gate = dataset["accumulation_score"] >= 60
    lstm_gate = dataset[prediction_column].fillna(0).gt(0)

    dataset["buy_signal_zone_gate"] = zone_gate
    dataset["buy_signal_score_gate"] = score_gate
    dataset["buy_signal_lstm_gate"] = lstm_gate
    dataset["buy_signal_strength"] = zone_gate.astype(int) + score_gate.astype(int) + lstm_gate.astype(int)
    dataset["buy_signal"] = zone_gate & score_gate & lstm_gate
    dataset["buy_signal_label"] = np.where(dataset["buy_signal"], "Buy", "Wait")
    return dataset


def _merge_lstm_predictions_into_dataset(
    dataset: pd.DataFrame,
    core_asset: str,
    lstm_result: LSTMTrainingResult,
    forecast_horizon: int,
) -> pd.DataFrame:
    prediction_columns = [
        "date",
        "target_date",
        "sample_split",
        lstm_result.prediction_column,
        f"{core_asset}_lstm_actual_forward_return_{forecast_horizon}p",
        f"{core_asset}_lstm_residual_forward_return_{forecast_horizon}p",
    ]
    predictions = lstm_result.predictions[prediction_columns].copy()
    predictions = predictions.rename(
        columns={
            "target_date": f"{core_asset}_lstm_target_date_{forecast_horizon}p",
            "sample_split": f"{core_asset}_lstm_sample_split",
        }
    )
    dataset = dataset.merge(predictions, on="date", how="left")
    return dataset


def _build_summary(
    dataset: pd.DataFrame,
    core_asset: str,
    macro_assets: list[str],
    correlation_window: int,
    config: PipelineConfig,
    gap_purge_summary: dict,
    lstm_result: LSTMTrainingResult,
) -> dict:
    latest = dataset.dropna(subset=["accumulation_score"]).iloc[-1]
    close_column = f"{core_asset}_close"
    drawdown_column = f"{core_asset}_drawdown"
    distance_column = f"{core_asset}_distance_to_{config.long_window}d"

    drivers: list[dict] = []
    for asset in macro_assets:
        spec = infer_indicator_spec(asset)
        drivers.append(
            {
                "asset": asset,
                "label": spec.label,
                "direction": "Tailwind when rising" if spec.direction > 0 else "Headwind when rising",
                "weight": round(spec.weight, 2),
                "latest_return": float(latest[f"{asset}_return_1d"]),
                "latest_scaled_return": float(latest[f"{asset}_return_1d_scaled"]),
                "latest_contribution": float(latest[f"{asset}_macro_contribution"]),
                "rolling_correlation": float(latest.get(f"{asset}_corr_{correlation_window}d", np.nan)),
                "thesis": spec.thesis,
            }
        )

    drivers = sorted(drivers, key=lambda item: item["latest_contribution"], reverse=True)
    strongest_tailwinds = [item["label"] for item in drivers if item["latest_contribution"] > 0][:3]
    strongest_headwinds = [item["label"] for item in reversed(drivers) if item["latest_contribution"] < 0][:3]
    buy_signal_dates = dataset.loc[dataset["buy_signal"], "date"].dropna()
    latest_buy_signal_date = buy_signal_dates.max() if not buy_signal_dates.empty else pd.NaT
    split_summary: dict[str, dict[str, object]] = {}
    for split_name in ("train", "validation", "test"):
        split_frame = dataset[dataset["dataset_split"] == split_name]
        split_summary[split_name] = {
            "rows": int(len(split_frame)),
            "ratio": round(float(len(split_frame) / len(dataset)), 4),
            "start_date": str(split_frame["date"].min().date()) if not split_frame.empty else None,
            "end_date": str(split_frame["date"].max().date()) if not split_frame.empty else None,
        }

    narrative = (
        f"As of {latest['date'].date()}, {core_asset.upper()} sits in the '{latest['signal_zone']}' zone with an "
        f"accumulation score of {latest['accumulation_score']:.1f}/100. "
        f"Current valuation stress is {latest['valuation_component'] * 100:.1f}, macro support is {latest['macro_component'] * 100:.1f}, "
        f"and participation is {latest['participation_component'] * 100:.1f}. "
        f"Gap purge is active at {gap_purge_summary['macro_gap_purge_days']} stale day(s), removing "
        f"{gap_purge_summary['rows_removed_by_gap_purge']} row(s). "
        f"The LSTM forecasts the next {config.lstm_forecast_horizon} aligned period(s) from a "
        f"{config.lstm_sequence_length}-step history window. "
        f"The dataset is split chronologically into train/validation/test at "
        f"{split_summary['train']['ratio']:.2%}/{split_summary['validation']['ratio']:.2%}/{split_summary['test']['ratio']:.2%}. "
        f"Combined buy signals have triggered {int(dataset['buy_signal'].sum())} time(s). "
        f"Leading tailwinds: {', '.join(strongest_tailwinds) if strongest_tailwinds else 'none'}. "
        f"Leading headwinds: {', '.join(strongest_headwinds) if strongest_headwinds else 'none'}."
    )

    return {
        "as_of_date": str(latest["date"].date()),
        "core_asset": core_asset,
        "close": float(latest[close_column]),
        "accumulation_score": round(float(latest["accumulation_score"]), 2),
        "signal_zone": latest["signal_zone"],
        "valuation_score": round(float(latest["valuation_component"] * 100), 2),
        "macro_score": round(float(latest["macro_component"] * 100), 2),
        "participation_score": round(float(latest["participation_component"] * 100), 2),
        "drawdown": round(float(latest[drawdown_column]), 4),
        "distance_to_trend": round(float(latest[distance_column]), 4),
        "macro_support_score": round(float(latest["macro_support_score"]), 4),
        "gap_purge": gap_purge_summary,
        "split_summary": split_summary,
        "scaler_fit_split": "train",
        "buy_signal": {
            "latest_is_active": bool(latest["buy_signal"]),
            "latest_signal_date": str(latest_buy_signal_date.date()) if pd.notna(latest_buy_signal_date) else None,
            "signal_count": int(dataset["buy_signal"].sum()),
        },
        "lstm": {
            "metrics": lstm_result.metrics,
            "latest_forecast": lstm_result.latest_forecast,
            "target_column": lstm_result.target_column,
            "prediction_column": lstm_result.prediction_column,
            "sequence_length": lstm_result.sequence_length,
            "forecast_horizon_periods": lstm_result.forecast_horizon,
        },
        "strongest_tailwinds": strongest_tailwinds,
        "strongest_headwinds": strongest_headwinds,
        "drivers": drivers,
        "narrative": narrative,
    }


def run_analysis_from_frames(
    core_frames: dict[str, pd.DataFrame],
    macro_frames: dict[str, pd.DataFrame],
    config: PipelineConfig,
) -> AnalysisResult:
    core_assets = sorted(core_frames)
    macro_assets = sorted(macro_frames)
    if not core_assets:
        raise ValueError("At least one core asset frame is required.")
    if not macro_assets:
        raise ValueError("At least one macro asset frame is required.")
    if config.core_asset not in core_assets:
        raise ValueError(f"Core asset '{config.core_asset}' was not provided.")

    selected_macro_assets = config.macro_assets or macro_assets
    missing_macro_assets = sorted(set(selected_macro_assets) - set(macro_assets))
    if missing_macro_assets:
        missing = ", ".join(missing_macro_assets)
        raise ValueError(f"Unknown macro assets requested: {missing}")

    selected_macro_frames = {asset: macro_frames[asset] for asset in selected_macro_assets}
    core_frame = core_frames[config.core_asset]
    dataset, gap_purge_summary = _assemble_dataset(config.core_asset, core_frame, selected_macro_frames, config)
    dataset, feature_columns = _engineer_return_features(dataset, config.core_asset)
    dataset = _assign_dataset_splits(dataset, config)
    dataset, scaled_feature_columns = _scale_feature_columns(dataset, feature_columns, config.scaler)
    dataset, _ = _add_supervised_target_columns(dataset, config.core_asset, config.lstm_forecast_horizon)
    dataset = _add_macro_and_signal_columns(dataset, config.core_asset, selected_macro_assets, config)
    dataset = _add_rolling_correlations(dataset, config.core_asset, selected_macro_assets, config.correlation_window)
    lstm_result = train_lstm_model(
        dataset=dataset,
        core_asset=config.core_asset,
        feature_columns=scaled_feature_columns,
        config=config,
    )
    dataset = _merge_lstm_predictions_into_dataset(dataset, config.core_asset, lstm_result, config.lstm_forecast_horizon)
    dataset = _add_buy_signal_columns(dataset, config.core_asset, config.lstm_forecast_horizon)
    summary = _build_summary(
        dataset,
        config.core_asset,
        selected_macro_assets,
        config.correlation_window,
        config,
        gap_purge_summary,
        lstm_result,
    )

    return AnalysisResult(
        dataset=dataset,
        summary=summary,
        gap_purge_summary=gap_purge_summary,
        lstm_result=lstm_result,
        core_assets=core_assets,
        macro_assets=macro_assets,
        selected_macro_assets=selected_macro_assets,
        core_asset=config.core_asset,
        config=config,
    )


def run_project_analysis(project_root: Path, config: PipelineConfig) -> AnalysisResult:
    core_assets, macro_assets = discover_assets(project_root)
    if not core_assets:
        raise ValueError("No core asset CSV files were found in data/core.")
    if not macro_assets:
        raise ValueError("No macro CSV files were found in data/macro.")

    all_core_frames = load_asset_directory(project_root / "data" / "core")
    all_macro_frames = load_asset_directory(project_root / "data" / "macro")
    return run_analysis_from_frames(all_core_frames, all_macro_frames, config)
