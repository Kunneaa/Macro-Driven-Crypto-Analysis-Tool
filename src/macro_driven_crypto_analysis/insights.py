from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .pipeline import AnalysisResult


def _write_markdown_snapshot(result: AnalysisResult, markdown_path: Path) -> None:
    summary = result.summary
    lines = [
        f"# {result.core_asset.upper()} Macro Snapshot",
        "",
        f"- As of: {summary['as_of_date']}",
        f"- Signal zone: {summary['signal_zone']}",
        f"- Accumulation score: {summary['accumulation_score']}",
        f"- Valuation score: {summary['valuation_score']}",
        f"- Macro score: {summary['macro_score']}",
        f"- Participation score: {summary['participation_score']}",
        f"- Drawdown from peak: {summary['drawdown']:.2%}",
        f"- Distance to long trend: {summary['distance_to_trend']:.2%}",
        f"- Gap purge: {summary['gap_purge']['macro_gap_purge_days']} stale day(s), "
        f"{summary['gap_purge']['rows_removed_by_gap_purge']} rows removed",
        f"- Chronological split: train {summary['split_summary']['train']['ratio']:.2%}, "
        f"validation {summary['split_summary']['validation']['ratio']:.2%}, "
        f"test {summary['split_summary']['test']['ratio']:.2%}",
        f"- Scaler fit split: {summary['scaler_fit_split']}",
        f"- LSTM forecast horizon: {summary['lstm']['forecast_horizon_periods']} periods",
        f"- LSTM latest forecast: {summary['lstm']['latest_forecast']['predicted_forward_return']:.2%} "
        f"({summary['lstm']['latest_forecast']['predicted_direction']})",
        f"- Buy signals: {summary['buy_signal']['signal_count']} total, "
        f"latest at {summary['buy_signal']['latest_signal_date'] or 'n/a'}",
        "",
        summary["narrative"],
        "",
        "## Dataset Splits",
        "",
    ]

    for split_name in ("train", "validation", "test"):
        split_info = summary["split_summary"][split_name]
        lines.append(
            f"- {split_name}: {split_info['rows']} rows, {split_info['start_date']} -> {split_info['end_date']}"
        )

    lines.extend([
        "",
        "## LSTM Metrics",
        "",
    ])

    for split_name in ("train", "validation", "test"):
        split_metrics = summary["lstm"]["metrics"][split_name]
        lines.append(
            f"- {split_name}: count {split_metrics['count']}, mae {split_metrics['mae']:.4f}, "
            f"rmse {split_metrics['rmse']:.4f}, direction {split_metrics['directional_accuracy']:.2%}"
        )

    lines.extend([
        "",
        "## Macro Drivers",
        "",
    ])

    for driver in summary["drivers"]:
        lines.append(
            f"- {driver['label']}: contribution {driver['latest_contribution']:.3f}, "
            f"corr {driver['rolling_correlation']:.3f}, thesis: {driver['thesis']}"
        )

    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def export_analysis(result: AnalysisResult, output_directory: Path) -> dict[str, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = f"{result.core_asset}_{result.config.scaler}"

    dataset_path = output_directory / f"{stem}_analysis_dataset.csv"
    signal_history_path = output_directory / f"{stem}_signal_history.csv"
    summary_path = output_directory / f"{stem}_summary.json"
    snapshot_path = output_directory / f"{stem}_snapshot.md"
    train_path = output_directory / f"{stem}_train.csv"
    validation_path = output_directory / f"{stem}_validation.csv"
    test_path = output_directory / f"{stem}_test.csv"
    lstm_predictions_path = output_directory / f"{stem}_lstm_predictions.csv"
    lstm_history_path = output_directory / f"{stem}_lstm_history.csv"
    lstm_metrics_path = output_directory / f"{stem}_lstm_metrics.json"

    result.dataset.to_csv(dataset_path, index=False)
    result.dataset[result.dataset["dataset_split"] == "train"].to_csv(train_path, index=False)
    result.dataset[result.dataset["dataset_split"] == "validation"].to_csv(validation_path, index=False)
    result.dataset[result.dataset["dataset_split"] == "test"].to_csv(test_path, index=False)
    result.lstm_result.predictions.to_csv(lstm_predictions_path, index=False)
    result.lstm_result.history.to_csv(lstm_history_path, index=False)
    lstm_metrics_path.write_text(json.dumps(result.lstm_result.metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    history_columns = [
        "date",
        "dataset_split",
        f"{result.core_asset}_close",
        "accumulation_score",
        "signal_zone",
        "valuation_component",
        "macro_component",
        "participation_component",
        "macro_support_score",
        "buy_signal",
        "buy_signal_label",
        "buy_signal_strength",
    ]
    history = result.dataset[[column for column in history_columns if column in result.dataset.columns]].copy()
    history.to_csv(signal_history_path, index=False)

    summary_path.write_text(json.dumps(result.summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown_snapshot(result, snapshot_path)

    return {
        "dataset": dataset_path,
        "train": train_path,
        "validation": validation_path,
        "test": test_path,
        "signal_history": signal_history_path,
        "lstm_predictions": lstm_predictions_path,
        "lstm_history": lstm_history_path,
        "lstm_metrics": lstm_metrics_path,
        "summary": summary_path,
        "snapshot": snapshot_path,
    }
