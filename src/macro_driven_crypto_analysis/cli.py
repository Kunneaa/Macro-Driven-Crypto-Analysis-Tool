from __future__ import annotations

import argparse
from pathlib import Path

from .config import PipelineConfig, SUPPORTED_SCALERS
from .insights import export_analysis
from .pipeline import run_project_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a macro-aware processed dataset and latest accumulation snapshot."
    )
    parser.add_argument("--project-root", type=Path, default=Path(".").resolve())
    parser.add_argument("--core", default="btc", help="Core asset slug from data/core, for example btc")
    parser.add_argument(
        "--macro",
        nargs="*",
        default=None,
        help="Optional list of macro asset slugs from data/macro. Defaults to every available macro file.",
    )
    parser.add_argument("--scaler", choices=SUPPORTED_SCALERS, default="standard")
    parser.add_argument("--macro-gap-purge-days", type=int, default=0)
    parser.add_argument("--lstm-sequence-length", type=int, default=60)
    parser.add_argument("--lstm-forecast-horizon", type=int, default=30)
    parser.add_argument("--lstm-units", type=int, default=48)
    parser.add_argument("--lstm-dense-units", type=int, default=16)
    parser.add_argument("--lstm-dropout", type=float, default=0.15)
    parser.add_argument("--lstm-learning-rate", type=float, default=0.001)
    parser.add_argument("--lstm-batch-size", type=int, default=32)
    parser.add_argument("--lstm-epochs", type=int, default=20)
    parser.add_argument("--lstm-patience", type=int, default=5)
    parser.add_argument("--correlation-window", type=int, default=90)
    parser.add_argument("--macro-trend-window", type=int, default=21)
    parser.add_argument("--long-window", type=int, default=200)
    parser.add_argument("--volume-window", type=int, default=21)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config = PipelineConfig(
        project_root=args.project_root.resolve(),
        core_asset=args.core,
        macro_assets=args.macro,
        scaler=args.scaler,
        macro_gap_purge_days=args.macro_gap_purge_days,
        lstm_sequence_length=args.lstm_sequence_length,
        lstm_forecast_horizon=args.lstm_forecast_horizon,
        lstm_units=args.lstm_units,
        lstm_dense_units=args.lstm_dense_units,
        lstm_dropout=args.lstm_dropout,
        lstm_learning_rate=args.lstm_learning_rate,
        lstm_batch_size=args.lstm_batch_size,
        lstm_epochs=args.lstm_epochs,
        lstm_patience=args.lstm_patience,
        correlation_window=args.correlation_window,
        macro_trend_window=args.macro_trend_window,
        long_window=args.long_window,
        volume_window=args.volume_window,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
    )

    result = run_project_analysis(config.project_root, config)
    output_directory = args.output_dir
    if not output_directory.is_absolute():
        output_directory = config.project_root / output_directory

    paths = export_analysis(result, output_directory)

    print(f"Built analysis for {result.core_asset.upper()} on {result.summary['as_of_date']}")
    print(f"Signal zone: {result.summary['signal_zone']}")
    print(f"Accumulation score: {result.summary['accumulation_score']}")
    print(
        f"Gap purge: {result.summary['gap_purge']['macro_gap_purge_days']} stale day(s), "
        f"removed {result.summary['gap_purge']['rows_removed_by_gap_purge']} row(s)"
    )
    print(
        "Split ratios: "
        f"train={result.summary['split_summary']['train']['ratio']:.2%}, "
        f"validation={result.summary['split_summary']['validation']['ratio']:.2%}, "
        f"test={result.summary['split_summary']['test']['ratio']:.2%}"
    )
    print(
        "LSTM: "
        f"val_rmse={result.summary['lstm']['metrics']['validation']['rmse']:.4f}, "
        f"test_rmse={result.summary['lstm']['metrics']['test']['rmse']:.4f}, "
        f"latest_forecast={result.summary['lstm']['latest_forecast']['predicted_forward_return']:.2%}"
    )
    print(f"Scaler fit split: {result.summary['scaler_fit_split']}")
    print(f"Dataset: {paths['dataset']}")
    print(f"Train split: {paths['train']}")
    print(f"Validation split: {paths['validation']}")
    print(f"Test split: {paths['test']}")
    print(f"Signal history: {paths['signal_history']}")
    print(f"LSTM predictions: {paths['lstm_predictions']}")
    print(f"LSTM history: {paths['lstm_history']}")
    print(f"LSTM metrics: {paths['lstm_metrics']}")
    print(f"Summary JSON: {paths['summary']}")
    print(f"Snapshot: {paths['snapshot']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
