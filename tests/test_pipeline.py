from __future__ import annotations

import numpy as np
import pandas as pd

from macro_driven_crypto_analysis.config import PipelineConfig
from macro_driven_crypto_analysis.io import clean_market_frame, load_uploaded_csv
from macro_driven_crypto_analysis.pipeline import run_analysis_from_frames, run_project_analysis


def test_clean_market_frame_drops_yahoo_ticker_row():
    raw = pd.DataFrame(
        {
            "Date": ["", "2024-01-01", "2024-01-02"],
            "Close": ["^GSPC", "4700", "4720"],
            "Volume": ["^GSPC", "100", "120"],
        }
    )

    cleaned = clean_market_frame(raw, asset_name="sp500")

    assert len(cleaned) == 2
    assert cleaned["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-01", "2024-01-02"]
    assert cleaned["close"].tolist() == [4700.0, 4720.0]


def test_load_uploaded_csv_infers_asset_name_and_preserves_ohlc():
    payload = (
        b"Date,Open,High,Low,Close,Volume\n"
        b"2024-01-01,100,110,95,108,1000\n"
        b"2024-01-02,108,112,101,111,1200\n"
    )

    asset_name, frame = load_uploaded_csv("BTC Spot.csv", payload)

    assert asset_name == "btc_spot"
    assert frame.columns.tolist() == ["date", "open", "high", "low", "close", "volume"]
    assert frame["close"].tolist() == [108.0, 111.0]


def test_run_project_analysis_builds_scores(tmp_path):
    project_root = tmp_path
    (project_root / "data" / "core").mkdir(parents=True)
    (project_root / "data" / "macro").mkdir(parents=True)

    dates = pd.date_range("2023-01-01", periods=320, freq="D")
    cycle = np.linspace(0, 6 * np.pi, len(dates))

    btc_close = 20000 + np.sin(cycle) * 2500 + np.linspace(0, 6000, len(dates))
    btc_volume = 1_000_000 + (np.cos(cycle) + 1.2) * 100_000
    sp500_close = 4000 + np.linspace(0, 700, len(dates)) + np.sin(cycle / 2) * 80
    dxy_close = 100 - np.linspace(0, 4, len(dates)) + np.cos(cycle / 2) * 1.5

    pd.DataFrame(
        {
            "Date": dates,
            "Open": btc_close * 0.99,
            "High": btc_close * 1.01,
            "Low": btc_close * 0.98,
            "Close": btc_close,
            "Volume": btc_volume,
        }
    ).to_csv(project_root / "data" / "core" / "btc.csv", index=False)

    pd.DataFrame(
        {
            "Date": dates,
            "Close": sp500_close,
        }
    ).to_csv(project_root / "data" / "macro" / "sp500.csv", index=False)

    pd.DataFrame(
        {
            "Date": dates,
            "Close": dxy_close,
        }
    ).to_csv(project_root / "data" / "macro" / "dxy.csv", index=False)

    result = run_project_analysis(
        project_root,
        PipelineConfig(
            project_root=project_root,
            core_asset="btc",
            scaler="standard",
            lstm_sequence_length=12,
            lstm_forecast_horizon=5,
            lstm_units=4,
            lstm_dense_units=4,
            lstm_batch_size=8,
            lstm_epochs=2,
            lstm_patience=1,
            correlation_window=60,
            macro_trend_window=21,
            long_window=120,
            volume_window=21,
        ),
    )

    assert not result.dataset.empty
    assert "dataset_split" in result.dataset.columns
    assert "accumulation_score" in result.dataset.columns
    assert "signal_zone" in result.dataset.columns
    assert "btc_open" in result.dataset.columns
    assert "btc_high" in result.dataset.columns
    assert "btc_low" in result.dataset.columns
    assert "buy_signal" in result.dataset.columns
    assert "buy_signal_strength" in result.dataset.columns
    assert set(result.dataset["dataset_split"]) == {"train", "validation", "test"}
    split_counts = result.dataset["dataset_split"].value_counts().to_dict()
    assert split_counts["train"] > split_counts["validation"] > 0
    assert split_counts["test"] > 0
    train_end_date = result.dataset.loc[result.dataset["dataset_split"] == "train", "date"].max()
    validation_start_date = result.dataset.loc[result.dataset["dataset_split"] == "validation", "date"].min()
    validation_end_date = result.dataset.loc[result.dataset["dataset_split"] == "validation", "date"].max()
    test_start_date = result.dataset.loc[result.dataset["dataset_split"] == "test", "date"].min()
    assert train_end_date < validation_start_date < test_start_date
    assert validation_end_date < test_start_date
    train_scaled_mean = result.dataset.loc[
        result.dataset["dataset_split"] == "train",
        "btc_return_1d_scaled",
    ].mean()
    assert abs(train_scaled_mean) < 1e-9
    assert result.summary["scaler_fit_split"] == "train"
    assert result.summary["split_summary"]["train"]["rows"] == split_counts["train"]
    assert result.summary["gap_purge"]["macro_gap_purge_days"] == 0
    assert "lstm" in result.summary
    assert "buy_signal" in result.summary
    assert result.summary["lstm"]["metrics"]["test"]["count"] > 0
    assert result.summary["lstm"]["latest_forecast"]["predicted_direction"] in {"Bullish", "Bearish"}
    assert result.summary["signal_zone"] in {
        "High Conviction Accumulation",
        "Layered Accumulation",
        "Macro Headwind",
        "Extended / Chase Risk",
        "Neutral",
    }
    assert isinstance(result.summary["drivers"], list)
    assert result.lstm_result.history.shape[0] >= 1


def test_run_analysis_from_frames_supports_in_memory_upload_flow():
    dates = pd.date_range("2023-01-01", periods=180, freq="D")
    cycle = np.linspace(0, 5 * np.pi, len(dates))

    core_frame = clean_market_frame(
        pd.DataFrame(
            {
                "Date": dates,
                "Open": 15000 + np.sin(cycle) * 200,
                "High": 15200 + np.sin(cycle) * 220,
                "Low": 14800 + np.sin(cycle) * 180,
                "Close": 15100 + np.sin(cycle) * 210 + np.linspace(0, 2500, len(dates)),
                "Volume": 900_000 + (np.cos(cycle) + 1.2) * 80_000,
            }
        ),
        asset_name="btc",
    )
    sp500_frame = clean_market_frame(
        pd.DataFrame(
            {
                "Date": dates,
                "Close": 3800 + np.linspace(0, 500, len(dates)) + np.sin(cycle / 2) * 50,
            }
        ),
        asset_name="sp500",
    )
    dxy_frame = clean_market_frame(
        pd.DataFrame(
            {
                "Date": dates,
                "Close": 104 - np.linspace(0, 3, len(dates)) + np.cos(cycle / 2) * 1.2,
            }
        ),
        asset_name="dxy",
    )

    result = run_analysis_from_frames(
        core_frames={"btc": core_frame},
        macro_frames={"sp500": sp500_frame, "dxy": dxy_frame},
        config=PipelineConfig(
            core_asset="btc",
            macro_assets=["sp500", "dxy"],
            scaler="standard",
            lstm_sequence_length=10,
            lstm_forecast_horizon=4,
            lstm_units=4,
            lstm_dense_units=4,
            lstm_batch_size=8,
            lstm_epochs=2,
            lstm_patience=1,
            correlation_window=20,
            macro_trend_window=10,
            long_window=30,
            volume_window=10,
        ),
    )

    assert not result.dataset.empty
    assert result.core_assets == ["btc"]
    assert result.selected_macro_assets == ["sp500", "dxy"]
    assert result.dataset["buy_signal_label"].isin(["Buy", "Wait"]).all()


def test_gap_purge_removes_stale_macro_rows(tmp_path):
    project_root = tmp_path
    (project_root / "data" / "core").mkdir(parents=True)
    (project_root / "data" / "macro").mkdir(parents=True)

    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    macro_dates = dates[::2]

    pd.DataFrame(
        {
            "Date": dates,
            "Close": np.linspace(100, 111, len(dates)),
            "Volume": np.linspace(1000, 1110, len(dates)),
        }
    ).to_csv(project_root / "data" / "core" / "btc.csv", index=False)

    pd.DataFrame(
        {
            "Date": macro_dates,
            "Close": np.linspace(200, 210, len(macro_dates)),
        }
    ).to_csv(project_root / "data" / "macro" / "sp500.csv", index=False)

    pd.DataFrame(
        {
            "Date": macro_dates,
            "Close": np.linspace(90, 84, len(macro_dates)),
        }
    ).to_csv(project_root / "data" / "macro" / "dxy.csv", index=False)

    result = run_project_analysis(
        project_root,
        PipelineConfig(
            project_root=project_root,
            core_asset="btc",
            scaler="standard",
            macro_gap_purge_days=0,
            lstm_sequence_length=8,
            lstm_forecast_horizon=3,
            lstm_units=4,
            lstm_dense_units=4,
            lstm_batch_size=4,
            lstm_epochs=2,
            lstm_patience=1,
            correlation_window=3,
            macro_trend_window=2,
            long_window=3,
            volume_window=2,
        ),
    )

    assert result.summary["gap_purge"]["rows_removed_by_gap_purge"] > 0
    assert result.dataset["sp500_staleness_days"].eq(0).all()
    assert result.dataset["dxy_staleness_days"].eq(0).all()
    remaining_dates = result.dataset["date"].dt.strftime("%Y-%m-%d").tolist()
    assert remaining_dates[0] == "2024-01-03"
    assert remaining_dates[-1] == "2024-03-19"
    assert len(remaining_dates) == 39
    assert result.summary["lstm"]["metrics"]["validation"]["count"] > 0
