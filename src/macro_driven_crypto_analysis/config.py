from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


SUPPORTED_SCALERS = ("standard", "minmax", "none")


@dataclass(frozen=True)
class IndicatorSpec:
    slug: str
    label: str
    direction: int
    weight: float
    thesis: str


DEFAULT_INDICATOR_SPECS: dict[str, IndicatorSpec] = {
    "sp500": IndicatorSpec(
        slug="sp500",
        label="S&P 500",
        direction=1,
        weight=1.0,
        thesis="Higher equity risk appetite often coincides with stronger crypto flows.",
    ),
    "nasdaq": IndicatorSpec(
        slug="nasdaq",
        label="NASDAQ",
        direction=1,
        weight=1.15,
        thesis="Growth-heavy equity strength often aligns with higher crypto beta demand.",
    ),
    "eth": IndicatorSpec(
        slug="eth",
        label="ETH",
        direction=1,
        weight=0.95,
        thesis="ETH often behaves as a higher-beta crypto proxy for risk appetite inside the asset class.",
    ),
    "dxy": IndicatorSpec(
        slug="dxy",
        label="US Dollar Index",
        direction=-1,
        weight=1.15,
        thesis="A stronger dollar typically tightens global liquidity and pressures crypto multiples.",
    ),
    "vix": IndicatorSpec(
        slug="vix",
        label="VIX",
        direction=-1,
        weight=1.1,
        thesis="Rising volatility stress usually reflects falling risk appetite across markets.",
    ),
    "us10y": IndicatorSpec(
        slug="us10y",
        label="US 10Y Yield",
        direction=-1,
        weight=0.9,
        thesis="Higher long-end yields can pressure long-duration and liquidity-sensitive assets.",
    ),
    "us2y": IndicatorSpec(
        slug="us2y",
        label="US 2Y Yield",
        direction=-1,
        weight=0.8,
        thesis="Higher front-end yields often signal tighter policy and a less supportive macro backdrop.",
    ),
    "gold": IndicatorSpec(
        slug="gold",
        label="Gold",
        direction=1,
        weight=0.35,
        thesis="Gold can act as a partial store-of-value proxy, but the relationship is weaker than equities or DXY.",
    ),
    "oil_wti": IndicatorSpec(
        slug="oil_wti",
        label="WTI Crude",
        direction=-1,
        weight=0.45,
        thesis="Higher oil can feed inflation pressure and keep policy tighter for longer.",
    ),
    "oil_brent": IndicatorSpec(
        slug="oil_brent",
        label="Brent Crude",
        direction=-1,
        weight=0.45,
        thesis="Higher oil can feed inflation pressure and keep policy tighter for longer.",
    ),
}


@dataclass
class PipelineConfig:
    project_root: Path = field(default_factory=lambda: Path(".").resolve())
    core_asset: str = "btc"
    macro_assets: list[str] | None = None
    scaler: str = "standard"
    macro_gap_purge_days: int = 0
    lstm_sequence_length: int = 60
    lstm_forecast_horizon: int = 30
    lstm_units: int = 48
    lstm_dense_units: int = 16
    lstm_dropout: float = 0.15
    lstm_learning_rate: float = 0.001
    lstm_batch_size: int = 32
    lstm_epochs: int = 20
    lstm_patience: int = 5
    lstm_seed: int = 42
    correlation_window: int = 90
    macro_trend_window: int = 21
    long_window: int = 200
    volume_window: int = 21
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15

    def __post_init__(self) -> None:
        self.core_asset = self.core_asset.lower()
        if self.macro_assets:
            self.macro_assets = [asset.lower() for asset in self.macro_assets]
        if self.scaler not in SUPPORTED_SCALERS:
            choices = ", ".join(SUPPORTED_SCALERS)
            raise ValueError(f"Unsupported scaler '{self.scaler}'. Choose from: {choices}")
        ratios = (self.train_ratio, self.validation_ratio, self.test_ratio)
        if any(ratio < 0 for ratio in ratios):
            raise ValueError("Train, validation, and test ratios must be non-negative.")
        if abs(sum(ratios) - 1.0) > 1e-9:
            raise ValueError("Train, validation, and test ratios must sum to 1.0.")
        if self.macro_gap_purge_days < 0:
            raise ValueError("macro_gap_purge_days must be >= 0.")
        if self.lstm_sequence_length < 2:
            raise ValueError("lstm_sequence_length must be >= 2.")
        if self.lstm_forecast_horizon < 1:
            raise ValueError("lstm_forecast_horizon must be >= 1.")
        if self.lstm_units < 1 or self.lstm_dense_units < 1:
            raise ValueError("lstm_units and lstm_dense_units must be >= 1.")
        if not 0.0 <= self.lstm_dropout < 1.0:
            raise ValueError("lstm_dropout must be in [0, 1).")
        if self.lstm_learning_rate <= 0:
            raise ValueError("lstm_learning_rate must be > 0.")
        if self.lstm_batch_size < 1 or self.lstm_epochs < 1 or self.lstm_patience < 1:
            raise ValueError("lstm_batch_size, lstm_epochs, and lstm_patience must be >= 1.")
