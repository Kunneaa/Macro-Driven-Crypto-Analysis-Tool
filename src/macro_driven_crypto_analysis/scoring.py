from __future__ import annotations

from .config import DEFAULT_INDICATOR_SPECS, IndicatorSpec


def infer_indicator_spec(asset: str) -> IndicatorSpec:
    asset = asset.lower()
    if asset in DEFAULT_INDICATOR_SPECS:
        return DEFAULT_INDICATOR_SPECS[asset]

    if any(keyword in asset for keyword in ("dxy", "yield", "us10", "us2", "vix", "oil", "brent", "wti")):
        direction = -1
        thesis = "This series is treated as a headwind when rising because it resembles a liquidity or stress indicator."
    else:
        direction = 1
        thesis = "This series is treated as a tailwind when rising because it resembles a risk or growth proxy."

    return IndicatorSpec(
        slug=asset,
        label=asset.replace("_", " ").upper(),
        direction=direction,
        weight=0.6,
        thesis=thesis,
    )

