from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from macro_driven_crypto_analysis import (
    PipelineConfig,
    discover_assets,
    export_analysis,
    run_analysis_from_frames,
    run_project_analysis,
)
from macro_driven_crypto_analysis.io import infer_asset_name, load_uploaded_csv


st.set_page_config(
    page_title="Macro-Driven Crypto Analysis Tool",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
    html, body, [class*="css"]  {
        font-family: "IBM Plex Sans", sans-serif;
    }
    h1, h2, h3 {
        font-family: "Fraunces", serif;
        letter-spacing: -0.02em;
    }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(184,91,41,0.12), transparent 28%),
            radial-gradient(circle at top right, rgba(19,36,48,0.10), transparent 26%),
            linear-gradient(180deg, #f6f1e8 0%, #f1e7d9 100%);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.7);
        border: 1px solid rgba(19, 36, 48, 0.08);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: 0 12px 30px rgba(19, 36, 48, 0.07);
    }
    .eyebrow {
        text-transform: uppercase;
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        color: #6a4f3a;
    }
    .hero-note {
        background: rgba(19, 36, 48, 0.88);
        color: #f6f1e8;
        border-radius: 18px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_uploaded_frames(upload_entries: tuple[tuple[str, bytes], ...]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    duplicates: list[str] = []

    for upload_name, payload in upload_entries:
        asset_name, frame = load_uploaded_csv(upload_name, payload)
        if asset_name in frames:
            duplicates.append(asset_name)
            continue
        frames[asset_name] = frame

    if duplicates:
        duplicate_list = ", ".join(sorted(set(duplicates)))
        raise ValueError(
            "Duplicate uploaded asset names were found after filename normalization: "
            f"{duplicate_list}. Rename the files and upload again."
        )

    return frames


@st.cache_data(show_spinner=False)
def build_workspace_analysis(
    project_root: str,
    core_asset: str,
    macro_assets: tuple[str, ...],
    scaler: str,
    macro_gap_purge_days: int,
    correlation_window: int,
    macro_trend_window: int,
    long_window: int,
    volume_window: int,
):
    config = PipelineConfig(
        project_root=Path(project_root),
        core_asset=core_asset,
        macro_assets=list(macro_assets),
        scaler=scaler,
        macro_gap_purge_days=macro_gap_purge_days,
        correlation_window=correlation_window,
        macro_trend_window=macro_trend_window,
        long_window=long_window,
        volume_window=volume_window,
    )
    return run_project_analysis(Path(project_root), config)


@st.cache_data(show_spinner=False)
def build_uploaded_analysis(
    project_root: str,
    core_upload_name: str,
    core_upload_payload: bytes,
    macro_uploads: tuple[tuple[str, bytes], ...],
    macro_assets: tuple[str, ...],
    scaler: str,
    macro_gap_purge_days: int,
    correlation_window: int,
    macro_trend_window: int,
    long_window: int,
    volume_window: int,
):
    core_asset, core_frame = load_uploaded_csv(core_upload_name, core_upload_payload)
    macro_frames = load_uploaded_frames(macro_uploads)
    config = PipelineConfig(
        project_root=Path(project_root),
        core_asset=core_asset,
        macro_assets=list(macro_assets),
        scaler=scaler,
        macro_gap_purge_days=macro_gap_purge_days,
        correlation_window=correlation_window,
        macro_trend_window=macro_trend_window,
        long_window=long_window,
        volume_window=volume_window,
    )
    return run_analysis_from_frames({core_asset: core_frame}, macro_frames, config)


def format_pct(value: float) -> str:
    return f"{value:.2%}"


def metric_card(label: str, value: str, detail: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="eyebrow">{label}</div>
            <div style="font-size: 1.9rem; font-weight: 700; color: #132430;">{value}</div>
            <div style="color: #5d6971; font-size: 0.92rem;">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def normalize_prices(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    normalized = frame[["date", *columns]].dropna().copy()
    if normalized.empty:
        return normalized
    for column in columns:
        normalized[column] = normalized[column] / normalized[column].iloc[0] * 100
    return normalized


def build_score_chart(frame: pd.DataFrame) -> go.Figure:
    chart = go.Figure()
    chart.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["accumulation_score"],
            mode="lines",
            name="Accumulation Score",
            line={"color": "#b85b29", "width": 3},
        )
    )
    chart.add_hrect(y0=72, y1=100, fillcolor="rgba(57, 127, 82, 0.12)", line_width=0)
    chart.add_hrect(y0=60, y1=72, fillcolor="rgba(184, 91, 41, 0.12)", line_width=0)
    chart.add_hrect(y0=35, y1=60, fillcolor="rgba(19, 36, 48, 0.08)", line_width=0)
    chart.add_hrect(y0=0, y1=35, fillcolor="rgba(135, 44, 44, 0.10)", line_width=0)
    chart.update_layout(
        height=420,
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        yaxis_title="Score",
        xaxis_title="Date",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.35)",
        legend={"orientation": "h"},
    )
    return chart


def build_overlay_chart(frame: pd.DataFrame, price_columns: list[str]) -> go.Figure:
    normalized = normalize_prices(frame, price_columns)
    chart = go.Figure()
    palette = ["#132430", "#b85b29", "#3a6d8c", "#5c7f67", "#7b4b2a", "#7d6572", "#3b3b55", "#8a9a5b"]

    for index, column in enumerate(price_columns):
        label = column.removesuffix("_close").upper()
        chart.add_trace(
            go.Scatter(
                x=normalized["date"],
                y=normalized[column],
                mode="lines",
                name=label,
                line={"width": 2.5, "color": palette[index % len(palette)]},
            )
        )

    chart.update_layout(
        height=430,
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        yaxis_title="Normalized to 100",
        xaxis_title="Date",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.35)",
        legend={"orientation": "h"},
    )
    return chart


def build_correlation_chart(frame: pd.DataFrame, macro_asset: str, correlation_window: int) -> go.Figure:
    chart = go.Figure()
    chart.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame[f"{macro_asset}_corr_{correlation_window}d"],
            mode="lines",
            name=f"{macro_asset.upper()} Corr",
            line={"width": 2.5, "color": "#132430"},
        )
    )
    chart.add_hline(y=0, line_width=1, line_dash="dash", line_color="#6e6e6e")
    chart.update_layout(
        height=380,
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        yaxis_title="Rolling Correlation",
        xaxis_title="Date",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.35)",
        legend={"orientation": "h"},
    )
    return chart


def build_candlestick_chart(frame: pd.DataFrame, core_asset: str, forecast_horizon: int) -> go.Figure:
    open_column = f"{core_asset}_open"
    high_column = f"{core_asset}_high"
    low_column = f"{core_asset}_low"
    close_column = f"{core_asset}_close"
    prediction_column = f"{core_asset}_lstm_pred_forward_return_{forecast_horizon}p"

    chart = go.Figure()
    chart.add_trace(
        go.Candlestick(
            x=frame["date"],
            open=frame[open_column],
            high=frame[high_column],
            low=frame[low_column],
            close=frame[close_column],
            name=f"{core_asset.upper()} Candles",
            increasing_line_color="#2d6f4b",
            decreasing_line_color="#8c3b34",
        )
    )

    buy_frame = frame[frame["buy_signal"]].copy()
    if not buy_frame.empty:
        marker_price = buy_frame[low_column].fillna(buy_frame[close_column]) * 0.985
        hover_text = [
            (
                f"{date.date()}<br>"
                f"Zone: {zone}<br>"
                f"Accumulation score: {score:.1f}<br>"
                f"LSTM forecast: {prediction:.2%}"
            )
            for date, zone, score, prediction in zip(
                buy_frame["date"],
                buy_frame["signal_zone"],
                buy_frame["accumulation_score"],
                buy_frame[prediction_column],
            )
        ]
        chart.add_trace(
            go.Scatter(
                x=buy_frame["date"],
                y=marker_price,
                mode="markers",
                name="Buy Signal",
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                marker={
                    "symbol": "triangle-up",
                    "size": 11,
                    "color": "#2d6f4b",
                    "line": {"color": "#f6f1e8", "width": 1},
                },
            )
        )

    chart.update_layout(
        height=470,
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        xaxis_title="Date",
        yaxis_title="Price",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.35)",
        legend={"orientation": "h"},
        xaxis_rangeslider_visible=False,
    )
    return chart


workspace_core_assets, workspace_macro_assets = discover_assets(PROJECT_ROOT)
default_macro_selection = workspace_macro_assets[: min(6, len(workspace_macro_assets))]

st.title("Macro-Driven Crypto Analysis Tool")
st.markdown(
    """
    <div class="hero-note">
        Built for long-term accumulation, not short-term trading. Upload new core and macro CSV files directly in the app
        or point the dashboard at the workspace data folders. The workflow validates data, aligns macro series to the
        core timeline, purges stale macro gaps, creates chronological train/validation/test splits, fits scaling on
        train only, trains an LSTM, and now overlays buy signals on a Japanese candlestick chart when OHLC data exists.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Pipeline Controls")
    data_source_mode = st.radio("Data source", options=["Workspace data folders", "Upload CSV files"], index=0)
    scaler = st.selectbox("Scaler", options=["standard", "minmax", "none"], index=0)
    macro_gap_purge_days = st.slider("Macro gap purge (days)", min_value=0, max_value=7, value=0, step=1)
    correlation_window = st.slider("Correlation window", min_value=30, max_value=252, value=90, step=10)
    macro_trend_window = st.slider("Macro trend window", min_value=10, max_value=90, value=21, step=1)
    long_window = st.slider("Long trend window", min_value=90, max_value=365, value=200, step=5)
    volume_window = st.slider("Volume window", min_value=7, max_value=90, value=21, step=1)
    export_button = st.button("Export processed files", use_container_width=True)

    core_asset: str | None = None
    selected_macro_assets: list[str] = []
    uploaded_core_name = ""
    uploaded_core_payload = b""
    macro_upload_entries: tuple[tuple[str, bytes], ...] = ()

    if data_source_mode == "Workspace data folders":
        if not workspace_core_assets:
            st.error("No core CSV files were found in data/core. Switch to upload mode or add files to the project.")
            st.stop()
        if not workspace_macro_assets:
            st.error("No macro CSV files were found in data/macro. Switch to upload mode or add files to the project.")
            st.stop()

        core_asset = st.selectbox(
            "Core asset",
            options=workspace_core_assets,
            index=workspace_core_assets.index("btc") if "btc" in workspace_core_assets else 0,
        )
        selected_macro_assets = st.multiselect(
            "Macro assets",
            options=workspace_macro_assets,
            default=default_macro_selection,
        )
    else:
        st.caption("Upload 1 core CSV with `Date`, `Close`, `Volume`, and optional `Open/High/Low` for candlesticks.")
        uploaded_core_file = st.file_uploader("Upload core CSV", type="csv", accept_multiple_files=False)
        uploaded_macro_files = st.file_uploader("Upload macro CSV files", type="csv", accept_multiple_files=True)

        if uploaded_core_file is not None:
            uploaded_core_name = uploaded_core_file.name
            uploaded_core_payload = uploaded_core_file.getvalue()
            core_asset = infer_asset_name(uploaded_core_name)
            st.caption(f"Detected core asset slug: `{core_asset}`")

        macro_upload_entries = tuple(
            (uploaded_file.name, uploaded_file.getvalue()) for uploaded_file in (uploaded_macro_files or [])
        )
        uploaded_macro_slugs = [infer_asset_name(upload_name) for upload_name, _ in macro_upload_entries]
        if uploaded_macro_slugs:
            selected_macro_assets = st.multiselect(
                "Macro assets",
                options=uploaded_macro_slugs,
                default=uploaded_macro_slugs,
            )
            st.caption("Detected macro assets: " + ", ".join(asset.upper() for asset in uploaded_macro_slugs))

if not core_asset:
    st.info("Choose a workspace core asset or upload a core CSV to begin.")
    st.stop()

if not selected_macro_assets:
    st.warning("Select at least one macro series to build the analysis.")
    st.stop()

if data_source_mode == "Upload CSV files" and (not uploaded_core_payload or not macro_upload_entries):
    st.info("Upload one core CSV and at least one macro CSV, then the app will process them immediately.")
    st.stop()

try:
    with st.spinner("Processing aligned dataset, training the LSTM, and preparing dashboard views..."):
        if data_source_mode == "Workspace data folders":
            result = build_workspace_analysis(
                str(PROJECT_ROOT),
                core_asset,
                tuple(selected_macro_assets),
                scaler,
                macro_gap_purge_days,
                correlation_window,
                macro_trend_window,
                long_window,
                volume_window,
            )
        else:
            result = build_uploaded_analysis(
                str(PROJECT_ROOT),
                uploaded_core_name,
                uploaded_core_payload,
                macro_upload_entries,
                tuple(selected_macro_assets),
                scaler,
                macro_gap_purge_days,
                correlation_window,
                macro_trend_window,
                long_window,
                volume_window,
            )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

dataset = result.dataset.dropna(subset=["accumulation_score"]).copy()
summary = result.summary

if export_button:
    paths = export_analysis(result, PROJECT_ROOT / "data" / "processed")
    st.success(f"Processed outputs written to {paths['dataset'].parent}")

metric_columns = st.columns(4)
with metric_columns[0]:
    metric_card("Signal Zone", summary["signal_zone"], f"As of {summary['as_of_date']}")
with metric_columns[1]:
    metric_card("Accumulation Score", f"{summary['accumulation_score']:.1f}", "Higher suggests stronger long-term accumulation context")
with metric_columns[2]:
    metric_card("Drawdown", format_pct(summary["drawdown"]), "Distance from prior all-time high")
with metric_columns[3]:
    metric_card("Macro Support", f"{summary['macro_score']:.1f}", "Composite of weighted macro tailwinds and headwinds")

st.caption(summary["narrative"])
st.caption(
    f"Gap purge: {summary['gap_purge']['macro_gap_purge_days']} stale day(s), "
    f"{summary['gap_purge']['rows_removed_by_gap_purge']} row(s) removed. "
    "Chronological split: "
    f"train {summary['split_summary']['train']['rows']} rows, "
    f"validation {summary['split_summary']['validation']['rows']} rows, "
    f"test {summary['split_summary']['test']['rows']} rows. "
    f"Scaler fit split: {summary['scaler_fit_split']}. "
    f"LSTM latest forecast: {summary['lstm']['latest_forecast']['predicted_forward_return']:.2%} "
    f"({summary['lstm']['latest_forecast']['predicted_direction']}). "
    f"Buy signals found: {summary['buy_signal']['signal_count']}."
)

timeframe = st.select_slider(
    "Displayed history",
    options=["1Y", "2Y", "3Y", "5Y", "All"],
    value="3Y",
)

if timeframe == "All":
    visible = dataset.copy()
else:
    years = int(timeframe.rstrip("Y"))
    cutoff = dataset["date"].max() - pd.DateOffset(years=years)
    visible = dataset[dataset["date"] >= cutoff].copy()

tab_overview, tab_structure, tab_candles, tab_lstm, tab_drivers = st.tabs(
    ["Accumulation", "Macro Structure", "Candlestick", "LSTM", "Drivers"]
)

with tab_overview:
    st.plotly_chart(build_score_chart(visible), use_container_width=True)

    overview_columns = st.columns(4)
    with overview_columns[0]:
        st.metric("Close", f"{summary['close']:,.2f}")
    with overview_columns[1]:
        st.metric("Distance To Trend", format_pct(summary["distance_to_trend"]))
    with overview_columns[2]:
        st.metric("Participation Score", f"{summary['participation_score']:.1f}")
    with overview_columns[3]:
        st.metric("Latest Buy Signal", summary["buy_signal"]["latest_signal_date"] or "n/a")

    signal_history = visible[
        [
            "date",
            "dataset_split",
            f"{core_asset}_close",
            "accumulation_score",
            "signal_zone",
            "buy_signal",
            "buy_signal_strength",
            "valuation_component",
            "macro_component",
            "participation_component",
        ]
    ].copy()
    st.dataframe(signal_history.tail(20), use_container_width=True, hide_index=True)

with tab_structure:
    price_columns = [f"{core_asset}_close", *[f"{asset}_close" for asset in selected_macro_assets]]
    st.plotly_chart(build_overlay_chart(visible, price_columns), use_container_width=True)

    focus_macro = st.selectbox("Macro series for rolling correlation", options=selected_macro_assets)
    st.plotly_chart(build_correlation_chart(visible, focus_macro, correlation_window), use_container_width=True)

with tab_candles:
    ohlc_columns = [f"{core_asset}_open", f"{core_asset}_high", f"{core_asset}_low", f"{core_asset}_close"]
    if all(column in visible.columns for column in ohlc_columns):
        candle_metrics = st.columns(3)
        with candle_metrics[0]:
            st.metric("Buy Signal Count", str(summary["buy_signal"]["signal_count"]))
        with candle_metrics[1]:
            st.metric("Latest Buy Signal", summary["buy_signal"]["latest_signal_date"] or "n/a")
        with candle_metrics[2]:
            st.metric("Latest LSTM Bias", summary["lstm"]["latest_forecast"]["predicted_direction"])

        st.plotly_chart(
            build_candlestick_chart(visible, core_asset, summary["lstm"]["forecast_horizon_periods"]),
            use_container_width=True,
        )

        prediction_column = summary["lstm"]["prediction_column"]
        buy_signal_frame = visible[visible["buy_signal"]].copy()
        if buy_signal_frame.empty:
            st.info("No dates currently satisfy the combined buy-signal gates for the visible timeframe.")
        else:
            st.dataframe(
                buy_signal_frame[
                    [
                        "date",
                        f"{core_asset}_close",
                        "signal_zone",
                        "accumulation_score",
                        prediction_column,
                        "buy_signal_strength",
                    ]
                ].tail(20),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info(
            "Candlestick mode needs `Date`, `Open`, `High`, `Low`, and `Close` in the core CSV. "
            "The rest of the pipeline still works with close-only core data."
        )

with tab_lstm:
    lstm_prediction_column = summary["lstm"]["prediction_column"]
    lstm_actual_column = f"{core_asset}_lstm_actual_forward_return_{summary['lstm']['forecast_horizon_periods']}p"
    lstm_split_column = f"{core_asset}_lstm_sample_split"
    lstm_target_date_column = f"{core_asset}_lstm_target_date_{summary['lstm']['forecast_horizon_periods']}p"

    lstm_metrics_columns = st.columns(3)
    with lstm_metrics_columns[0]:
        st.metric("Latest Forecast", f"{summary['lstm']['latest_forecast']['predicted_forward_return']:.2%}")
    with lstm_metrics_columns[1]:
        st.metric("Validation RMSE", f"{summary['lstm']['metrics']['validation']['rmse']:.4f}")
    with lstm_metrics_columns[2]:
        st.metric("Test RMSE", f"{summary['lstm']['metrics']['test']['rmse']:.4f}")

    history_frame = result.lstm_result.history.copy()
    history_chart = go.Figure()
    history_chart.add_trace(go.Scatter(x=history_frame["epoch"], y=history_frame["loss"], mode="lines", name="Train Loss"))
    history_chart.add_trace(go.Scatter(x=history_frame["epoch"], y=history_frame["val_loss"], mode="lines", name="Validation Loss"))
    history_chart.update_layout(
        height=320,
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        xaxis_title="Epoch",
        yaxis_title="Loss",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.35)",
        legend={"orientation": "h"},
    )
    st.plotly_chart(history_chart, use_container_width=True)

    prediction_frame = dataset.dropna(subset=[lstm_prediction_column]).copy()
    prediction_frame = prediction_frame[prediction_frame[lstm_split_column].isin(["validation", "test"])]
    prediction_chart = go.Figure()
    prediction_chart.add_trace(
        go.Scatter(
            x=prediction_frame["date"],
            y=prediction_frame[lstm_actual_column],
            mode="lines",
            name="Actual Forward Return",
            line={"color": "#132430", "width": 2.5},
        )
    )
    prediction_chart.add_trace(
        go.Scatter(
            x=prediction_frame["date"],
            y=prediction_frame[lstm_prediction_column],
            mode="lines",
            name="Predicted Forward Return",
            line={"color": "#b85b29", "width": 2.5},
        )
    )
    prediction_chart.update_layout(
        height=360,
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        xaxis_title="Prediction Date",
        yaxis_title="Forward Return",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.35)",
        legend={"orientation": "h"},
    )
    st.plotly_chart(prediction_chart, use_container_width=True)

    st.dataframe(
        prediction_frame[
            [
                "date",
                lstm_target_date_column,
                lstm_split_column,
                lstm_actual_column,
                lstm_prediction_column,
            ]
        ].tail(20),
        use_container_width=True,
        hide_index=True,
    )

with tab_drivers:
    driver_frame = pd.DataFrame(summary["drivers"])
    bar_chart = go.Figure()
    bar_chart.add_trace(
        go.Bar(
            x=driver_frame["label"],
            y=driver_frame["latest_contribution"],
            marker_color=["#3d7a4a" if value >= 0 else "#8c3b34" for value in driver_frame["latest_contribution"]],
        )
    )
    bar_chart.update_layout(
        height=380,
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        yaxis_title="Latest Contribution",
        xaxis_title="Macro Asset",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.35)",
    )
    st.plotly_chart(bar_chart, use_container_width=True)

    display_columns = [
        "label",
        "direction",
        "weight",
        "latest_return",
        "latest_scaled_return",
        "latest_contribution",
        "rolling_correlation",
        "thesis",
    ]
    st.dataframe(driver_frame[display_columns], use_container_width=True, hide_index=True)

st.markdown("### Method Notes")
st.write(
    "The score is heuristic. Rising equities and ETH are treated as tailwinds, while rising DXY, VIX, yields, and oil are treated as headwinds. "
    "Gap purge removes rows where macro data stayed stale beyond the configured day threshold. "
    "The dataset is split chronologically into train, validation, and test, scaling is fit on train only, and the LSTM is trained on the scaled sequence features. "
    "Buy markers on the candlestick chart require the core CSV to include OHLC data and fire only when the accumulation zone gates align with a bullish LSTM forward-return forecast."
)
