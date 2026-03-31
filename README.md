# Macro-Driven Crypto Analysis Tool

Decision-support toolkit for studying how macro regimes influence long-term crypto accumulation opportunities.

This rebuild keeps the original `data/` folder and replaces the rest of the project with a focused workflow:

1. Validate and standardize `core/` + `macro/` CSV files.
2. Align every macro series to the core asset timeline.
3. Forward-fill macro gaps, track staleness, and purge stale macro rows.
4. Compute returns and core volume change.
5. Assign a chronological train/validation/test split.
6. Scale engineered features after return calculation, fitting the scaler on train only.
7. Train an LSTM on the scaled time-series feature windows.
8. Estimate macro tailwinds/headwinds, drawdown context, an accumulation score, and a buy-signal overlay.
9. Export processed datasets and inspect them in a Streamlit dashboard with upload support.

## Project Structure

```text
data/
  core/         # core asset CSVs, expected columns: Date, Close, Volume, optional Open/High/Low
  macro/        # macro indicator CSVs, expected columns: Date, Close
  processed/    # generated outputs from the CLI
app/
  streamlit_app.py
src/
  macro_driven_crypto_analysis/
tests/
```

## Input Format

The loader is intentionally generic:

- It accepts uppercase or mixed-case headers such as `Date`, `Close`, `Adj Close`, `Volume`.
- For the core asset, `Open`, `High`, and `Low` are optional but recommended if you want the candlestick buy-signal chart.
- It automatically drops the extra Yahoo ticker row often found on line 2.
- It parses `date`, coerces price/volume columns to numeric, sorts by date, forward-fills later during alignment, and removes invalid rows.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Build processed artifacts:

```bash
python -m macro_driven_crypto_analysis.cli --core btc --scaler standard --macro-gap-purge-days 0
```

Run the dashboard:

```bash
streamlit run app/streamlit_app.py
```

## Dashboard Upload Flow

The deployed Streamlit app now supports two ingestion modes:

- `Workspace data folders`: read CSV files already stored in `data/core` and `data/macro`.
- `Upload CSV files`: upload one core CSV and one or more macro CSVs directly in the browser, then process them immediately without manually copying files into the repo.

Upload mode infers asset slugs from filenames, validates the files with the same cleaning logic used by the CLI, then runs the full pipeline in memory before showing the charts.

## CLI Output

Running the CLI writes these files into `data/processed/`:

- `*_analysis_dataset.csv`: full aligned dataset with returns, scaled features, rolling correlations, and signal columns
- `*_train.csv`, `*_validation.csv`, `*_test.csv`: chronological modeling splits
- `*_signal_history.csv`: compact history for the accumulation score plus buy-signal flags
- `*_lstm_predictions.csv`: per-date LSTM predictions and actual forward returns
- `*_lstm_history.csv`: training and validation loss history
- `*_lstm_metrics.json`: regression metrics by split plus latest forecast
- `*_summary.json`: latest snapshot and driver breakdown
- `*_snapshot.md`: short human-readable report

Default split ratios are chronological `70 / 15 / 15` for `train / validation / test`.
Default gap purge is `0` stale days, which means the analytical dataset keeps only rows where every selected macro series has a fresh observation on that date. Increase the threshold if you want to tolerate longer forward-filled macro gaps.

## LSTM Training

The model stage uses an `LSTM` regression model trained on scaled sequence windows of the engineered return features.

Default assumptions in this rebuild:

- input window: `60` aligned periods
- forecast target: core asset `forward return` over the next `30` aligned periods
- training split: chronological `70 / 15 / 15`
- early stopping: validation loss with restored best weights

You can override these values from the CLI.

## Accumulation Framework

The score is heuristic, not a prediction model.

It combines:

- valuation stress in the core asset
- macro support or macro headwind from aligned indicators
- participation via core volume change

Default directional assumptions:

- Tailwind when rising: `sp500`, `nasdaq`, `eth`
- Headwind when rising: `dxy`, `vix`, `us10y`, `us2y`, `oil_wti`, `oil_brent`
- Mildly supportive when rising: `gold`

Unknown macro files are still loaded automatically with a neutral-positive default so you can extend the dataset without rewriting code.

## Models In This Rebuild

- `LSTM regression model`: sequence model trained on scaled return features.
- `Accumulation score model`: a rule-based composite model built from valuation, macro support, and participation.
- `StandardScaler`: available for return-based feature normalization.
- `MinMaxScaler`: available as an alternative normalization mode for overlays and bounded comparisons.

This rebuild currently uses `LSTM` as the predictive model. It does not include `XGBoost` or `LightGBM`.

## Dashboard Highlights

- browser upload flow for new `core` and `macro` CSVs
- accumulation score timeline
- Japanese candlestick chart with buy-signal markers when core OHLC data is available
- normalized price overlays
- rolling correlation view for any macro series
- latest macro contribution ranking
- processed driver table for long-term bottom hunting

## Notes

- The score is intended for long-term context, not short-term trading.
- Scaling is always applied after return engineering.
- Gap purge is applied before split/scaling so stale macro prints do not leak into train, validation, or test.
- The scaler is fit on the training split only, then applied to validation and test.
- The LSTM target is the core asset forward return over the configured horizon, measured in aligned periods.
- The current buy signal requires both an accumulation-zone setup and a bullish LSTM forward-return forecast.
- Adding a new CSV to `data/core/` or `data/macro/` makes it available automatically.
