# BTC Macro Snapshot

- As of: 2024-01-19
- Signal zone: Macro Headwind
- Accumulation score: 33.09
- Valuation score: 22.57
- Macro score: 47.2
- Participation score: 35.26
- Drawdown from peak: -38.40%
- Distance to long trend: 31.42%
- Gap purge: 0 stale day(s), 1856 rows removed
- Chronological split: train 69.97%, validation 14.98%, test 15.05%
- Scaler fit split: train
- LSTM forecast horizon: 30 periods
- LSTM latest forecast: 25.77% (Bullish)
- Buy signals: 48 total, latest at 2022-12-05

As of 2024-01-19, BTC sits in the 'Macro Headwind' zone with an accumulation score of 33.1/100. Current valuation stress is 22.6, macro support is 47.2, and participation is 35.3. Gap purge is active at 0 stale day(s), removing 1856 row(s). The LSTM forecasts the next 30 aligned period(s) from a 60-step history window. The dataset is split chronologically into train/validation/test at 69.97%/14.98%/15.05%. Combined buy signals have triggered 48 time(s). Leading tailwinds: NASDAQ, S&P 500, US Dollar Index. Leading headwinds: none.

## Dataset Splits

- train: 1088 rows, 2017-11-10 -> 2022-03-10
- validation: 233 rows, 2022-03-11 -> 2023-02-13
- test: 234 rows, 2023-02-14 -> 2024-01-19

## LSTM Metrics

- train: count 999, mae 0.0609, rmse 0.0782, direction 93.19%
- validation: count 233, mae 0.1564, rmse 0.1847, direction 72.10%
- test: count 234, mae 0.2315, rmse 0.3026, direction 51.71%

## Macro Drivers

- NASDAQ: contribution 1.241, corr 0.049, thesis: Growth-heavy equity strength often aligns with higher crypto beta demand.
- S&P 500: contribution 0.891, corr 0.031, thesis: Higher equity risk appetite often coincides with stronger crypto flows.
- US Dollar Index: contribution 0.771, corr 0.025, thesis: A stronger dollar typically tightens global liquidity and pressures crypto multiples.
- VIX: contribution 0.729, corr -0.163, thesis: Rising volatility stress usually reflects falling risk appetite across markets.
- Gold: contribution 0.127, corr -0.198, thesis: Gold can act as a partial store-of-value proxy, but the relationship is weaker than equities or DXY.
- Brent Crude: contribution 0.126, corr -0.224, thesis: Higher oil can feed inflation pressure and keep policy tighter for longer.
- ETH: contribution 0.079, corr 0.731, thesis: ETH often behaves as a higher-beta crypto proxy for risk appetite inside the asset class.
- WTI Crude: contribution 0.029, corr -0.228, thesis: Higher oil can feed inflation pressure and keep policy tighter for longer.
- US 2Y Yield: contribution 0.006, corr 0.135, thesis: Higher front-end yields often signal tighter policy and a less supportive macro backdrop.
- US 10Y Yield: contribution 0.004, corr 0.076, thesis: Higher long-end yields can pressure long-duration and liquidity-sensitive assets.