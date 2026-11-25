# Model documentation (developer's spec for independent validation)

**Status:** handed to the validator on 2026-11-25.
**Developer implementation:** the original research pipeline and live paper-trading
system, built with AI assistance (Claude Code) in Aug–Nov 2026. It is kept in private
repositories and is referred to here as the *developer implementation*.
**Validator:** this repository. It re-implements the evaluation from this spec alone,
without copying developer code.

The spec says what the models do, not how well they did. **Blindness protocol:** the
validator does not read the developer implementation's results or run its code until
the validator's own numbers are committed and tagged. A few headline numbers had
already been seen before validation began. The report says which ones.

Every choice tagged **[A]** was arbitrary (picked without testing alternatives).
Every choice tagged **[T]** was picked or kept *after* seeing results. These are
inputs to the validation report's challenge section, not admissions that they're wrong.

---

## 1. Data

| Item | Spec |
|---|---|
| Source | yfinance `download(..., auto_adjust=True)`: open/high/low/close are split- and dividend-adjusted; volume is as reported. Lower-case column names; drop rows with any NaN. |
| Research study (§6.1) | Tickers SPY, QQQ, SCHD, VXUS, O, BRK-B, AAPL. `start="2010-01-01"`, `end="2026-07-31"` (yfinance `end` is exclusive). |
| Live-config study (§6.2) and timing rules (§8) | Tickers SPY, QQQ, SCHD, AAPL, VZ, VXUS, VT, SPHD, BRK-B, CAT. Start 2010-01-01. Use only rows dated before 2026-08-27 (the first live day). |
| Main-strategy universe (§8) | SPY, IWM, EFA, EEM, VNQ, TLT, IEF, HYG, GLD, DBC. |
| Reproducibility | yfinance revises adjusted history, so re-downloads differ slightly. Download once, save a snapshot with the download date in its filename, and run everything from that snapshot. Expect small differences from the developer's numbers for this reason alone. |

Tickers with shorter histories (SCHD 2011, VXUS 2011, SPHD 2012) start at their first available bar. **[A]** All universes were chosen in 2026 from ETFs that still exist (survivorship/hindsight).

## 2. Features (all computed from data through day *t*'s close)

`close`, `high`, `low`, `volume` are day *t*'s values. All rolling windows are trailing and include day *t*.

| Name | Definition |
|---|---|
| `ret_1d`, `ret_5d`, `ret_20d` | `close / close.shift(k) - 1`, k = 1, 5, 20 |
| `ma5`, `ma20`, `ma50` | `close / rolling_mean(close, k) - 1` |
| `realized_vol_10d`, `realized_vol_20d` | rolling sample std (ddof=1) of `ret_1d`, 10 / 20 days, **not** annualized |
| `high_low_range` | `(high - low) / close` |
| `rsi_14` | `100 - 100 / (1 + G/L)`, where G and L are 14-day **simple** rolling means of positive `close.diff()` and of the absolute value of negative `close.diff()` (not Wilder's smoothing) |
| `vol_ratio` | `volume / rolling_mean(volume, 20)` |

**[A]** The feature set, the windows and the use of simple-mean RSI.

## 3. Label

`target_t = 1` if `close_{t+1} > close_t`, else 0. The next-day return is
`ret_next_t = close_{t+1} / close_t - 1`. A row is usable for training only
if `close_{t+1}` actually exists. Rows missing any feature (the first ~50 days) are dropped.

## 4. Model

LightGBM `LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.03,
subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=1)`. Other settings are library defaults.
Output: `p_up = predict_proba[:, 1]`. One model per ticker, no pooling across tickers.
**[A]** Hyperparameters were never tuned. Note that `subsample` has no effect
unless `subsample_freq > 0` (library default is 0). This holds for the developer code as well.

## 5. Decision rule and P&L accounting

- Signal at day *t*'s close: **long** if `p_up_t > threshold`, otherwise **cash**. No shorting.
- Day *t*'s strategy return = `position_t × ret_next_t`. The position is set
  and earns at the **same close** that the features used. **[A]** This assumes you can trade at the close the signal was computed from.
- Cash earns **0%**. **[A]**
- Live threshold: 0.5. Research threshold grid: {0.2, 0.3, 0.5}. **[A]**

## 6. Walk-forward validation

### 6.1 Research study

- First test block starts at (first usable date + 5 years). Test blocks are 6
  calendar months long, half-open `[start, start + 6M)`. Blocks are back to back.
- Training: **expanding window**, all usable rows strictly before the block start.
- Stop when a block would have fewer than 0.9 × 126 trading days (the final partial block is dropped)
  or the training set has fewer than 250 rows.
- Retrain once per block.

### 6.2 Live-config replica

- Out-of-sample days run from 2015-01-01 to the last day before 2026-08-27.
- OOS days are split into consecutive blocks of 21 **trading** days.
  For each block, train on all usable rows dated before the block's first day,
  **then also drop the last of those rows**. That row's label uses the block's first
  close, so dropping it is deliberately conservative.
- Expanding window from 2010.
- **[A]** The live dashboard retrains every 7 calendar days, not every 21 trading days.
  The replica uses 21 to save compute.

## 7. Baselines and evaluation metrics

| Metric | Definition |
|---|---|
| Accuracy | share of OOS days where `(p_up > 0.5) == (ret_next > 0)` |
| Always-up baseline | share of OOS days with `ret_next > 0`. A day with exactly zero return counts as "not up". |
| Rank IC | Spearman correlation of `p_up` with `ret_next`, computed per ticker and pooled across all ticker-days |
| Share invested | mean of position |
| Switches/yr | `sum(|Δposition|) / n_days × 252` |
| CAGR | `equity_end ^ (252 / n_days) - 1` |
| Sharpe | `mean(daily r) / std(daily r) × √252`, risk-free = 0, days in cash included as r = 0 |
| Max drawdown | `min(equity / running_max(equity) - 1)` |
| Edge | strategy cumulative return − buy & hold cumulative return, over the same days |
| Edge excl. top N | sort days by daily edge (`strategy_r − bh_r`) descending, drop the N best days from **both** series, recompute Edge. N ∈ {1, 3}. |
| Edge concentration | the share of the sum of positive daily edges coming from the top {1, 3, 5, 10, 20, 50} days |
| Random timing | For each ticker, permute its sequence of daily positions uniformly at random (same number of long days, random dates). Recompute the final value of a $100 sleeve per ticker and sum across tickers. Repeat (developer: 3,000 offline / 5,000 on the dashboard, seed 0). Report the share of permutations with value ≥ the actual strategy's. Costs are charged per switch if costs are on. **[A]** Permutation is i.i.d. across days. |
| 57-day windows | rolling 57-trading-day windows, step 5 days, equal-weight across the 10 sleeves. 57 was the length of the live record at the Nov 2026 review. **[T]** |

## 8. Timing rules and the volatility target

Per-ticker exposure `e_t` in [0, 1], decided at day *t*'s close and applied to `ret_next_t`.

| Rule | Exposure |
|---|---|
| Buy & hold | 1 |
| Live ML rule | `1[p_up > 0.5]` |
| ML, smoothed | 5-day mean of `p_up`. Enter if > 0.51, exit if < 0.47, otherwise keep the previous state. **[T]** |
| 200d trend (daily) | `1[close > rolling_mean(close, 200)]` |
| 200d trend (monthly) | the same, evaluated on the month's last trading day and held for the next month |
| **Volatility target** | `min(1, 0.15 / σ_t)`, where `σ_t` = sample std of the last 20 daily close-to-close returns × √252. If σ is not finite or ≤ 0, exposure = 1. **Rebalanced daily.** |
| Trend × vol target | monthly trend × vol target |

**[A]** 15% target, 20-day window, 1× cap: picked as round numbers and never grid-searched.
**[T]** The vol target was then *selected* as the main live strategy because it had the best
risk-adjusted result among these **7 rules** on 2015–2026 data. Count all 7 as trials.

**Portfolio aggregation (offline comparison):** equal weight across tickers, **rebalanced
daily**. The portfolio's daily return is the mean of the sleeves' net daily returns.
**The live dashboard instead runs independent $100 sleeves with no rebalancing between them.**

**Periods:** full 2015-01-01..2026-08-26; development 2015-01-01..2024-08-31;
"holdout" 2024-09-01..2026-08-26. **[T]** The timing-rules holdout was not
pre-registered. Treat all of 2015–2026 as in-sample for these rules.

## 9. Transaction costs

3 bps (0.03%) of the amount traded. **[A]** No spread, borrow or tax modelling. There are two implementations:

- **Offline timing rules:** `cost_t = 0.0003 × |e_t − e_{t−1}|`, charged on day *t*. No
  adjustment for drift. The first day's entry is not charged.
- **Live dashboard:** the vol target charges `|e_t − e_{t−1}^drifted|`, where
  `e^drifted = e_{t−1}(1+r_t) / (1 + e_{t−1} r_t)`. The ML rule charges 1 per switch. Buy & hold charges
  1 at entry. The initial buy is charged everywhere. Costs compound on the sleeve value.

The original research study (§6.1) has **no** costs.

## 10. Live forward test

- Main strategy: vol target (§8) on the main-strategy universe, $100 per ETF, independent
  sleeves. Values for 2026-08-27..2026-11-20 are **backfilled** by applying the rule to past
  closes. See [`PREREGISTRATION.md`](PREREGISTRATION.md) for which data counts as out-of-sample.
- ML experiment: the §4 model on the live-config tickers, threshold 0.5, retrained every 7
  days on all history, $100 per ticker, independent sleeves. Live since 2026-08-27. Two days are missing
  (2026-09-21 and 2026-10-02).

## 11. Not in this spec

The developer implementation also contains a cross-sectional ETF-ranking study with
its own pre-registration. It is out of scope for the first replication pass.
