# Pre-registration log

Each entry records something fixed *before* the data that will judge it
existed. Entries are append-only: never edit an old one, add a new entry instead.
The git commit and GitHub push time of each entry are the timestamps that count.
The dates written in the text are not.

PR-1 and PR-2 were first recorded on 2026-11-25 at 16:22 ET in the private developer
repository, and copied here when this repository was created. The commit hashes below
belong to the private developer implementation and are available on request.
Entries from PR-3 on are recorded here first.

---

## PR-1: Main live strategy (recorded 2026-11-25)

**What was fixed, and when** (commits in the developer implementation):

- Rule: volatility target, exposure = min(1, 15% / trailing 20-day annualized vol),
  rebalanced daily, cash earns 0. First committed on 2026-11-22 14:48 ET (`7986350`) as a comparison sleeve.
- Universe: SPY, IWM, EFA, EEM, VNQ, TLT, IEF, HYG, GLD, DBC, $100 each, independent
  sleeves, benchmark = buy & hold of the same sleeves. Fixed and made the main strategy
  on **2026-11-23 17:11 ET** (`5fa00cf`).
- Costs reported: 3 bps of the amount traded (dashboard toggle, on by default).

**What counts as out-of-sample evidence:**

- Only daily returns **from the 2026-11-23 close to the 2026-11-24 close onward**.
  The `5fa00cf` commit came after the 2026-11-23 close, so strictly even that day's return was already known.
- Rows dated 2026-08-27..2026-11-20 in the live record are flagged `backfilled`. They are **in-sample**.
- Every backtest of this rule on 2010–2026-08 data is in-sample. The rule was picked as the
  best of 7 timing rules on that period (see [`MODEL_SPEC.md`](MODEL_SPEC.md) §8).

**Known caveats at registration:** The developer documentation says the universe was chosen for
diversification, not by backtest performance, and the developer implementation has no record that would
confirm or refute that. The 15% / 20-day / 1× parameters were round numbers and were never grid-searched.

**Planned evaluation:** None pre-committed yet. Add the test, horizon and decision
rule (e.g. "after N trading days, compare Sharpe and max drawdown with buy & hold, using X")
as a new entry *before* looking at the forward data.

---

## PR-2: ML side experiment (recorded 2026-11-25)

Unchanged since 2026-08-27 (LightGBM next-day model, 10 tickers, threshold 0.5,
weekly retrain). Its live record from 2026-08-27 on is out-of-sample for the model.
Its parameters were not changed after its live results were seen.
