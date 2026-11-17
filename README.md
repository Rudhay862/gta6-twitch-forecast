# GTA VI Launch-Day Twitch Viewers — Quantile Forecaster

![competition](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)
![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Rudhay862/gta6-twitch-forecast/rank.svg)
![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Rudhay862/gta6-twitch-forecast/reward.svg)

Entry for [Forecast GTA VI's launch-day Twitch viewers hour by hour](https://make.fun/c/gta6-twitch-ccu-forecast-01) (`gta6-twitch-ccu-forecast-01`).

## What it does

`main.py` forecasts Twitch concurrent viewers for GTA VI on its release day
(2026-11-19 UTC), hours 2–24, as five quantiles (5/25/50/75/95%) of each
hour's count, anchored on the known hour-1 print.

**Model — comparables log-ratio shape model with a size-dependent shape:**

1. Keep only past launches whose unlock text says `Tier: comparable`.
2. For each launch, take release-day hourly viewers `v_h` and form
   log-ratios `r_h = ln(v_h / v_1)`.
3. Per forecast hour, fit a weighted linear model of `r_h` on
   `ln(v_1) - ln(p1)` across launches (Gaussian kernel, bandwidth 1.5 in log
   space), then take weighted quantiles of the residuals. The day shape
   depends strongly on launch size — with a midnight-local unlock, hour 1
   only catches New Zealand/Australia, so small launches explode later
   (100x+) while big ones stay roughly flat.
4. Smooth the quantile curves over neighbouring hours; blend toward the
   unweighted global quantiles when the weighted pool is thin; fall back to
   the house baseline shape if no comparables are usable.

Python 3.13, standard library only, no network, deterministic.

## How to run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes
`<id>.out` for each into `$MAKEFUN_OUTPUT_DIR` (else `/output`).

## Backtest

`backtest.py` runs a leave-one-out backtest over the 26 pooled
Tier-comparable launches from the 4 practice cases (each launch forecast
from the other 25, scored against its actual release-day hours 2–24):

```sh
python3 backtest.py
```

Results (LOO, 598 scored hours):

| | mean pinball loss (log) | typical miss | 90% coverage |
|---|---|---|---|
| This model | **0.451** | 90.4% | 0.826 |
| House baseline | 1.084 | — | — |

On the launches most like GTA VI (hour-1 viewers ≥ 50k, 299 hours):
**0.183 vs baseline 0.243**; for hour-1 ≥ 200k: **0.125 vs 0.235**.

## Files

- `main.py` — the entry (single file, stdlib only)
- `backtest.py` — leave-one-out backtest vs the house baseline
- `LICENSE` — MIT
