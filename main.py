#!/usr/bin/env python3
"""Quantile forecaster for GTA VI launch-day Twitch viewers.

Model: comparables log-ratio shape model with a size-dependent shape.
  - Keep only past launches whose unlock text says "Tier: comparable".
  - For each launch, take its release-day hourly viewers v_h (h = 1..24) and
    form log-ratios r_h = ln(v_h / v_1).
  - Per forecast hour h, fit a weighted linear model of r_h on
    x = ln(v_1) - ln(p1) across launches (Gaussian kernel weights,
    bandwidth SIGMA in log space), then take weighted quantiles of the
    residuals around the fitted line.  The day shape depends strongly on
    launch size: with a midnight-local unlock, hour 1 only catches
    NZ/Australia, so small launches explode later (100x+) while big ones
    stay roughly flat.  The linear fit captures that dependence while
    pooling all launches for the spread.
  - The quantile curves are smoothed over neighbouring hours; when the
    weighted pool is thin, the forecast blends toward the unweighted
    global quantiles and widens the log-spread a little around the
    median (the residual spread is understated when few launches
    carry weight).  q_tau(h) = p1 * exp(quantile_tau(r_h)).
  - Falls back to the house baseline shape if no comparables are usable.

Standard library only. Deterministic. No network.
"""

import glob
import json
import math
import os
import sys
from datetime import datetime

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard-normal quantiles for the house-baseline fallback.
Z = {0.05: -1.6448536269514722, 0.25: -0.6744897501960817,
     0.5: 0.0, 0.75: 0.6744897501960817, 0.95: 1.6448536269514722}

SIGMA = 1.5          # kernel bandwidth in log(v1) space
SHRINK = 1.0         # ridge-style shrinkage on the per-hour slope
SPREAD_THIN = 0.6    # extra log-spread when the weighted pool is thin
MIN_EFF_N = 6.0      # below this effective n, blend toward global quantiles
MAX_Q = 1_000_000_000.0


def parse_ts(s):
    """Parse an ISO-8601 timestamp (trailing Z allowed) to a datetime."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def release_day_series(comp):
    """Return {hour: viewers} for the launch's release day (hours 1..24).

    Returns None when hour 1 is missing or unusable.
    """
    try:
        day_start = parse_ts(comp["day_start"])
    except Exception:
        return None
    out = {}
    for entry in comp.get("hourly") or []:
        try:
            t = parse_ts(entry["from"])
            viewers = float(entry["viewers"])
        except Exception:
            continue
        h = int((t - day_start).total_seconds() // 3600) + 1
        if 1 <= h <= 24 and viewers >= 0:
            out[h] = viewers
    if 1 not in out or out[1] < 1:
        return None
    return out


def weighted_quantile(pairs, tau):
    """Weighted quantile of values.

    pairs: list of (value, weight) with weight > 0. Linear interpolation on
    the weighted cumulative distribution.
    """
    pairs = sorted((v, w) for v, w in pairs if w > 0)
    if not pairs:
        return None
    total = sum(w for _, w in pairs)
    if total <= 0:
        return None
    target = tau * total
    cum = 0.0
    prev_v, prev_cum = pairs[0][0], 0.0
    for v, w in pairs:
        cum += w
        if cum >= target:
            if cum == w:  # first point
                return v
            frac = (target - prev_cum) / w if w > 0 else 0.0
            return prev_v + frac * (v - prev_v)
        prev_v, prev_cum = v, cum
    return pairs[-1][0]


def _quantiles_for_hour(ratios, tau_list):
    """ratios: list of (log_ratio, weight). Returns {tau: log-quantile}."""
    out = {}
    for tau in tau_list:
        q = weighted_quantile(ratios, tau)
        out[tau] = q
    return out


def forecast(p1, series_list):
    """Forecast hours 2..24.

    p1: known hour-1 viewers (>= 1).
    series_list: list of {hour: viewers} dicts for comparable launches.
    Returns {hour: [q05, q25, q50, q75, q95]} (floats, rounded to 2 dp).
    """
    p1 = max(float(p1), 1.0)
    log_p1 = math.log(p1)

    clean = [s for s in series_list if 1 in s and s[1] >= 1]
    if not clean:
        return _baseline_forecast(p1)

    log_q = {}   # hour -> {tau: log-quantile of the ratio}
    for h in range(2, 25):
        pts = []
        for s in clean:
            if h in s and s[h] > 0:
                x = math.log(s[1]) - log_p1
                y = math.log(s[h] / s[1])
                w = math.exp(-0.5 * (x / SIGMA) ** 2)
                pts.append((x, y, w))
        if not pts:
            # No launch has this hour: fall back to the baseline shape.
            log_q[h] = {tau: Z[tau] * 0.15 * math.sqrt(h - 1)
                        for tau in LEVELS}
            continue
        sw = sum(w for _, _, w in pts)
        mx = sum(w * x for x, _, w in pts) / sw
        my = sum(w * y for _, y, w in pts) / sw
        sxx = sum(w * (x - mx) ** 2 for x, _, w in pts)
        sxy = sum(w * (x - mx) * (y - my) for x, y, w in pts)
        b = (sxy / sxx) if sxx > 1e-12 else 0.0
        # Shrink the slope toward 0 when the weighted x-spread is thin.
        b *= sxx / (sxx + SHRINK)
        a = my - b * mx
        resid = [(y - (a + b * x), w) for x, y, w in pts]
        q = {}
        for tau in LEVELS:
            rq = weighted_quantile(resid, tau)
            q[tau] = a + (rq if rq is not None else 0.0)
        # If the weighted pool is very thin, blend toward the unweighted
        # global quantiles of the raw ratios.
        sw2 = sum(w * w for _, _, w in pts)
        n_eff = (sw * sw / sw2) if sw2 > 0 else 0.0
        if n_eff < MIN_EFF_N:
            grow = [(math.log(s[h] / s[1]), 1.0) for s in clean
                    if h in s and s[h] > 0]
            gq = _quantiles_for_hour(grow, LEVELS)
            alpha = n_eff / MIN_EFF_N
            for tau in LEVELS:
                if gq[tau] is not None:
                    q[tau] = alpha * q[tau] + (1.0 - alpha) * gq[tau]
            # Thin pool: the residual spread is understated, so widen the
            # log-spread around the median, most when the pool is thinnest.
            # (A global widening was tested and rejected: it hurts the
            # big-launch subset that the final case resembles.)
            k = 1.0 + SPREAD_THIN * (1.0 - alpha)
            med = q[0.5]
            for tau in LEVELS:
                q[tau] = med + k * (q[tau] - med)
        log_q[h] = q

    # Smooth each quantile curve over neighbouring hours (3-point average).
    for tau in LEVELS:
        vals = [log_q[h][tau] for h in range(2, 25)]
        smoothed = []
        for i, v in enumerate(vals):
            lo = vals[i - 1] if i > 0 else v
            hi = vals[i + 1] if i < len(vals) - 1 else v
            smoothed.append(0.25 * lo + 0.5 * v + 0.25 * hi)
        for i, h in enumerate(range(2, 25)):
            log_q[h][tau] = smoothed[i]

    out = {}
    for h in range(2, 25):
        qs = []
        for tau in LEVELS:
            q = p1 * math.exp(log_q[h][tau])
            q = min(max(q, 0.01), MAX_Q)
            qs.append(round(q, 2))
        # Enforce non-decreasing quantiles.
        qs.sort()
        out[h] = qs
    return out


def _baseline_forecast(p1):
    """House baseline: flat at p1, spread widening 0.15*sqrt(h-1)."""
    out = {}
    for h in range(2, 25):
        qs = []
        for tau in LEVELS:
            q = p1 * math.exp(Z[tau] * 0.15 * math.sqrt(h - 1))
            qs.append(round(min(max(q, 0.01), MAX_Q), 2))
        qs.sort()
        out[h] = qs
    return out


def run_case(case):
    """Build the forecast object for one parsed case dict."""
    known = case.get("known") or []
    p1 = 1.0
    for row in known:
        if row.get("hour") == 1:
            try:
                p1 = float(row.get("viewers") or 1.0)
            except Exception:
                p1 = 1.0
            break
    if p1 < 1:
        p1 = 1.0

    series_list = []
    for comp in case.get("comparables") or []:
        unlock = comp.get("unlock") or ""
        if "Tier: comparable" not in unlock:
            continue
        s = release_day_series(comp)
        if s is not None:
            series_list.append(s)

    fc = forecast(p1, series_list)
    hours = [{"hour": h, "q": fc[h]} for h in range(2, 25)]
    return {"hours": hours}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(in_dir, "*.in")))
    for path in files:
        case_id = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r", encoding="utf-8") as f:
            case = json.load(f)
        result = run_case(case)
        out_path = os.path.join(out_dir, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, separators=(",", ":"))
            f.write("\n")


if __name__ == "__main__":
    main()
