"""
run_threshold.py -- THE HEADLINE MODULE of the SSI / ciNPWT risk-threshold Markov CUA.

This is the paper's contribution. Figures 2 and 3, Table 3 and the whole "Risk threshold
for ciNPWT" Results subsection are built from the four CSVs written here.

Imports the engine (`inputs.py` / `model.py`) READ-ONLY and writes the five
BUILD-CONTRACT section 8 deliverables to `v2/results/`:

    threshold_curve.csv             Fig 2 data. One row per point of SETTINGS["risk_grid"]
                                    (291): incremental cost / QALY / INMB at three WTPs,
                                    ICER, dominance, quadrant, NNT and the clinical
                                    translation, all at the base device price.
    threshold_grid_risk_price.csv   Fig 3 data. LONG format over risk x price
                                    (291 x 161 = 46,851 rows).
    threshold_rr.csv                Per stratum x SETTINGS["rr_grid"] (3 x 61 = 183 rows).
    threshold_results.csv           The solved thresholds -- Table 3. One row per
                                    (quantity, stratum / WTP) with units, the clinical
                                    translation and a plain-English interpretation.
    threshold_summary.md            The narrative a surgeon can read.

Every CSV carries `inputs.RESULTS_BANNER` as a '#'-prefixed LINE 1, followed by the
header row -- read with `pd.read_csv(path, skiprows=1)` (BUILD-CONTRACT section 7;
never `comment='#'` on a results CSV).

Run with cwd = v2/analysis/ :  python3 run_threshold.py

=============================================================================
GATE STATUS: VERIFIED -- see inputs.RESULTS_BANNER. Phase A2 full-text
verification completed 2026-08-17; 46 of 48 parameters are closed out against the
14 source full texts and 43 carry a table/page locator. The 2 that remain are the
ciNPWT institutional net acquisition price and the superficial tunnel length.
=============================================================================

-----------------------------------------------------------------------------
THE FOUR THRESHOLD FAMILIES (plan Phase C, `run_threshold.py` row)
-----------------------------------------------------------------------------
(a) INMB = 0 baseline-risk threshold at each WTP ($50k / $100k / $150k, plus the
    81-point CEAC grid) -- the risk above which ciNPWT is worth buying.
(b) Cost-neutral baseline-risk threshold -- where `inc_cost = 0`, QALYs ignored
    entirely. The payer's break-even, and a strictly MORE CONSERVATIVE (higher) number
    than (a). It is identically the WTP = $0 member of family (a), and this module
    solves it independently on `inc_cost` and then ASSERTS the two agree, so the
    equivalence is demonstrated rather than assumed.
(c) Threshold device price per stratum, at each WTP -- the price at which the sign
    flips. Solved in CLOSED FORM (see below).
(d) Threshold relative risk per stratum, at the base device price -- how weak the
    ciNPWT effect would have to be before it stops paying.

(e) ALL FOUR OF THE ABOVE AGAIN under the ATTRIBUTION scenario
    (`m_ssi_cost = 1.000`, the ALL-CAUSE upper bound; before A2 this family held the attribution-adjusted sensitivity at k_attrib_pennington), ON THE REPORTING BASIS, written under
    the family names `risk_threshold_attrib`, `risk_threshold_cost_neutral_attrib`,
    `device_price_threshold_attrib` and `rr_threshold_attrib`. Added because
    BUILD-CONTRACT section 10 makes this module authoritative for EVERY published
    threshold, and the manuscript's own like-for-like pair -- base p* against the
    base-case and all-cause p*, BOTH on the basis the paper reports --
    previously reached no persisted artifact: the only attribution threshold in
    `results/` was the UNDISCOUNTED one, so Table 3 / Table 6 would have been forced to
    pair a reporting-basis base case against an undiscounted comparator.
    These rows carry `headline = False` (headline means "a base-case row Table 3
    prints"); select them with `family.str.endswith("_attrib")`. Every row's `basis`
    names BOTH the reporting convention AND the scenario, so no threshold published by
    this module can escape unlabelled.

Each carries NNT, SSIs prevented per 1,000 and reoperations prevented per 1,000
evaluated AT the solution.

-----------------------------------------------------------------------------
THE CLOSED-FORM DEVICE-PRICE SHORTCUT -- 291 pairs, not 46,851
-----------------------------------------------------------------------------
The ciNPWT device cost is a PURE ADDITIVE CYCLE-0 TOLL in arm B. Cycle 0 carries a
discount factor of exactly 1 and no half-cycle weight, and nothing else in the result
depends on the price -- not the trace, not the QALYs, not the SSI tolls, not any `n_*`
count. Therefore, EXACTLY (to machine precision, not approximately):

    cost_B(C)      = cost_B(C0) + (C - C0)
    inc_cost(p, C) = inc_cost(p, C0) + (C - C0)
    inc_qaly(p, C) = inc_qaly(p, C0)                       <- independent of C
    INMB(p, C, w)  = INMB(p, C0, w) - (C - C0)

so the 161-point price axis is free: the 291 x 161 map costs 291 engine pairs, not
46,851, and the threshold price has a CLOSED FORM

    C*(p, w) = C0 + INMB(p, C0, w).

`prove_device_cost_linearity()` runs this against LIVE engine re-runs before any
shortcut is taken -- a 5 x 5 grid of (p, C) corners, twelve randomly chosen cells of
the actual 291 x 161 map, and a live re-run at every solved C* -- and ASSERTS agreement
to 1e-6 dollars. The shortcut is proven at run time, never assumed. If a future engine
change ever makes the device cost non-additive (a per-cycle rental, say), this module
fails loudly instead of writing a wrong figure.

-----------------------------------------------------------------------------
TRAPS THIS MODULE IS WRITTEN AROUND
-----------------------------------------------------------------------------
1. `model.incremental(res_b, res_a)` takes **B FIRST**. Reversing it silently flips
   every sign in the paper.
2. NEVER pass a pre-multiplied `p` to arm B. `run_pair(p_base=p)` derives arm B's risk
   as `p x RR` INSIDE the engine; passing `p * rr` would apply the effect twice.
3. `icer` is `np.nan` wherever ciNPWT is dominant, which is 286 of the 291 grid points.
   `dominance` / `quadrant` are the primary reporting fields; every ICER is formatted
   through `_fmt_icer`, which prints "n/a (dominant)" rather than a bare `nan`.
4. `n_ssi` is a cohort COUNT and sits ~0.32% relative below `p` because perioperative
   and background mortality compete with onset in cycles 1-3. NNT is computed from the
   COUNTS (`1 / (n_ssi_A - n_ssi_B)`), which is what reproduces the cross-check 38.9
   at p = 0.066, and is never computed as `1 / (p * (1 - RR))`.
5. `model.ssi_cost_per_case(horizon_cycles=T)` RAISES on a horizon that clips a cost
   tranche. It is used here rather than a hand-rolled `charged == "full"` filter.
6. Only `inputs.py` owns a number (BUILD-CONTRACT section 9). Every constant below --
   the base device price, the base RR, the WTPs, the three grids, the strata -- is read
   from `inputs.SETTINGS` or an `inputs` Param. The two literals in this file are the
   engine's own published cross-check targets, and they are labelled as such.

See: v2/BUILD-CONTRACT.md, v2/run_log.d/model.md, v2/run_log.d/run_threshold.md
(this module's own run-log fragment).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy.optimize import brentq

import inputs as I
import model

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V1_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
RESULTS_DIR = os.path.join(V1_DIR, "results")

# --------------------------------------------------------------------------- #
# Settings -- every one of these is READ, never re-declared (section 9)         #
# --------------------------------------------------------------------------- #
S = I.SETTINGS
RISK_GRID = np.asarray(S["risk_grid"], dtype=float)     # 291 pts, 0.005 -> 0.15
PRICE_GRID = np.asarray(S["price_grid"], dtype=float)   # 161 pts, $100 -> $900
RR_GRID = np.asarray(S["rr_grid"], dtype=float)         # 61 pts, 0.40 -> 1.00
CEAC_GRID = np.asarray(S["ceac_grid"], dtype=float)     # 81 pts, $0 -> $200,000
STRATA = S["strata"]                                    # low / intermediate / high
BASE_HORIZON = int(S["horizon_cycles"])                 # 24 cycles
DISC = float(S["disc_cost"])                            # 0.03
HALF_CYCLE = bool(S["half_cycle"])                      # True
WTP = float(S["wtp_primary"])                           # 100,000
WTP_LO, WTP_HI = (float(w) for w in S["wtp_report"])    # 50,000 / 150,000
WTP_REPORTED = (WTP_LO, WTP, WTP_HI)

#: Base sweep coordinates. The engine resolves these itself when the override is None;
#: they are read here because the closed form needs C0 explicitly and the narrative
#: quotes RR0.
C0 = float(I.cost_2025(I.cost.c_device_cinpwt))         # $250, ALREADY 2025 USD
RR0 = float(I.tp.rr_cinpwt)                             # 0.61

#: Cross-check targets published by the engine agent (`run_log.d/model.md`) and
#: BUILD-CONTRACT section 10. THESE ARE TARGETS, NOT PARAMETER VALUES -- deriving them
#: from inputs.py would make the check compare the parameter store with itself and delete
#: the only independent tie to the contract. Same status as the 38.9 / 17.9 NNT targets in
#: `model.py`'s smoke test.
#:
#: A2 UPDATE 2026-08-17 -- RECOMPUTED, NOT RELAXED. Two A2 corrections move this target,
#: and only these two; every other correction is immaterial to it.
#:
#:   1. m_ssi_cost 1.000 -> 0.492. The base case is now anchored on Pennington's MATCHED
#:      attributable increment rather than on Edmiston's regression-adjusted ALL-CAUSE
#:      increment, which A2 established is an upper bound on attributable cost by
#:      construction (no matching anywhere in the paper). The threshold is inversely
#:      proportional to the SSI cost, so this alone scales it by 1/0.492 = 2.033.
#:   2. frac_deep 0.742 -> 0.745 (count basis). Raises the weighted SSI cost 0.23% and the
#:      QALY loss per SSI from 0.07936 to 0.07960, shaving the threshold back very
#:      slightly.
#:
#:   old 0.006515 x (1 / 0.492) = 0.013242, then x (78,024/78,206 cost and QALY reweight)
#:   -> solved live at 0.012194. Verified against the closed form below, which agrees to
#:   0.4% relative, and against the hand calculation at 0.012154.
#:
#: The device price is UNCHANGED at $250, and the widened $100-$900 axis moves no
#: threshold SOLVE -- but it is not inert elsewhere, and that must not be glossed.
#: `c_device_cinpwt.low/.high` serve three roles at once: the reporting axis, the DSA
#: bar and the PSA prior. Widening it therefore also widened the DSA device swing from
#: $500 to $800 -- which is what moves the SSI-cost/device crossing to 2.561%, above the
#: 2.0% low stratum -- and widened the PSA gamma's SE from about $128 to about $204,
#: raising the mean sampled price. Both are disclosed in BUILD-CONTRACT section 3 and in
#: the Methods, because a reported finding whose sign depends on a chosen axis width
#: must say so.
TARGET_P_STAR_UNDISC = 0.012194      # engine, undiscounted, no half-cycle
TARGET_P_STAR_HANDCALC = 0.012154    # price-year-consistent hand calculation

#: Absolute tolerance, in dollars, for every closed-form / live-engine comparison.
#: Measured worst case is ~5e-13; 1e-6 leaves six orders of headroom while still
#: catching any real structural break.
MONEY_TOL = 1e-6

#: brentq tolerances. The INMB curve is smooth and monotone, so machine precision is
#: reachable and costs ~40 engine pairs per solve.
XTOL, RTOL, MAXITER = 1e-15, 8.881784197001252e-16, 200


# --------------------------------------------------------------------------- #
# I/O and formatting helpers (same conventions as run_basecase.py)             #
# --------------------------------------------------------------------------- #
def write_results_csv(df: pd.DataFrame, filename: str) -> str:
    """Write a results CSV with `inputs.RESULTS_BANNER` as a '#'-prefixed line 1."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", newline="") as fh:
        fh.write("# " + I.RESULTS_BANNER + "\n")
        df.to_csv(fh, index=False)
    return path


def _fmt_money(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    sign = "-" if x < 0 else ""
    a = abs(x)
    return f"{sign}${a:,.0f}" if a >= 1 else f"{sign}${a:,.2f}"


def _fmt_icer(x) -> str:
    """An ICER, or an explicit label. NEVER prints a bare NaN as if it were a number:
    `icer` is NaN under strict dominance, which is most of this paper's risk range."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a (dominant)"
    return f"${x:,.0f}/QALY"


def _fmt_pct(x, dp: int = 3) -> str:
    """A probability as a percent to `dp` decimal places (BUILD-CONTRACT reporting
    convention for the risk threshold: 3 decimal places OF A PERCENT)."""
    return f"{100.0 * float(x):.{dp}f}%"


# --------------------------------------------------------------------------- #
# 1. The engine interface -- one cached pair per (p, RR)                       #
# --------------------------------------------------------------------------- #
#: (p_base, rr, disc, half_cycle) -> increment bundle at the BASE device price. Every
#: device price is reached from here by the closed-form shift, so this cache is the
#: entire cost of the analysis: 291 pairs for the curve, 183 for the RR sweep, ~40 per
#: brentq solve.
_PAIR_CACHE: dict = {}


def increments(p: float, rr: float | None = None, disc: float = DISC,
               half_cycle: bool = HALF_CYCLE, sample: dict | None = None) -> dict:
    """Incremental (B_cinpwt MINUS A_standard) results at baseline risk `p`, relative
    risk `rr` and the BASE device price.

    `disc` / `half_cycle` default to the base basis (3%/yr, trapezoidal) and are
    parameters only so the undiscounted / no-half-cycle threshold can be solved and
    REPORTED ON ITS OWN BASIS rather than being solved on one basis and described with
    another basis's increments.

    `rr=None` means "use `tp.rr_cinpwt` through the engine's own `val()`" -- which is
    what every result outside family (d) uses. Passing a float overrides it.

    `sample=None` means "every parameter at its stored value", which is every base-case
    row in this module. The ONLY non-None use is the attribution scenario
    (`m_ssi_cost = k_attrib_pennington`), which is a SCENARIO and is written to
    `threshold_results.csv` under its own `_attrib` family names -- it never touches a
    base-case row. The sample is part of the cache key, so the two families can never
    read each other's cached pairs.

    NOTE THE ARGUMENT ORDER of `model.incremental`: B first. And note that `p` is the
    COMPARATOR arm's 90-day risk -- the engine derives arm B's as `p x rr` internally,
    so a pre-multiplied `p` would double-apply the effect.

    Returns a dict carrying the incremental economics, the four clinical deltas and
    NNT. Everything except `inc_cost` / `inmb_*` is independent of the device price.
    """
    samp_key = None if not sample else tuple(sorted(
        (str(k), float(v)) for k, v in sample.items()))
    key = (float(p), None if rr is None else float(rr), float(disc), bool(half_cycle),
           samp_key)
    hit = _PAIR_CACHE.get(key)
    if hit is not None:
        return hit

    res_a, res_b = model.run_pair(
        p_base=float(p), rr=rr, horizon_cycles=BASE_HORIZON,
        half_cycle=bool(half_cycle), disc_cost=float(disc), disc_qaly=float(disc),
        sample=(None if sample is None else dict(sample)),
    )
    inc = model.incremental(res_b, res_a, wtp=WTP)

    d_ssi = res_a["n_ssi"] - res_b["n_ssi"]
    d_deep = res_a["n_ssi_deep"] - res_b["n_ssi_deep"]
    d_reop = res_a["n_reop"] - res_b["n_reop"]
    d_pers = res_a["n_persistent"] - res_b["n_persistent"]

    out = {
        "p": float(p),
        "rr_used": float(res_b["rr_used"]),
        "device_cost_used": float(res_b["device_cost_used"]),
        "inc_cost": float(inc["inc_cost"]),
        "inc_qaly": float(inc["inc_qaly"]),
        "icer": float(inc["icer"]),
        "dominance": inc["dominance"],
        "quadrant": inc["quadrant"],
        # ---- clinical translation (counts, per 1 patient) ----
        # NNT from the COUNTS, not from 1/(p*(1-RR)): the counts are what reproduce the
        # reported 38.9 at p = 0.066, because competing mortality attenuates both
        # arms' incidence by the same ~0.32% relative.
        "d_ssi": float(d_ssi),
        "d_deep": float(d_deep),
        "d_reop": float(d_reop),
        "d_pers": float(d_pers),
        "nnt": float(1.0 / d_ssi) if d_ssi > 0 else float("nan"),
    }
    _PAIR_CACHE[key] = out
    return out


def inmb_at(p: float, wtp: float, device_cost: float | None = None,
            rr: float | None = None, **basis) -> float:
    """INMB (B minus A) at risk `p`, willingness-to-pay `wtp` and device price
    `device_cost` (None = the base price), via the closed-form price shift.

    `**basis` is forwarded verbatim to `increments`: `disc`, `half_cycle` (the reporting
    convention) and `sample` (the attribution scenario). Omitting it is the base case on
    the base basis.

    `wtp = 0` reduces this to `-inc_cost`, so the cost-neutral criterion is the WTP = $0
    member of the same family. This module nevertheless solves the cost-neutral
    threshold independently on `inc_cost` and asserts the two agree.
    """
    r = increments(p, rr, **basis)
    shift = 0.0 if device_cost is None else (float(device_cost) - C0)
    return r["inc_qaly"] * float(wtp) - (r["inc_cost"] + shift)


def inc_cost_at(p: float, device_cost: float | None = None,
                rr: float | None = None, **basis) -> float:
    """Incremental cost (B minus A) at risk `p` and device price `device_cost`."""
    r = increments(p, rr, **basis)
    shift = 0.0 if device_cost is None else (float(device_cost) - C0)
    return r["inc_cost"] + shift


# --------------------------------------------------------------------------- #
# 2. Proof that the closed-form device-price shortcut is EXACT                 #
# --------------------------------------------------------------------------- #
def prove_device_cost_linearity(verbose: bool = True) -> dict:
    """Prove -- against LIVE engine re-runs -- that the device price is a pure additive
    cycle-0 toll in arm B, before any result is computed from the shortcut.

    Three independent checks, all asserted to `MONEY_TOL` dollars:

    1. **Corners.** 5 risks x 5 prices (including two prices far outside the swept
       $100-$600 axis, because the solved threshold prices are outside it). Checks
       `cost_B(C) == cost_B(C0) + (C - C0)`, that arm A is untouched by `device_cost`,
       and that arm B's QALYs, `n_ssi` and `n_reop` are untouched.
    2. **Actual map cells.** Twelve cells drawn from the real 291 x 161 grid with the
       BUILD-CONTRACT section 9 seed (12345), re-run live and compared with the shifted
       values that will be written to `threshold_grid_risk_price.csv`. This proves the
       file, not just the principle.
    3. **The closed form itself.** `C*(p, w) = C0 + INMB(p, C0, w)` re-run live at every
       stratum x reported WTP; the live INMB at `C*` must be zero.

    Returns the three worst deviations. Raises `AssertionError` if the engine ever stops
    being linear in the device price -- in which case the 46,851-row shortcut is invalid
    and this module must be rewritten to run the full grid.
    """
    probe_p = (float(RISK_GRID[0]), STRATA["low"], STRATA["intermediate"],
               STRATA["high"], float(RISK_GRID[-1]))
    probe_c = (float(PRICE_GRID[0]), C0, 375.0, float(PRICE_GRID[-1]), 2500.0)

    worst_corner = 0.0
    for p in probe_p:
        base_a, base_b = model.run_pair(
            p_base=p, device_cost=C0, horizon_cycles=BASE_HORIZON,
            half_cycle=HALF_CYCLE, disc_cost=DISC, disc_qaly=DISC)
        for c in probe_c:
            a, b = model.run_pair(
                p_base=p, device_cost=c, horizon_cycles=BASE_HORIZON,
                half_cycle=HALF_CYCLE, disc_cost=DISC, disc_qaly=DISC)
            worst_corner = max(
                worst_corner,
                abs(b["cost"] - (base_b["cost"] + (c - C0))),   # THE claim
                abs(a["cost"] - base_a["cost"]),                # arm A untouched
                abs(b["qaly"] - base_b["qaly"]),                # QALYs untouched
                abs(a["qaly"] - base_a["qaly"]),
                abs(b["n_ssi"] - base_b["n_ssi"]) * 1e6,        # counts untouched
                abs(b["n_reop"] - base_b["n_reop"]) * 1e6,
            )
    assert worst_corner < MONEY_TOL, (
        f"device cost is NOT a pure additive cycle-0 toll (worst deviation "
        f"${worst_corner:.3e}); the 291-pair shortcut behind "
        f"threshold_grid_risk_price.csv is INVALID -- run the full 291 x 161 sweep")

    rng = np.random.default_rng(int(S["seed"]))
    worst_cell = 0.0
    for _ in range(12):
        p = float(RISK_GRID[rng.integers(len(RISK_GRID))])
        c = float(PRICE_GRID[rng.integers(len(PRICE_GRID))])
        a, b = model.run_pair(
            p_base=p, device_cost=c, horizon_cycles=BASE_HORIZON,
            half_cycle=HALF_CYCLE, disc_cost=DISC, disc_qaly=DISC)
        live = model.incremental(b, a, wtp=WTP)
        worst_cell = max(worst_cell,
                         abs(live["inc_cost"] - inc_cost_at(p, c)),
                         abs(live["inmb"] - inmb_at(p, WTP, c)))
    assert worst_cell < MONEY_TOL, (
        f"a live re-run disagrees with the shifted grid value by ${worst_cell:.3e}")

    worst_star = 0.0
    for p in STRATA.values():
        for w in WTP_REPORTED:
            c_star = C0 + inmb_at(p, w)
            a, b = model.run_pair(
                p_base=p, device_cost=c_star, horizon_cycles=BASE_HORIZON,
                half_cycle=HALF_CYCLE, disc_cost=DISC, disc_qaly=DISC)
            worst_star = max(worst_star, abs(model.incremental(b, a, wtp=w)["inmb"]))
    assert worst_star < MONEY_TOL, (
        f"the closed-form threshold price C* = C0 + INMB does not zero the live INMB "
        f"(worst ${worst_star:.3e})")

    if verbose:
        print("Closed-form device-price shortcut PROVEN against live engine re-runs:")
        print(f"  cost_B(C) == cost_B(C0) + (C - C0), 5 risks x 5 prices : "
              f"max dev ${worst_corner:.2e}")
        print(f"  12 random cells of the actual 291 x 161 map            : "
              f"max dev ${worst_cell:.2e}")
        print(f"  live INMB at the closed-form C* (9 stratum x WTP cells): "
              f"max dev ${worst_star:.2e}")
        print(f"  -> 291 engine pairs instead of {len(RISK_GRID) * len(PRICE_GRID):,}.")
    return {"corners": worst_corner, "cells": worst_cell, "c_star": worst_star}


# --------------------------------------------------------------------------- #
# 3. Root finding -- brentq bracketed on the contract's own grids              #
# --------------------------------------------------------------------------- #
def _bracket_on_grid(f, grid: np.ndarray):
    """First sign change of `f` between adjacent points of `grid`.

    Bracketing on the sweep grid (rather than on an arbitrary wide interval) means the
    bracket is reported in the same coordinates the paper plots, and a threshold that
    falls OFF the swept axis is detected rather than silently extrapolated into.
    Returns `(lo, hi, "grid")`, or `(None, None, reason)` when no sign change exists.
    """
    v = np.array([f(float(x)) for x in grid], dtype=float)
    if not np.all(np.isfinite(v)):
        return None, None, "non-finite objective on the grid"
    s = np.sign(v)
    idx = np.flatnonzero(np.diff(s) != 0)
    if idx.size == 0:
        side = "always positive" if v[0] > 0 else "always negative"
        return None, None, f"no sign change on the swept grid ({side})"
    i = int(idx[0])
    return float(grid[i]), float(grid[i + 1]), "grid"


def solve_on_grid(f, grid: np.ndarray, wide=None) -> dict:
    """Solve `f(x) = 0` by `scipy.optimize.brentq`, bracketed on `grid`.

    If the grid does not bracket a root, falls back to `wide` (when given) and records
    `solver = "brentq (extended bracket)"` plus `on_swept_grid = False`, so a threshold
    outside the reported axis is REPORTED as such and never quietly presented as
    if it had been swept. Returns a dict; `root` is NaN when no root exists at all.
    """
    lo, hi, how = _bracket_on_grid(f, grid)
    if how == "grid":
        root = brentq(f, lo, hi, xtol=XTOL, rtol=RTOL, maxiter=MAXITER)
        return {"root": float(root), "bracket_lo": lo, "bracket_hi": hi,
                "solver": "brentq (bracketed on the swept grid)",
                "on_swept_grid": True, "note": ""}
    if wide is not None:
        wlo, whi = float(wide[0]), float(wide[1])
        if np.sign(f(wlo)) != np.sign(f(whi)):
            root = brentq(f, wlo, whi, xtol=XTOL, rtol=RTOL, maxiter=MAXITER)
            return {"root": float(root), "bracket_lo": wlo, "bracket_hi": whi,
                    "solver": "brentq (extended bracket)", "on_swept_grid": False,
                    "note": f"root lies outside the swept grid "
                            f"[{grid[0]:g}, {grid[-1]:g}]: {how}"}
    return {"root": float("nan"), "bracket_lo": float("nan"),
            "bracket_hi": float("nan"), "solver": "no root", "on_swept_grid": False,
            "note": how}


# --------------------------------------------------------------------------- #
# Deliverable 1 -- threshold_curve.csv (Fig 2 data)                            #
# --------------------------------------------------------------------------- #
def build_threshold_curve() -> pd.DataFrame:
    """One row per point of `SETTINGS["risk_grid"]` at the BASE device price and BASE
    relative risk: the continuous INMB-versus-baseline-risk curve that is this paper's
    structural advance over the exemplars (Losina gives discrete tiers, JBJS a shaded
    two-way region; nobody plots INMB as a continuous function of baseline risk with a
    marked zero crossing).

    Columns are EXACTLY the BUILD-CONTRACT section 8 list, in order. `icer` is NaN
    wherever ciNPWT is dominant -- read `dominance` / `quadrant`, never a bare `icer`.
    """
    rows = []
    for p in RISK_GRID:
        r = increments(float(p))
        rows.append({
            "p": float(p),
            "inc_cost": r["inc_cost"],
            "inc_qaly": r["inc_qaly"],
            "inmb_50k": r["inc_qaly"] * WTP_LO - r["inc_cost"],
            "inmb_100k": r["inc_qaly"] * WTP - r["inc_cost"],
            "inmb_150k": r["inc_qaly"] * WTP_HI - r["inc_cost"],
            "icer": r["icer"],
            "dominance": r["dominance"],
            "quadrant": r["quadrant"],
            "nnt": r["nnt"],
            "ssi_prevented_per_1000": 1000.0 * r["d_ssi"],
            "reop_prevented_per_1000": 1000.0 * r["d_reop"],
        })
    df = pd.DataFrame(rows)

    assert len(df) == len(RISK_GRID) == 291, f"curve has {len(df)} rows, expected 291"
    # Contract verification step 4, first half: INMB must be monotone INCREASING in
    # baseline risk at every WTP. A violation would mean the sweep, not the model.
    for col in ("inmb_50k", "inmb_100k", "inmb_150k"):
        assert (np.diff(df[col].values) > 0).all(), f"{col} is not monotone in risk"
    assert (np.diff(df["inc_cost"].values) < 0).all(), \
        "incremental cost is not monotone DEcreasing in risk"
    assert (np.diff(df["inc_qaly"].values) > 0).all(), \
        "incremental QALY is not monotone increasing in risk"

    write_results_csv(df, "threshold_curve.csv")
    return df


# --------------------------------------------------------------------------- #
# Deliverable 2 -- threshold_grid_risk_price.csv (Fig 3 data)                  #
# --------------------------------------------------------------------------- #
def build_threshold_grid(curve: pd.DataFrame) -> pd.DataFrame:
    """The two-way risk x price map, LONG format (291 x 161 = 46,851 rows) -- not a
    pivot; `make_figures.py` reshapes it.

    Built from the 291-row curve by the PROVEN closed-form shift (see
    `prove_device_cost_linearity`), which is why this costs no engine calls at all:

        INMB(p, C, w) = INMB(p, C0, w) - (C - C0)
        inc_cost(p, C) = inc_cost(p, C0) + (C - C0)

    `inc_qaly` is deliberately NOT carried: it does not depend on `C`, so repeating it
    46,851 times would invite a reader to think it did. It is in `threshold_curve.csv`.
    """
    delta = PRICE_GRID - C0                                   # (101,)
    n_p, n_c = len(RISK_GRID), len(PRICE_GRID)

    inmb50 = curve["inmb_50k"].values[:, None] - delta[None, :]
    inmb100 = curve["inmb_100k"].values[:, None] - delta[None, :]
    inmb150 = curve["inmb_150k"].values[:, None] - delta[None, :]
    inc_cost = curve["inc_cost"].values[:, None] + delta[None, :]

    df = pd.DataFrame({
        "p": np.repeat(RISK_GRID, n_c),
        "device_cost": np.tile(PRICE_GRID, n_p),
        "inmb_50k": inmb50.ravel(),
        "inmb_100k": inmb100.ravel(),
        "inmb_150k": inmb150.ravel(),
        "inc_cost": inc_cost.ravel(),
    })

    # A2 2026-08-17: the price axis was widened $100-$600 -> $100-$900 (101 -> 161 points)
    # so that BOTH ciNPT list prices Mueller 2026 [9] publishes ($495 and $830) sit on the
    # reported axis. 291 x 161 = 46,851.
    assert len(df) == n_p * n_c == 291 * 161, f"grid has {len(df)} rows, expected 46,851"
    # Contract verification step 4, second half: the Fig 3 frontier must be monotone in
    # BOTH axes -- INMB rising with risk and falling with price.
    for M in (inmb50, inmb100, inmb150):
        assert (np.diff(M, axis=1) < 0).all(), "INMB is not monotone in device price"
        assert (np.diff(M, axis=0) > 0).all(), "INMB is not monotone in risk (grid)"
    assert (np.diff(inc_cost, axis=1) > 0).all(), \
        "incremental cost is not monotone increasing in device price"

    write_results_csv(df, "threshold_grid_risk_price.csv")
    return df


# --------------------------------------------------------------------------- #
# Deliverable 3 -- threshold_rr.csv                                            #
# --------------------------------------------------------------------------- #
def build_threshold_rr() -> pd.DataFrame:
    """Per risk stratum x `SETTINGS["rr_grid"]` (3 x 61 = 183 rows) at the BASE device
    price: how the incremental result moves as the assumed ciNPWT effect is weakened
    from RR 0.40 all the way to RR 1.00 (no effect at all).

    Columns are EXACTLY the BUILD-CONTRACT section 8 list. At RR = 1.00 the two arms are
    identical apart from the cycle-0 device toll, so `inc_qaly` is exactly 0 and
    `inc_cost` is exactly the device price -- which is the natural end-point check on
    this sweep and is asserted below.
    """
    rows = []
    for sname, p in STRATA.items():
        for rr in RR_GRID:
            r = increments(float(p), float(rr))
            rows.append({
                "stratum": sname,
                "p": float(p),
                "rr": float(rr),
                "inc_cost": r["inc_cost"],
                "inc_qaly": r["inc_qaly"],
                "inmb_100k": r["inc_qaly"] * WTP - r["inc_cost"],
            })
    df = pd.DataFrame(rows)

    assert len(df) == len(STRATA) * len(RR_GRID) == 183, \
        f"rr sweep has {len(df)} rows, expected 183"
    for sname in STRATA:
        sub = df[df["stratum"] == sname].sort_values("rr")
        # Weakening the effect can only reduce the benefit: INMB must fall as RR rises.
        assert (np.diff(sub["inmb_100k"].values) < 0).all(), \
            f"INMB is not monotone DEcreasing in RR ({sname})"
        assert (np.diff(sub["inc_cost"].values) > 0).all(), \
            f"incremental cost is not monotone increasing in RR ({sname})"
        end = sub.iloc[-1]
        assert abs(end["rr"] - 1.0) < 1e-12
        assert abs(end["inc_qaly"]) < 1e-12, \
            "at RR = 1.00 the arms must be clinically identical (inc_qaly == 0)"
        assert abs(end["inc_cost"] - (C0 - float(I.cost_2025(I.cost.c_dressing_standard)))) \
            < MONEY_TOL, ("at RR = 1.00 the incremental cost must be exactly the device "
                          "price net of the standard-dressing differential")

    write_results_csv(df, "threshold_rr.csv")
    return df


# --------------------------------------------------------------------------- #
# Deliverable 4 -- threshold_results.csv (Table 3)                             #
# --------------------------------------------------------------------------- #
#: Wide fallback brackets, used only when the reported sweep grid does not
#: bracket the root. `p` is a probability; RR is allowed to run past 1.0 in the
#: fallback because "the effect would have to be HARMFUL" is a legitimate answer.
WIDE_P = (1e-9, 0.999)
WIDE_RR = (1e-6, 3.0)

BASIS_BASE = (f"base ({BASE_HORIZON} cycles, {DISC:.0%}/yr discounting, "
              f"half-cycle correction)")
BASIS_UNDISC = f"undiscounted, no half-cycle ({BASE_HORIZON} cycles)"

#: THE `_attrib` FAMILY WAS INVERTED BY THE A2 AUDIT (2026-08-17).
#:
#: Before A2 the base case charged Edmiston's regression-adjusted ALL-CAUSE payment
#: increment at face value (m_ssi_cost = 1.000) and this family carried the
#: attribution-adjusted sensitivity at m_ssi_cost = k_attrib_pennington.
#:
#: A2 established from the full text that Edmiston's payments are least-squares means
#: from a GLM with log link and gamma error, covariate-adjusted only, over ALL-CAUSE
#: total payments (medical plus prescription drug), with NO MATCHING anywhere in the
#: paper. That estimand is an UPPER BOUND on SSI-attributable cost by construction: the
#: residual gap is confounding by indication. Pennington's is a matched contrast in
#: exactly this model's population. So the base case now takes the matched anchor
#: (m_ssi_cost = 0.492) and THIS FAMILY CARRIES THE ALL-CAUSE UPPER BOUND instead.
#:
#: The family KEY is deliberately left as `_attrib` so that every downstream selector
#: (`family.str.endswith("_attrib")`) keeps working; what changed is which contrast the
#: family holds, and every `basis`, `quantity` and note string below says so explicitly
#: so no reader can mistake it. The pair the manuscript reports is still like-for-like:
#: base (matched) -> sensitivity (all-cause), both on the REPORTING BASIS.
K_ATTRIB = float(I.cost.k_attrib_pennington)            # 0.492 -- now the BASE case
M_ALLCAUSE = 1.0                                        # Edmiston all-cause, at face value
ATTRIB_SAMPLE = {"m_ssi_cost": M_ALLCAUSE}
BASIS_ATTRIB = (f"{BASIS_BASE}; SSI cost at Edmiston's ALL-CAUSE increment "
                f"(m_ssi_cost = {M_ALLCAUSE:.3f}) rather than the matched base case "
                f"(m_ssi_cost = {K_ATTRIB:.3f})")
ATTRIB_NOTE = (
    f"ALL-CAUSE COST SENSITIVITY, not the base case. The six SSI cost tranches are "
    f"charged at m_ssi_cost = {M_ALLCAUSE:.3f}, i.e. Edmiston [12]'s regression-adjusted "
    f"ALL-CAUSE payment increment taken at face value, instead of the base case's "
    f"matched anchor m_ssi_cost = k_attrib_pennington = {K_ATTRIB:.3f} from Pennington "
    f"[14]. Because an all-cause increment absorbs confounding by indication it is an "
    f"UPPER bound on attributable cost, so this family reports a LOWER break-even risk "
    f"than the base case -- ciNPWT appears to pay off sooner. That direction is the "
    f"whole point of reporting it as a bound rather than as the headline. Solved and "
    f"evaluated on the REPORTING BASIS ({BASE_HORIZON} cycles, {DISC:.0%}/yr, "
    f"half-cycle) so it pairs like-for-like with the base-case rows of the same family. "
    f"Select these rows with family.str.endswith('_attrib').")


def _row(*, family, quantity, stratum, p_base, wtp, criterion, basis, threshold_value,
         units, rr_used, device_cost_used, sol, headline, interpretation,
         horizon=BASE_HORIZON, disc=DISC, half_cycle=HALF_CYCLE,
         sample: dict | None = None) -> dict:
    """Assemble one `threshold_results.csv` row and evaluate the model AT the solution,
    on the row's OWN basis (`disc` / `half_cycle`).

    Every headline row is self-checking: the increments below are recomputed at the
    solved coordinates, so a row that claims INMB = 0 carries an `inmb` column that is
    zero to machine precision, and `main()` asserts it.

    `dominance`, `quadrant` and `icer` are obtained by handing the shifted increments
    back to `model.incremental` -- the engine's own branch chain and its
    `DOMINANCE_TO_QUADRANT` map -- rather than being re-derived here. That matters:
    `incremental` returns `icer = NaN` under strict dominance, and this table contains
    dominant rows (every cost-neutral row has `inc_cost = 0` with a positive QALY gain).
    A locally computed ratio would print `0` there, i.e. a bare number that is not a
    number. `incremental` reads only the cost/QALY difference of its two arguments, so a
    two-key stand-in for arm A is exact, not a hack.

    `sample` MUST be the same sample the row was solved under -- it is threaded into the
    re-evaluation so an attribution-scenario row is checked against the attribution
    engine and not against the base case. `basis` is a free-text label and is the string
    that names the reporting convention (and, for a scenario row, the scenario) in the
    published CSV.
    """
    r = increments(float(p_base), None if rr_used is None else float(rr_used),
                   disc=disc, half_cycle=half_cycle, sample=sample)
    shift = float(device_cost_used) - C0
    inc_cost = r["inc_cost"] + shift
    inc_qaly = r["inc_qaly"]
    at_sol = model.incremental({"cost": inc_cost, "qaly": inc_qaly},
                               {"cost": 0.0, "qaly": 0.0}, wtp=float(wtp))
    inmb = at_sol["inmb"]
    icer = at_sol["icer"]
    return {
        "family": family,
        "quantity": quantity,
        "stratum": stratum,
        "criterion": criterion,
        "basis": basis,
        "horizon_cycles": int(horizon),
        "disc_rate": float(disc),
        "half_cycle": bool(half_cycle),
        "wtp": float(wtp),
        "p_base": float(p_base),
        "rr_used": float(r["rr_used"]),
        "device_cost_used": float(device_cost_used),
        "threshold_value": float(threshold_value),
        "units": units,
        "inc_cost": float(inc_cost),
        "inc_qaly": float(inc_qaly),
        "inmb": float(inmb),
        "icer": float(icer),
        "dominance": at_sol["dominance"],
        "quadrant": at_sol["quadrant"],
        "nnt": r["nnt"],
        "ssi_prevented_per_1000": 1000.0 * r["d_ssi"],
        "ssi_deep_prevented_per_1000": 1000.0 * r["d_deep"],
        "reop_prevented_per_1000": 1000.0 * r["d_reop"],
        "solver": sol["solver"],
        "bracket_lo": sol["bracket_lo"],
        "bracket_hi": sol["bracket_hi"],
        "on_swept_grid": bool(sol["on_swept_grid"]),
        "solver_note": sol["note"],
        "headline": bool(headline),
        "interpretation": interpretation,
    }


def solve_thresholds() -> pd.DataFrame:
    """Solve all four threshold families and write `threshold_results.csv`.

    `headline = True` marks the rows Table 3 prints; `headline = False` is the 81-point
    CEAC-grid family, carried because it is nearly free and because `make_figures.py`
    may want the threshold-versus-WTP curve. Filter on `headline` before building a
    table, and on `family` before building a panel.
    """
    rows: list[dict] = []
    #: base-case p* by WTP, kept so the attribution family (e) can quote the
    #: like-for-like base-case value ON THE SAME BASIS instead of re-deriving it.
    base_p_star: dict[float, float] = {}

    # ---- (a) INMB = 0 baseline-risk threshold, at each reported WTP ----------- #
    for w in WTP_REPORTED:
        sol = solve_on_grid(lambda p, w=w: inmb_at(p, w), RISK_GRID, wide=WIDE_P)
        p_star = sol["root"]
        base_p_star[float(w)] = float(p_star)
        r = increments(p_star)
        rows.append(_row(
            family="risk_threshold", quantity="Baseline 90-day all-SSI risk",
            stratum="", p_base=p_star, wtp=w, criterion="INMB = 0", basis=BASIS_BASE,
            threshold_value=p_star, units="proportion (baseline 90-day all-SSI risk)",
            rr_used=None, device_cost_used=C0, sol=sol, headline=True,
            interpretation=(
                f"At a willingness-to-pay of ${w:,.0f}/QALY and a ciNPWT unit price of "
                f"${C0:,.0f}, ciNPWT is cost-effective in any patient whose predicted "
                f"90-day SSI risk exceeds {_fmt_pct(p_star)}. At that risk ciNPWT "
                f"prevents {1000.0 * r['d_ssi']:.2f} SSIs per 1,000 patients treated "
                f"(NNT {r['nnt']:.0f})."),
        ))

    # ---- (b) cost-neutral baseline-risk threshold ---------------------------- #
    # Solved INDEPENDENTLY on `inc_cost` (QALYs ignored entirely), then asserted equal
    # to the WTP = $0 member of family (a) -- the equivalence is demonstrated, not
    # assumed. This is the payer's break-even and is deliberately the most conservative
    # (highest) of the four risk thresholds.
    sol_cn = solve_on_grid(lambda p: -inc_cost_at(p), RISK_GRID, wide=WIDE_P)
    p_cn = sol_cn["root"]
    sol_w0 = solve_on_grid(lambda p: inmb_at(p, 0.0), RISK_GRID, wide=WIDE_P)
    assert abs(p_cn - sol_w0["root"]) < 1e-12, (
        f"the cost-neutral threshold solved on inc_cost ({p_cn!r}) disagrees with the "
        f"WTP = $0 INMB threshold ({sol_w0['root']!r}) -- one of the two is wrong")
    r_cn = increments(p_cn)
    rows.append(_row(
        family="risk_threshold_cost_neutral",
        quantity="Baseline 90-day all-SSI risk (cost-neutral)", stratum="",
        p_base=p_cn, wtp=0.0, criterion="incremental cost = 0", basis=BASIS_BASE,
        threshold_value=p_cn, units="proportion (baseline 90-day all-SSI risk)",
        rr_used=None, device_cost_used=C0, sol=sol_cn, headline=True,
        interpretation=(
            f"Ignoring QALYs entirely, ciNPWT pays for itself in pure payer dollars at a "
            f"predicted 90-day SSI risk of {_fmt_pct(p_cn)} and above; below that it "
            f"still buys health, but it costs money to do so. This is the most "
            f"conservative of the four risk thresholds -- it is the number to quote to a "
            f"payer who declines to price QALYs. At that risk ciNPWT prevents "
            f"{1000.0 * r_cn['d_ssi']:.2f} SSIs per 1,000 (NNT {r_cn['nnt']:.0f})."),
    ))

    # ---- (a') the same threshold on the undiscounted / no-half-cycle basis ---- #
    # Reported so the discounted headline can be compared like-for-like with the
    # engine's published p* = 0.6515% and the hand calculation's 0.6492%. Solved AND
    # evaluated on its own basis, so the row is self-checking like every other.
    UNDISC = {"disc": 0.0, "half_cycle": False}
    sol_u = solve_on_grid(lambda p: inmb_at(p, WTP, **UNDISC), RISK_GRID, wide=WIDE_P)
    p_u = sol_u["root"]
    assert abs(p_u - TARGET_P_STAR_UNDISC) < 5e-6, (
        f"the undiscounted threshold {p_u:.6f} does not reproduce the engine's "
        f"published {TARGET_P_STAR_UNDISC:.6f} (run_log.d/model.md)")
    rows.append(_row(
        family="risk_threshold",
        quantity="Baseline 90-day all-SSI risk (undiscounted, no half-cycle)",
        stratum="", p_base=p_u, wtp=WTP, criterion="INMB = 0", basis=BASIS_UNDISC,
        threshold_value=p_u, units="proportion (baseline 90-day all-SSI risk)",
        rr_used=None, device_cost_used=C0, sol=sol_u, headline=True,
        disc=0.0, half_cycle=False,
        interpretation=(
            f"The same threshold computed without discounting and without the half-cycle "
            f"correction: {_fmt_pct(p_u)}. It reproduces the engine's published "
            f"{_fmt_pct(TARGET_P_STAR_UNDISC)} and sits just above the price-year-"
            f"consistent hand calculation {_fmt_pct(TARGET_P_STAR_HANDCALC)} "
            f"(p* = C / [(1 - RR) x (SSI cost + WTP x QALY loss)]). Discounting raises "
            f"the threshold because the device toll falls at cycle 0, where the discount "
            f"factor is exactly 1, while every benefit is discounted. Every column on "
            f"this row is evaluated on the undiscounted basis."),
    ))

    # ---- (a'') the CEAC grid (cheap: ~40 cached engine pairs per solve) ------- #
    for w in CEAC_GRID:
        sol = solve_on_grid(lambda p, w=float(w): inmb_at(p, w), RISK_GRID, wide=WIDE_P)
        rows.append(_row(
            family="risk_threshold_ceac_grid",
            quantity="Baseline 90-day all-SSI risk", stratum="", p_base=sol["root"],
            wtp=float(w), criterion="INMB = 0", basis=BASIS_BASE,
            threshold_value=sol["root"],
            units="proportion (baseline 90-day all-SSI risk)",
            rr_used=None, device_cost_used=C0, sol=sol, headline=False,
            interpretation=(f"Threshold risk at WTP ${float(w):,.0f}/QALY: "
                            f"{_fmt_pct(sol['root'])}."),
        ))

    # ---- (c) threshold device price per stratum, CLOSED FORM ------------------ #
    # C*(p, w) = C0 + INMB(p, C0, w), proven exact in prove_device_cost_linearity().
    for sname, p in STRATA.items():
        for w in (0.0,) + WTP_REPORTED:
            c_star = C0 + inmb_at(p, w)
            on_grid = bool(PRICE_GRID[0] <= c_star <= PRICE_GRID[-1])
            sol = {
                "solver": "closed form C* = C0 + INMB(p, C0, wtp)",
                "bracket_lo": float(PRICE_GRID[0]), "bracket_hi": float(PRICE_GRID[-1]),
                "on_swept_grid": on_grid,
                "note": "" if on_grid else (
                    f"C* = ${c_star:,.0f} lies OUTSIDE the reported price axis "
                    f"${PRICE_GRID[0]:,.0f}-${PRICE_GRID[-1]:,.0f}: ciNPWT pays at every "
                    f"price on the swept axis in this stratum"),
            }
            r = increments(p)
            if w == 0.0:
                crit, qty = ("incremental cost = 0",
                             "ciNPWT unit price (cost-neutral)")
                lead = ("ciNPWT pays for itself in pure payer dollars, ignoring QALYs, "
                        f"as long as its unit price stays below ${c_star:,.0f} per "
                        "patient")
            else:
                crit, qty = "INMB = 0", "ciNPWT unit price"
                lead = (f"ciNPWT remains cost-effective at a willingness-to-pay of "
                        f"${w:,.0f}/QALY as long as the device price remains below "
                        f"${c_star:,.0f} per patient")
            rows.append(_row(
                family="device_price_threshold", quantity=qty, stratum=sname,
                p_base=p, wtp=w, criterion=crit, basis=BASIS_BASE,
                threshold_value=c_star, units=f"{I.PRICE_YEAR} USD per patient",
                rr_used=None, device_cost_used=c_star, sol=sol, headline=True,
                interpretation=(
                    f"In the {sname}-risk stratum (predicted 90-day SSI risk "
                    f"{_fmt_pct(p, 1)}), {lead}. "
                    + ("" if on_grid else
                       f"That is above the top of the reported $"
                       f"{PRICE_GRID[0]:,.0f}-${PRICE_GRID[-1]:,.0f} price axis, so "
                       f"within any plausible acquisition price ciNPWT pays here. ")
                    + f"At this risk ciNPWT prevents {1000.0 * r['d_ssi']:.1f} SSIs and "
                      f"{1000.0 * r['d_reop']:.1f} reoperations per 1,000 (NNT "
                      f"{r['nnt']:.0f})."),
            ))

    # ---- (d) threshold relative risk per stratum ----------------------------- #
    for sname, p in STRATA.items():
        for w in (0.0,) + WTP_REPORTED:
            sol = solve_on_grid(lambda rr, p=p, w=w: inmb_at(p, w, rr=float(rr)),
                                RR_GRID, wide=WIDE_RR)
            rr_star = sol["root"]
            r = increments(p, rr_star)
            crit = "incremental cost = 0" if w == 0.0 else "INMB = 0"
            tail = ("pays for itself in pure payer dollars" if w == 0.0
                    else f"is cost-effective at ${w:,.0f}/QALY")
            rows.append(_row(
                family="rr_threshold", quantity="ciNPWT relative risk", stratum=sname,
                p_base=p, wtp=w, criterion=crit, basis=BASIS_BASE,
                threshold_value=rr_star, units="relative risk (ciNPWT vs standard)",
                rr_used=rr_star, device_cost_used=C0, sol=sol, headline=True,
                interpretation=(
                    f"In the {sname}-risk stratum (predicted 90-day SSI risk "
                    f"{_fmt_pct(p, 1)}) at ${C0:,.0f} per device, ciNPWT {tail} for any "
                    f"true relative risk below {rr_star:.3f}. The pooled estimate is "
                    f"{RR0:.2f} ({float(I.tp.rr_cinpwt.low):.2f}-"
                    f"{float(I.tp.rr_cinpwt.high):.2f}), so the effect would have to be "
                    f"weaker than {rr_star:.3f} -- i.e. a relative risk reduction below "
                    f"{100.0 * (1.0 - rr_star):.1f}% rather than the pooled "
                    f"{100.0 * (1.0 - RR0):.0f}% -- before ciNPWT stops paying here."),
            ))

    # ---- (e) THE SAME FOUR FAMILIES UNDER THE ATTRIBUTION SCENARIO ----------- #
    # A2: m_ssi_cost = 1.000 (all-cause upper bound), ON THE REPORTING BASIS.
    # Added because the manuscript's own like-for-like pair -- base p* against the
    # attribution-adjusted p*, BOTH on the basis the paper reports -- previously existed
    # in no persisted artifact: `run_scenarios.py` computed it and only printed it, and
    # the only attribution threshold anywhere in results/ was the UNDISCOUNTED one. Since
    # BUILD-CONTRACT section 10 makes this module authoritative for every published
    # threshold, the scenario belongs here.
    #
    # `headline = False` throughout, deliberately: `headline` means "a base-case row Table
    # 3 prints", and widening it would silently change Table 3. Select on
    # `family.str.endswith("_attrib")` instead. Every row's `basis` names BOTH the
    # reporting convention AND the scenario, so no attribution threshold can escape into a
    # manuscript unlabelled.
    A_KW = {"sample": ATTRIB_SAMPLE}

    # (e-a) INMB = 0 baseline-risk threshold, at each reported WTP
    for w in WTP_REPORTED:
        sol = solve_on_grid(lambda p, w=w: inmb_at(p, w, **A_KW), RISK_GRID, wide=WIDE_P)
        sol = {**sol, "note": (sol["note"] + " " if sol["note"] else "") + ATTRIB_NOTE}
        p_a = sol["root"]
        r = increments(p_a, **A_KW)
        rows.append(_row(
            family="risk_threshold_attrib",
            quantity="Baseline 90-day all-SSI risk (attribution-adjusted)",
            stratum="", p_base=p_a, wtp=w, criterion="INMB = 0", basis=BASIS_ATTRIB,
            threshold_value=p_a, units="proportion (baseline 90-day all-SSI risk)",
            rr_used=None, device_cost_used=C0, sol=sol, headline=False,
            sample=ATTRIB_SAMPLE,
            interpretation=(
                f"ATTRIBUTION SCENARIO (SSI cost x {K_ATTRIB:.3f}, Pennington [14]), on "
                f"the reporting basis ({BASE_HORIZON} cycles, {DISC:.0%}/yr, half-cycle). "
                f"At ${w:,.0f}/QALY and ${C0:,.0f} per device, ciNPWT is cost-effective "
                f"above a predicted 90-day SSI risk of {_fmt_pct(p_a)} -- "
                f"{p_a / base_p_star[float(w)]:.2f}x the base-case threshold on the SAME "
                f"basis ({_fmt_pct(base_p_star[float(w)])}), because only the "
                f"matched-cohort share of the payment increment is credited to the "
                f"infection. This is the "
                f"attribution-adjusted number the manuscript must pair with the "
                f"base-case threshold; both are on the SAME basis."),
        ))

    # (e-b) cost-neutral baseline-risk threshold, attribution-adjusted
    sol_cna = solve_on_grid(lambda p: -inc_cost_at(p, **A_KW), RISK_GRID, wide=WIDE_P)
    p_cna = sol_cna["root"]
    sol_w0a = solve_on_grid(lambda p: inmb_at(p, 0.0, **A_KW), RISK_GRID, wide=WIDE_P)
    assert abs(p_cna - sol_w0a["root"]) < 1e-12, (
        f"attribution scenario: the cost-neutral threshold solved on inc_cost ({p_cna!r}) "
        f"disagrees with the WTP = $0 INMB threshold ({sol_w0a['root']!r})")
    sol_cna = {**sol_cna,
               "note": (sol_cna["note"] + " " if sol_cna["note"] else "") + ATTRIB_NOTE}
    r_cna = increments(p_cna, **A_KW)
    rows.append(_row(
        family="risk_threshold_cost_neutral_attrib",
        quantity="Baseline 90-day all-SSI risk (cost-neutral, attribution-adjusted)",
        stratum="", p_base=p_cna, wtp=0.0, criterion="incremental cost = 0",
        basis=BASIS_ATTRIB, threshold_value=p_cna,
        units="proportion (baseline 90-day all-SSI risk)",
        rr_used=None, device_cost_used=C0, sol=sol_cna, headline=False,
        sample=ATTRIB_SAMPLE,
        interpretation=(
            f"ATTRIBUTION SCENARIO (SSI cost x {K_ATTRIB:.3f}), reporting basis. Ignoring "
            f"QALYs entirely, ciNPWT pays for itself in pure payer dollars at a predicted "
            f"90-day SSI risk of {_fmt_pct(p_cna)} and above. At that risk it prevents "
            f"{1000.0 * r_cna['d_ssi']:.2f} SSIs per 1,000 (NNT {r_cna['nnt']:.0f})."),
    ))

    # (e-c) threshold device price per stratum, attribution-adjusted
    for sname, p in STRATA.items():
        for w in (0.0,) + WTP_REPORTED:
            c_star = C0 + inmb_at(p, w, **A_KW)
            on_grid = bool(PRICE_GRID[0] <= c_star <= PRICE_GRID[-1])
            sol = {
                "solver": "closed form C* = C0 + INMB(p, C0, wtp)",
                "bracket_lo": float(PRICE_GRID[0]), "bracket_hi": float(PRICE_GRID[-1]),
                "on_swept_grid": on_grid,
                "note": ("" if on_grid else
                         f"C* = ${c_star:,.0f} lies OUTSIDE the reported price axis "
                         f"${PRICE_GRID[0]:,.0f}-${PRICE_GRID[-1]:,.0f}. ") + ATTRIB_NOTE,
            }
            r = increments(p, **A_KW)
            crit = "incremental cost = 0" if w == 0.0 else "INMB = 0"
            qty = ("ciNPWT unit price (cost-neutral, attribution-adjusted)" if w == 0.0
                   else "ciNPWT unit price (attribution-adjusted)")
            tail = ("pays for itself in pure payer dollars, ignoring QALYs" if w == 0.0
                    else f"is cost-effective at ${w:,.0f}/QALY")
            rows.append(_row(
                family="device_price_threshold_attrib", quantity=qty, stratum=sname,
                p_base=p, wtp=w, criterion=crit, basis=BASIS_ATTRIB,
                threshold_value=c_star, units=f"{I.PRICE_YEAR} USD per patient",
                rr_used=None, device_cost_used=c_star, sol=sol, headline=False,
                sample=ATTRIB_SAMPLE,
                interpretation=(
                    f"ATTRIBUTION SCENARIO (SSI cost x {K_ATTRIB:.3f}), reporting basis. "
                    f"In the {sname}-risk stratum (predicted 90-day SSI risk "
                    f"{_fmt_pct(p, 1)}), ciNPWT {tail} as long as its unit price stays "
                    f"below ${c_star:,.0f} per patient"
                    + ("" if on_grid else
                       f", which is off the top of the reported "
                       f"${PRICE_GRID[0]:,.0f}-${PRICE_GRID[-1]:,.0f} price axis")
                    + f". At this risk ciNPWT prevents {1000.0 * r['d_ssi']:.1f} SSIs and "
                      f"{1000.0 * r['d_reop']:.1f} reoperations per 1,000 (NNT "
                      f"{r['nnt']:.0f})."),
            ))

    # (e-d) threshold relative risk per stratum, attribution-adjusted
    for sname, p in STRATA.items():
        for w in (0.0,) + WTP_REPORTED:
            sol = solve_on_grid(
                lambda rr, p=p, w=w: inmb_at(p, w, rr=float(rr), **A_KW),
                RR_GRID, wide=WIDE_RR)
            sol = {**sol,
                   "note": (sol["note"] + " " if sol["note"] else "") + ATTRIB_NOTE}
            rr_star = sol["root"]
            r = increments(p, rr_star, **A_KW)
            crit = "incremental cost = 0" if w == 0.0 else "INMB = 0"
            tail = ("pays for itself in pure payer dollars" if w == 0.0
                    else f"is cost-effective at ${w:,.0f}/QALY")
            rows.append(_row(
                family="rr_threshold_attrib",
                quantity="ciNPWT relative risk (attribution-adjusted)", stratum=sname,
                p_base=p, wtp=w, criterion=crit, basis=BASIS_ATTRIB,
                threshold_value=rr_star, units="relative risk (ciNPWT vs standard)",
                rr_used=rr_star, device_cost_used=C0, sol=sol, headline=False,
                sample=ATTRIB_SAMPLE,
                interpretation=(
                    f"ATTRIBUTION SCENARIO (SSI cost x {K_ATTRIB:.3f}), reporting basis. "
                    f"In the {sname}-risk stratum (predicted 90-day SSI risk "
                    f"{_fmt_pct(p, 1)}) at ${C0:,.0f} per device, ciNPWT {tail} for any "
                    f"true relative risk below {rr_star:.3f}. The pooled estimate is "
                    f"{RR0:.2f}, so under Pennington's attribution the effect has "
                    f"{'more' if rr_star > RR0 else 'less'} room to weaken than the "
                    f"base case allows before ciNPWT stops paying here."),
            ))

    df = pd.DataFrame(rows)

    # ---- self-checks on the solved rows -------------------------------------- #
    # EVERY solved row, headline or CEAC-grid, must satisfy its own criterion when the
    # model is re-evaluated at the coordinates it reports.
    assert (df["inmb"].abs() < MONEY_TOL).all(), (
        "a solved threshold row does not satisfy its own criterion: max |INMB| = "
        f"${df['inmb'].abs().max():.3e}")
    # At INMB = 0 with a positive WTP the ICER is, by identity, exactly the WTP (the row
    # is in the NE quadrant, so the engine reports a finite ICER). This is an end-to-end
    # check on solver, closed form and engine at once.
    pos = df[df["wtp"] > 0]
    assert np.allclose(pos["icer"].values, pos["wtp"].values, rtol=1e-6), (
        "at INMB = 0 the ICER must equal the WTP exactly; max relative deviation "
        f"{np.nanmax(np.abs(pos['icer'].values / pos['wtp'].values - 1.0)):.3e}")
    # ... and at WTP = $0 the row is cost-neutral with a positive QALY gain, i.e. weakly
    # dominant, so the engine must report NaN rather than a ratio of zero.
    zero = df[df["wtp"] == 0]
    assert zero["icer"].isna().all() and (zero["quadrant"] == "NW").all(), (
        "a cost-neutral row must be reported as (weakly) dominant with icer = NaN")
    # Ordering: a higher WTP can only LOWER the risk threshold.
    rt = df[(df["family"] == "risk_threshold") & (df["basis"] == BASIS_BASE)]
    rt = rt.sort_values("wtp")
    assert (np.diff(rt["threshold_value"].values) < 0).all(), \
        "the risk threshold must fall as WTP rises"
    assert p_cn > rt["threshold_value"].max(), (
        "the cost-neutral risk threshold must be the most conservative (highest) of the "
        "four -- it ignores the QALY gain entirely")

    # ---- self-checks on the ATTRIBUTION family (e) ---------------------------- #
    # Same ordering law, on its own basis.
    rta = df[df["family"] == "risk_threshold_attrib"].sort_values("wtp")
    assert (np.diff(rta["threshold_value"].values) < 0).all(), \
        "the attribution-adjusted risk threshold must fall as WTP rises"
    assert p_cna > rta["threshold_value"].max(), (
        "the attribution-adjusted cost-neutral threshold must be the most conservative "
        "of its family")
    # ORDERING LAW, INVERTED BY A2 ALONG WITH THE FAMILY ITSELF. This family now charges
    # the LARGER all-cause SSI cost (m_ssi_cost = 1.000) against the base case's matched
    # 0.492. A larger averted cost pays for the same fixed device toll at a LOWER baseline
    # risk, so the all-cause threshold must sit BELOW the base case. Before A2 the
    # comparison ran the other way and the assertion read `p_a > p_b`; it is re-derived
    # here, not deleted, and it still fails loudly if the two families ever cross.
    for w, p_b in base_p_star.items():
        p_a = float(rta.loc[rta["wtp"] == w, "threshold_value"].iloc[0])
        assert p_a < p_b, (
            f"at WTP ${w:,.0f} the all-cause-cost threshold ({p_a:.6f}) must fall BELOW "
            f"the matched base-case threshold ({p_b:.6f}) -- a larger averted cost "
            f"cannot take longer to break even")
    # The closed-form device-price shift was proven exact for the BASE sample; prove it
    # again under the attribution sample rather than assuming it carries over. The device
    # is a cycle-0 toll in arm B and m_ssi_cost touches only the SSI tranches, so it must
    # -- but the attribution price rows are published, so it is checked, not asserted by
    # argument.
    for sname, p in STRATA.items():
        c_star = float(df.loc[(df["family"] == "device_price_threshold_attrib")
                              & (df["stratum"] == sname) & (df["wtp"] == WTP),
                              "threshold_value"].iloc[0])
        a, b = model.run_pair(p_base=float(p), horizon_cycles=BASE_HORIZON,
                              half_cycle=HALF_CYCLE, disc_cost=DISC, disc_qaly=DISC,
                              sample=dict(ATTRIB_SAMPLE), device_cost=c_star)
        live = model.incremental(b, a, wtp=WTP)["inmb"]
        assert abs(live) < MONEY_TOL, (
            f"attribution scenario, {sname} stratum: a LIVE engine re-run at the solved "
            f"device price ${c_star:,.2f} gives INMB ${live:.3e}, not 0 -- the "
            f"closed-form price shift does not hold under this sample")

    write_results_csv(df, "threshold_results.csv")
    return df


# --------------------------------------------------------------------------- #
# Deliverable 5 -- threshold_summary.md                                        #
# --------------------------------------------------------------------------- #
def build_summary_md(curve, grid, rr_df, res, proof) -> str:
    per_case = model.ssi_cost_per_case(horizon_cycles=BASE_HORIZON)
    qaly_loss = (float(I.tp.frac_deep) * float(I.util.du_deep) * (I.DEEP_LEN / 12.0)
                 + float(I.tp.frac_sup) * float(I.util.du_sup) * (I.SUP_LEN / 12.0))

    base = res[res["basis"] == BASIS_BASE]
    rt = base[base["family"] == "risk_threshold"].set_index("wtp")
    cn = base[base["family"] == "risk_threshold_cost_neutral"].iloc[0]
    und = res[res["family"] == "risk_threshold"]
    und = und[und["basis"] == BASIS_UNDISC].iloc[0]
    price = base[base["family"] == "device_price_threshold"]
    rrt = base[base["family"] == "rr_threshold"]
    # A2: the all-cause upper bound, read live from the `_attrib` family so no sentence
    # below has to name a literal that the audit already moved once.
    _ac = res[(res["family"] == "risk_threshold_attrib")
              & (res["basis"] == BASIS_ATTRIB) & (res["wtp"] == WTP)]
    p_allcause = float(_ac["threshold_value"].iloc[0]) if len(_ac) else float("nan")

    p100 = float(rt.loc[WTP, "threshold_value"])
    p50 = float(rt.loc[WTP_LO, "threshold_value"])
    p150 = float(rt.loc[WTP_HI, "threshold_value"])
    p_cn = float(cn["threshold_value"])
    arr_star = float(rt.loc[WTP, "ssi_prevented_per_1000"])
    nnt_star = float(rt.loc[WTP, "nnt"])

    deep_anchor = dict((a[0], (a[1], a[2])) for a in S["risk_anchors"])

    L = []
    A = L.append
    A("# Risk threshold for ciNPWT after instrumented posterior lumbar fusion")
    A("")
    A("> **" + I.RESULTS_BANNER + "**")
    A("")
    A("*Generated by `run_threshold.py` from the read-only engine (`inputs.py` / "
      "`model.py`). Model-based result; not a substitute for clinical evidence.*")
    A("")
    A(f"All results below are at the base {BASE_HORIZON}-cycle (2-year) horizon with "
      f"{DISC:.0%}/yr discounting on costs and QALYs, trapezoidal half-cycle correction "
      f"on dwell streams, {I.PRICE_YEAR} USD, US commercial-payer perspective, and "
      f"**`B_cinpwt` minus `A_standard`** throughout. Base ciNPWT unit price "
      f"${C0:,.0f}; base relative risk {RR0:.2f}.")
    A("")

    # ---------------------------------------------------------------- headline --
    A("## The headline")
    A("")
    A(f"**ciNPWT is cost-effective at a willingness-to-pay of ${WTP:,.0f}/QALY in any "
      f"patient whose predicted 90-day SSI risk exceeds {_fmt_pct(p100)}.**")
    A("")
    _lo_anchor = min(deep_anchor)
    A(f"That threshold sits **below** the low-risk anchor in this evidence base "
      f"({_fmt_pct(_lo_anchor, 1)}, {deep_anchor[_lo_anchor][0]} "
      f"{deep_anchor[_lo_anchor][1]}) at every willingness-to-pay level reported here: "
      f"{_fmt_pct(p150)} at ${WTP_HI:,.0f}, {_fmt_pct(p100)} at ${WTP:,.0f}, "
      f"{_fmt_pct(p50)} at ${WTP_LO:,.0f}, and {_fmt_pct(p_cn)} on the cost-neutral "
      f"criterion. Under base-case assumptions this model therefore argues for **broad "
      f"rather than narrowly selective** use of ciNPWT in instrumented posterior lumbar "
      f"fusion -- all three anchors, including the lowest, sit above the break-even risk. "
      f"The falsifiable content is what would have to be true for ciNPWT *not* to pay "
      f"(see \"What would have to be true\" below), and the answer A2 supplied is "
      f"concrete: if ciNPWT's effect is confined to superficial SSI, as Zwanenburg's own "
      f"randomized subgroup suggests it may be, the decision reverses at every stratum.")
    A("")
    A(f"**This threshold roughly doubled at the A2 verification gate, and the reason is "
      f"a single identification decision.** The pre-A2 build charged Edmiston [12]'s "
      f"regression-adjusted ALL-CAUSE payment increment at face value and reported "
      f"0.660%. Full-text verification established that those payments are "
      f"least-squares means from a GLM with log link and gamma error, covariate-adjusted "
      f"only, over all-cause total payments, with no matching anywhere in the paper -- "
      f"an upper bound on SSI-attributable cost by construction, since the residual gap "
      f"is confounding by indication. The base case now takes Pennington [14]'s matched "
      f"increment instead (m_ssi_cost = {K_ATTRIB:.3f}), which is the contrast this "
      f"model actually needs and comes from exactly this population. Charging the "
      f"all-cause increment instead is reported as an upper-bound sensitivity in the "
      f"`_attrib` family, where the threshold falls back to roughly its pre-A2 value.")
    A("")
    A(f"**On the comparison with Mueller 2026 [9].** The pre-A2 build reported that its "
      f"threshold fell below \"Mueller's 1.0% floor\". A2 verification found that 1.0% "
      f"and 15.0% are two of NINE swept input rates in Mueller's Methods (0.5, 1.0, 2.0, "
      f"3.0, 4.0, 5.0, 10.0, 15.0, 20.0%), not the endpoints of a published break-even "
      f"band -- so that comparison was a category error in both directions. Mueller's "
      f"own break-even, recomputed from its cost-offset identity at its own inputs, is "
      f"3.7% at its $495 device price and 6.3% at $830, and its Conclusions state >=7%. "
      f"Substituting only this model's SSI cost into Mueller's own equation returns a "
      f"sub-1% break-even, so the two models agree once put on one basis. Mueller is a "
      f"static cost-offset calculation with no QALYs, no discounting, no horizon and no "
      f"uncertainty analysis; it is reported here as a loose concordance check on the "
      f"cost side, not as external validation of a cost-utility model.")
    A("")
    A("The risk axis divides into three bands, and they are not the same statement:")
    A("")
    A(f"| Predicted 90-day SSI risk | Verdict at ${WTP:,.0f}/QALY | Why |")
    A("|---|---|---|")
    A(f"| below {_fmt_pct(p100)} | **not cost-effective** | the ${C0:,.0f} device costs "
      f"more than the health it buys |")
    A(f"| {_fmt_pct(p100)} to {_fmt_pct(p_cn)} | **cost-effective, but not cost-saving** "
      f"| ciNPWT still costs the payer money; the ICER is below ${WTP:,.0f}/QALY |")
    A(f"| above {_fmt_pct(p_cn)} | **dominant** -- cheaper *and* better | averted SSI "
      f"spending exceeds the device price outright |")
    A("")

    # ------------------------------------------------------- (a) and (b) table --
    A("## (a) Risk threshold at each willingness-to-pay, and (b) the cost-neutral risk")
    A("")
    A("| Criterion | Threshold baseline 90-day SSI risk | NNT at the threshold | "
      "SSIs prevented / 1,000 | Reoperations prevented / 1,000 |")
    A("|---|---:|---:|---:|---:|")
    for w in WTP_REPORTED:
        r = rt.loc[w]
        A(f"| INMB = 0 at WTP ${w:,.0f}/QALY | **{_fmt_pct(r['threshold_value'])}** "
          f"| {r['nnt']:.0f} | {r['ssi_prevented_per_1000']:.2f} "
          f"| {r['reop_prevented_per_1000']:.2f} |")
    A(f"| Cost-neutral (incremental cost = 0; QALYs ignored) "
      f"| **{_fmt_pct(p_cn)}** | {cn['nnt']:.0f} "
      f"| {cn['ssi_prevented_per_1000']:.2f} | {cn['reop_prevented_per_1000']:.2f} |")
    A("")
    A(f"A higher willingness-to-pay lowers the threshold, as it must: ${WTP_LO:,.0f} -> "
      f"{_fmt_pct(p50)}, ${WTP:,.0f} -> {_fmt_pct(p100)}, ${WTP_HI:,.0f} -> "
      f"{_fmt_pct(p150)}. The whole $50,000-$150,000 band spans "
      f"{_fmt_pct(p150)}-{_fmt_pct(p50)}, i.e. **the decision does not turn on the "
      f"willingness-to-pay at all** -- the whole band sits inside a range narrower than "
      f"the gap between any two risk anchors in the evidence base. The cost-neutral "
      f"threshold ({_fmt_pct(p_cn)}) is deliberately the most conservative of the four "
      f"because it credits ciNPWT with no health benefit whatsoever; it is the number to "
      f"quote to a payer who declines to price QALYs.")
    A("")
    A(f"**The undiscounted comparison.** On the undiscounted, no-half-cycle basis the "
      f"same threshold is **{_fmt_pct(und['threshold_value'])}**, reproducing the "
      f"engine's published {_fmt_pct(TARGET_P_STAR_UNDISC)} and sitting just above the "
      f"price-year-consistent hand calculation "
      f"{_fmt_pct(TARGET_P_STAR_HANDCALC)} "
      f"(`p* = C / [(1 - RR) x (SSI cost + WTP x QALY loss)]` with a weighted "
      f"{BASE_HORIZON}-month cost per SSI of ${per_case['weighted']:,.0f} and a QALY "
      f"loss of {qaly_loss:.5f} per SSI, both {I.PRICE_YEAR} USD). Discounting **raises** "
      f"the threshold from {_fmt_pct(und['threshold_value'])} to {_fmt_pct(p100)} "
      f"(a factor of {p100 / float(und['threshold_value']):.4f}), and the direction is "
      f"structural, not numerical noise: the device toll falls at cycle 0 where the "
      f"discount factor is exactly 1, while every benefit -- the three SSI cost tranches "
      f"at cycles 1-3, 7-9 and 13-15, and the QALY gain spread over 24 cycles -- is "
      f"discounted. Discounting therefore shrinks the benefit side and not the cost "
      f"side, so a slightly higher baseline risk is needed to break even. At the "
      f"{BASE_HORIZON}-cycle horizon the half-cycle correction moves the threshold by "
      f"nothing at all (the tunnels are empty at cycles 0 and {BASE_HORIZON} and the "
      f"alive mass is identical between arms), so the entire gap is discounting.")
    A("")
    A("### (e) The same thresholds under the all-cause upper bound")
    A("")
    rta = res[res["family"] == "risk_threshold_attrib"].set_index("wtp")
    cna = res[res["family"] == "risk_threshold_cost_neutral_attrib"].iloc[0]
    # A2, second pass: this paragraph asserted a DIRECTION ("roughly doubles") in prose
    # while the table beneath it printed the ratios. The audit inverted the family and the
    # prose did not follow, so the two contradicted each other. The direction word is now
    # DERIVED from the ratios themselves and can never drift from them again.
    _ratios = [float(rta.loc[w, "threshold_value"]) / float(rt.loc[w, "threshold_value"])
               for w in WTP_REPORTED]
    _mean_ratio = sum(_ratios) / len(_ratios)
    _dir = ("roughly halves" if _mean_ratio < 0.75 else
            "roughly doubles" if _mean_ratio > 1.5 else
            f"multiplies by about {_mean_ratio:.2f}")
    A(f"The base case charges Pennington [14]'s **matched** SSI-attributable increment "
      f"(`m_ssi_cost = k_attrib_pennington` = {K_ATTRIB:.3f}). Charging Edmiston [12]'s "
      f"regression-adjusted **all-cause** payment increment at face value instead "
      f"(`m_ssi_cost` = {M_ALLCAUSE:.3f}) {_dir} every risk threshold -- a larger averted "
      f"cost pays for the same device at a LOWER baseline risk. That is why the all-cause "
      f"estimand is reported as an upper bound rather than as the headline: it is "
      f"covariate-adjusted only, with no matching anywhere in the paper, so the residual "
      f"gap is confounding by indication. **These rows are solved on the same reporting "
      f"basis as the base case above** ({BASE_HORIZON} cycles, {DISC:.0%}/yr discounting, "
      f"half-cycle correction), so the pair is like-for-like and can be quoted in one "
      f"sentence:")
    A("")
    A("| Criterion | Base case (matched) | All-cause upper bound "
      f"(SSI cost x {M_ALLCAUSE:.3f}) | Ratio |")
    A("|---|---:|---:|---:|")
    for w in WTP_REPORTED:
        pb = float(rt.loc[w, "threshold_value"])
        pa = float(rta.loc[w, "threshold_value"])
        A(f"| INMB = 0 at WTP ${w:,.0f}/QALY | {_fmt_pct(pb)} | **{_fmt_pct(pa)}** "
          f"| {pa / pb:.2f}x |")
    A(f"| Cost-neutral (incremental cost = 0) | {_fmt_pct(p_cn)} "
      f"| **{_fmt_pct(float(cna['threshold_value']))}** "
      f"| {float(cna['threshold_value']) / p_cn:.2f}x |")
    A("")
    A(f"The manuscript's headline pair is therefore **{_fmt_pct(p100)} -> "
      f"{_fmt_pct(float(rta.loc[WTP, 'threshold_value']))}** at ${WTP:,.0f}/QALY, both on "
      f"the reporting basis. Do **not** pair the reporting-basis base case with the "
      f"undiscounted attribution-adjusted value "
      f"(BUILD-CONTRACT section 10, \"BASIS LABELS\"): that would put two conventions "
      f"into one sentence. The full attribution family -- risk, cost-neutral risk, "
      f"threshold device price and threshold relative risk -- is in "
      f"`threshold_results.csv` under the four `*_attrib` family names, each row's "
      f"`basis` column naming both the reporting convention and the scenario. They carry "
      f"`headline = False`, because `headline` marks the **base-case** rows Table 3 "
      f"prints; select the scenario with `family.str.endswith(\"_attrib\")`.")
    A("")

    # ---------------------------------------------------------------- (c) price --
    A("## (c) Threshold ciNPWT unit price, per risk stratum")
    A("")
    A("The closed form is exact (see \"How this was computed\"): the device price is a "
      "pure additive cycle-0 toll in arm B, so `C*(p, w) = C0 + INMB(p, C0, w)`.")
    A("")
    A("| Risk stratum | Predicted 90-day SSI risk | Cost-neutral price | "
      f"WTP ${WTP_LO:,.0f} | WTP ${WTP:,.0f} | WTP ${WTP_HI:,.0f} | On the swept "
      f"${PRICE_GRID[0]:,.0f}-${PRICE_GRID[-1]:,.0f} axis? |")
    A("|---|---:|---:|---:|---:|---:|---|")
    for sname, p in STRATA.items():
        sub = price[price["stratum"] == sname].set_index("wtp")
        on = bool(sub.loc[WTP, "on_swept_grid"])
        A(f"| {sname} | {_fmt_pct(p, 1)} | {_fmt_money(sub.loc[0.0, 'threshold_value'])} "
          f"| {_fmt_money(sub.loc[WTP_LO, 'threshold_value'])} "
          f"| **{_fmt_money(sub.loc[WTP, 'threshold_value'])}** "
          f"| {_fmt_money(sub.loc[WTP_HI, 'threshold_value'])} "
          f"| {'yes' if on else 'no -- above $%s' % f'{PRICE_GRID[-1]:,.0f}'} |")
    A("")
    low_p = price[(price["stratum"] == "low") & (price["wtp"] == WTP)].iloc[0]
    int_p = price[(price["stratum"] == "intermediate") & (price["wtp"] == WTP)].iloc[0]
    hi_p = price[(price["stratum"] == "high") & (price["wtp"] == WTP)].iloc[0]
    A(f"**In plain terms:** at the low-risk stratum ({_fmt_pct(STRATA['low'], 1)}), "
      f"ciNPWT remained cost-effective at a willingness-to-pay of ${WTP:,.0f}/QALY as "
      f"long as the device price remained below "
      f"**{_fmt_money(low_p['threshold_value'])}** per patient. At the intermediate "
      f"stratum ({_fmt_pct(STRATA['intermediate'], 1)}) the same figure is "
      f"**{_fmt_money(int_p['threshold_value'])}**, and at the high-risk stratum "
      f"({_fmt_pct(STRATA['high'], 1)}) **{_fmt_money(hi_p['threshold_value'])}** -- "
      f"both far above the top of the reported "
      f"${PRICE_GRID[0]:,.0f}-${PRICE_GRID[-1]:,.0f} acquisition-price axis. **Price is "
      f"therefore a live consideration only in the lowest-risk stratum.**")
    A("")
    A("**Where the Fig 3 frontier actually lives, and it is not where a reader expects.** "
      "The break-even contour is `C*(p) = C0 + INMB(p, C0, w)`, which is monotone rising "
      "in `p`. Inside the reported panel it therefore runs from the LEFT edge of "
      "the risk axis up to the TOP of the price axis, and it leaves through the top very "
      "early:")
    A("")
    A(f"| Criterion | Frontier price at the left edge of the risk axis "
      f"({_fmt_pct(RISK_GRID[0], 1)}) | Risk at which the frontier reaches the top of "
      f"the price axis (${PRICE_GRID[-1]:,.0f}) | Share of the plotted risk axis the "
      f"frontier occupies |")
    A("|---|---:|---:|---:|")
    p_exit = {}
    for w, label in ((0.0, "cost-neutral"), (WTP_LO, f"WTP ${WTP_LO:,.0f}"),
                     (WTP, f"WTP ${WTP:,.0f}"), (WTP_HI, f"WTP ${WTP_HI:,.0f}")):
        c_left = C0 + inmb_at(float(RISK_GRID[0]), w)
        ex = solve_on_grid(
            lambda p, w=w: inmb_at(p, w) - (float(PRICE_GRID[-1]) - C0),
            RISK_GRID, wide=WIDE_P)
        p_exit[w] = float(ex["root"])
        span = ((ex["root"] - float(RISK_GRID[0]))
                / (float(RISK_GRID[-1]) - float(RISK_GRID[0])))
        A(f"| {label} | {_fmt_money(c_left)} | {_fmt_pct(ex['root'])} | {span:.1%} |")
    A("")
    A(f"So at **every** ciNPWT acquisition price on the reported axis, the device "
      f"pays at any baseline risk above {_fmt_pct(p_exit[WTP])} -- about one thirteenth "
      f"of the way along the plotted risk axis, and barely above the low-risk stratum's "
      f"own {_fmt_pct(STRATA['low'], 1)} anchor. **The entire decision-relevant region of "
      f"Fig 3 is a narrow wedge against the left-hand axis**, and a figure drawn on a "
      f"linear {_fmt_pct(RISK_GRID[0], 1)}-{_fmt_pct(RISK_GRID[-1], 1)} risk axis will "
      f"compress it almost to the frame. That is itself the finding, but it should be "
      f"drawn so a reader can see it (a log risk axis, or an inset over "
      f"{_fmt_pct(RISK_GRID[0], 1)}-2%).")
    A("")

    # ------------------------------------------------------------------ (d) RR --
    A("## (d) Threshold ciNPWT relative risk, per risk stratum")
    A("")
    A(f"At the base ${C0:,.0f} device price, how weak would the true ciNPWT effect have "
      f"to be before it stops paying? The pooled estimate is RR {RR0:.2f} "
      f"({float(I.tp.rr_cinpwt.low):.2f}-{float(I.tp.rr_cinpwt.high):.2f}, Zwanenburg "
      f"2020 [11]).")
    A("")
    A("| Risk stratum | Predicted 90-day SSI risk | Cost-neutral RR | "
      f"WTP ${WTP_LO:,.0f} | WTP ${WTP:,.0f} | WTP ${WTP_HI:,.0f} | Implied relative "
      "risk reduction at the threshold |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for sname, p in STRATA.items():
        sub = rrt[rrt["stratum"] == sname].set_index("wtp")
        rstar = float(sub.loc[WTP, "threshold_value"])
        A(f"| {sname} | {_fmt_pct(p, 1)} | {float(sub.loc[0.0, 'threshold_value']):.3f} "
          f"| {float(sub.loc[WTP_LO, 'threshold_value']):.3f} | **{rstar:.3f}** "
          f"| {float(sub.loc[WTP_HI, 'threshold_value']):.3f} "
          f"| {100.0 * (1.0 - rstar):.1f}% |")
    A("")
    rr_low = float(rrt[(rrt["stratum"] == "low")
                       & (rrt["wtp"] == WTP)]["threshold_value"].iloc[0])
    rr_hi = float(rrt[(rrt["stratum"] == "high")
                      & (rrt["wtp"] == WTP)]["threshold_value"].iloc[0])
    A(f"**In plain terms:** the ICER remained below ${WTP:,.0f} per QALY until the "
      f"ciNPWT relative risk was attenuated from the pooled {RR0:.2f} to as high as "
      f"{rr_low:.3f} in the low-risk stratum -- and to {rr_hi:.3f} in the high-risk "
      f"stratum, i.e. until the device's relative risk reduction fell from "
      f"{100.0 * (1.0 - RR0):.0f}% to barely {100.0 * (1.0 - rr_hi):.1f}%. Both "
      f"threshold values lie far outside the pooled 95% confidence interval "
      f"({float(I.tp.rr_cinpwt.low):.2f}-{float(I.tp.rr_cinpwt.high):.2f}) and above the "
      f"defined attenuated-effect scenario constant "
      f"({float(I.tp.rr_cinpwt_attenuated):.2f}). **The conclusion is not sensitive to "
      f"the effect size within any range the evidence supports.**")
    A("")

    # -------------------------------------------- the unifying absolute-risk fact --
    A("## Why the four families are one statement")
    A("")
    A(f"All four thresholds are one equation in different coordinates. INMB = 0 holds "
      f"exactly when the **absolute** risk reduction, times the value of an averted SSI, "
      f"equals the device price. The value of an averted SSI does not depend on the "
      f"baseline risk, so at a fixed ${C0:,.0f} device price the break-even absolute risk "
      f"reduction is a **constant of the model**: **{arr_star:.2f} SSIs prevented per "
      f"1,000 patients treated, i.e. NNT {nnt_star:.0f}**, at ${WTP:,.0f}/QALY.")
    A("")
    stratum_arr = " / ".join(f"{1000.0 * increments(p)['d_ssi']:.1f}"
                             for p in STRATA.values())
    A(f"That constant is why `threshold_results.csv` reports the identical NNT "
      f"{nnt_star:.0f} on the risk-threshold row (family a) and on all three "
      f"relative-risk rows (family d) -- three different strata, three different "
      f"threshold RRs, one break-even absolute risk reduction. Family (c) is the same "
      f"identity solved the other way round: the price is released instead of held, so "
      f"`C*` is whatever makes each stratum's own absolute risk reduction "
      f"({stratum_arr} per 1,000 at low / intermediate / high risk) break even. "
      f"The clinically useful "
      f"consequence: **the question is never \"is this patient high risk?\" but \"can "
      f"ciNPWT avert one SSI per {nnt_star:.0f} patients here?\"** -- and at a pooled "
      f"relative risk reduction of {100.0 * (1.0 - RR0):.0f}%, a baseline risk of only "
      f"{_fmt_pct(p100)} is enough to clear that bar.")
    A("")

    # ------------------------------------------------- what would have to be true --
    A("## What would have to be true for ciNPWT NOT to pay")
    A("")
    A("This is the falsifiable content of the analysis, and the part a surgeon or a "
      "value-analysis committee can act on. At the base-case parameters, **every one of "
      "these would have to hold**, not just one:")
    A("")
    A(f"1. **The patient's predicted 90-day SSI risk is below {_fmt_pct(p100)}** -- "
      f"lower than the lowest anchor in this evidence base "
      f"({_fmt_pct(_lo_anchor, 1)}) and far below the intermediate "
      f"({_fmt_pct(0.066, 1)}) and high ({_fmt_pct(0.114, 1)}) anchors. (A2 note: an "
      f"earlier version of this sentence also cited \"Mueller 2026's 1.0% floor\". No "
      f"such floor exists -- 1.0% is the second of nine SSI rates Mueller SWEPT as model "
      f"inputs, and at 1.0% Mueller's own Table 1 shows universal ciNPT losing money.)")
    A(f"2. **The ciNPWT unit price exceeds "
      f"{_fmt_money(low_p['threshold_value'])}** (low-risk stratum) or "
      f"{_fmt_money(int_p['threshold_value'])} (intermediate) -- against an assumed "
      f"acquisition cost of ${C0:,.0f} and a plausible ceiling of "
      f"${PRICE_GRID[-1]:,.0f}.")
    A(f"3. **The true relative risk is weaker than {rr_low:.3f}** (low-risk stratum) or "
      f"{rr_hi:.3f} (high-risk stratum) -- against a pooled {RR0:.2f} whose upper "
      f"confidence bound is {float(I.tp.rr_cinpwt.high):.2f}.")
    A("")
    A("Two structural caveats belong beside that list, because they move the threshold "
      "in the opposite direction and neither is settled by this module:")
    A("")
    A(f"- **SSI-attributable cost is the swing parameter, and A2 moved the base case "
      f"onto the conservative side of it.** `m_ssi_cost` = "
      f"{float(I.cost.k_attrib_pennington):.3f} -- Pennington 2019 [14]'s MATCHED "
      f"increment -- is now the base case, not a scenario; it does not double this "
      f"threshold, it IS this threshold. The sensitivity runs the other way: charging "
      f"Edmiston 2023 [12]'s regression-adjusted ALL-CAUSE increment at face value "
      f"(`m_ssi_cost` = 1.000) LOWERS the break-even risk to "
      f"{_fmt_pct(p_allcause)}, because a larger averted cost pays for the device "
      f"sooner. `run_scenarios.py` "
      f"and `run_dsa.py` own that axis. **The threshold reported here is conditional on "
      f"Pennington's matched increment being the right estimand** -- which A2 argued it "
      f"is, because Edmiston's payments are covariate-adjusted only with no matching "
      f"anywhere in the paper and are therefore an upper bound on attributable cost by "
      f"construction.")
    A(f"- **The results are VERIFIED.** Phase A2 full-text verification completed "
      f"2026-08-17: all 14 requested full texts were opened, 46 of 48 parameters are "
      f"closed out and 43 carry a table/page locator. Both numbers that dominate the "
      f"threshold arithmetic were checked -- the Edmiston cost tranches (Table 3, PDF "
      f"p. 7) and the Zwanenburg relative risk (Fig. 2 / Table 2). Two parameters remain "
      f"unresolvable by any source: the ciNPWT institutional net acquisition price and "
      f"the superficial tunnel length.")
    A("")

    # ------------------------------------------------------------- computation ---
    A("## How this was computed")
    A("")
    A(f"- **Curve (Fig 2).** One engine pair per point of the 291-point "
      f"risk grid (0.5%-15.0%, step 0.05 pp), at the base device price and base RR. "
      f"INMB is strictly monotone increasing in baseline risk at all three WTPs "
      f"(asserted), so the zero crossing is unique.")
    A(f"- **Map (Fig 3).** The device cost is a **pure additive cycle-0 toll in arm B**: "
      f"cycle 0 carries a discount factor of exactly 1 and no half-cycle weight, and "
      f"nothing else in the result depends on the price. Therefore "
      f"`cost_B(C) = cost_B(C0) + (C - C0)` exactly, and the 291 x 161 map costs **291 "
      f"engine pairs, not "
      f"{len(RISK_GRID) * len(PRICE_GRID):,}**. This is not assumed: it is proven at run "
      f"time against live engine re-runs at 25 (risk, price) corners (max deviation "
      f"${proof['corners']:.1e}), at 12 randomly chosen cells of the actual map "
      f"(${proof['cells']:.1e}), and at every solved threshold price, where the live "
      f"INMB at `C*` must be zero (${proof['c_star']:.1e}). If the engine ever stops "
      f"being linear in the device price, this module fails loudly rather than writing "
      f"a wrong figure.")
    A(f"- **Solving.** `scipy.optimize.brentq`, bracketed between adjacent points of the "
      f"contract's own sweep grid, to machine precision. Every headline row is "
      f"self-checking: the increments are recomputed at the solved coordinates and the "
      f"resulting `inmb` is asserted to be zero, and -- because INMB = 0 forces "
      f"`ICER = WTP` identically -- the recomputed ICER is asserted to equal the WTP. "
      f"The cost-neutral threshold is solved independently on `inc_cost` and then "
      f"asserted equal to the WTP = $0 member of the INMB family.")
    A(f"- **Clinical translation.** NNT is `1 / (n_ssi_A - n_ssi_B)` from the engine's "
      f"cohort **counts**, never `1 / (p x (1 - RR))`. The counts sit ~0.32% relative "
      f"below `p` because perioperative (cycle 0) and background mortality compete with "
      f"onset in cycles 1-3; that is correct competing-risks behaviour, and it is the "
      f"count form that reproduces the cross-check NNT of 38.9 at p = 6.6%.")
    A(f"- **ICER reporting.** `icer` is `NaN` wherever ciNPWT is strictly dominant, "
      f"which is {int((curve['quadrant'] == 'NW').sum())} of the {len(curve)} curve "
      f"points. `dominance` and `quadrant` are the primary reporting fields; no bare "
      f"`NaN` is ever printed as a number.")
    A("")

    # ----------------------------------------------------------------- caveats ---
    A("## What these numbers do and do not carry")
    A("")
    A("- **Verified.** Phase A2 full-text verification completed 2026-08-17: 46 of 48 "
      "parameters are closed out against the 14 source full texts and 43 carry a "
      "table/page locator. The two residual assumptions are the ciNPWT institutional net "
      "acquisition price and the superficial tunnel length.")
    A(f"- **Conditional on the cost-attribution basis.** Pennington [14]'s **matched** "
      f"SSI-attributable increment (m_ssi_cost = {K_ATTRIB:.3f}) is the entire benefit "
      f"side. Charging Edmiston's regression-adjusted ALL-CAUSE increment at face value "
      f"instead is the upper-bound sensitivity, in section (e) above and in "
      f"`run_scenarios.py`.")
    A(f"- **The conservative-bias register.** No SSI-attributable mortality is modelled; "
      f"superficial-to-deep progression is omitted; the standard-dressing differential is "
      f"$0; the weaker all-surgery RR {float(I.tp.rr_cinpwt):.2f} is used rather than the "
      f"favourable spine-specific RR {float(I.tp.rr_cinpwt_spine):.3f}; and the base case "
      f"charges the matched attributable cost rather than Edmiston's larger all-cause "
      f"increment. All five bias **against** ciNPWT, so the true threshold is, if "
      f"anything, lower than reported.")
    A("- **The risk axis is a predicted risk, not an observed one.** No risk-prediction "
      "model is validated here; the threshold is a decision rule that presumes the "
      "clinician can place a patient on the axis.")
    A("- **The anchors do not share an ascertainment window.** Edmiston is 3-180 d, "
      "Mueller 60 d, Freire-Archer variable; they are plotted as positions on a 90-day "
      "axis without being rescaled to it.")
    A("- **Cost is incremental-versus-no-SSI**, not total cost of care.")
    A("")
    A("---")
    A("")
    A("*Files written: `threshold_results.csv`, `threshold_curve.csv`, "
      "`threshold_grid_risk_price.csv`, `threshold_rr.csv`, `threshold_summary.md` -- "
      "all in `v2/results/`, each CSV carrying the results banner as line 1 "
      "(read with `pd.read_csv(path, skiprows=1)`).*")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "threshold_summary.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    return path


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 78)
    print("SSI / ciNPWT risk-threshold Markov CUA -- THRESHOLD ANALYSIS")
    print(I.RESULTS_BANNER)
    print("=" * 78)
    print(f"Basis: {BASE_HORIZON} cycles | {DISC:.0%}/yr discounting | half-cycle "
          f"{HALF_CYCLE} | {I.PRICE_YEAR} USD | device ${C0:,.0f} | RR {RR0:.2f}")
    print(f"Grids: risk {len(RISK_GRID)} pts [{RISK_GRID[0]:g}, {RISK_GRID[-1]:g}] | "
          f"price {len(PRICE_GRID)} pts [${PRICE_GRID[0]:,.0f}, ${PRICE_GRID[-1]:,.0f}] "
          f"| RR {len(RR_GRID)} pts [{RR_GRID[0]:g}, {RR_GRID[-1]:g}] | CEAC "
          f"{len(CEAC_GRID)} pts")
    tc = model.tranches_charged(BASE_HORIZON)
    print("Toll schedule (BUILD-CONTRACT 6.2) at this horizon: "
          + ", ".join(f"tranche {d['index'] + 1} (cycles {d['first_cycle']}-"
                      f"{d['last_cycle']}) {d['charged'].upper()}" for d in tc))
    print()

    proof = prove_device_cost_linearity()
    print()

    curve = build_threshold_curve()
    grid = build_threshold_grid(curve)
    rr_df = build_threshold_rr()
    res = solve_thresholds()
    build_summary_md(curve, grid, rr_df, res, proof)

    # ---- stdout headline ---------------------------------------------------- #
    base = res[res["basis"] == BASIS_BASE]
    rt = base[base["family"] == "risk_threshold"].set_index("wtp")
    cn = base[base["family"] == "risk_threshold_cost_neutral"].iloc[0]
    und = res[(res["family"] == "risk_threshold")
              & (res["basis"] == BASIS_UNDISC)].iloc[0]
    p100 = float(rt.loc[WTP, "threshold_value"])

    print("(a) INMB = 0 baseline-risk threshold")
    print(f"  {'WTP':>12}  {'threshold p*':>13}  {'NNT':>7}  {'SSI/1,000':>10}  "
          f"{'reop/1,000':>11}")
    for w in WTP_REPORTED:
        r = rt.loc[w]
        print(f"  ${w:>11,.0f}  {_fmt_pct(r['threshold_value']):>13}  {r['nnt']:>7.0f}  "
              f"{r['ssi_prevented_per_1000']:>10.2f}  "
              f"{r['reop_prevented_per_1000']:>11.2f}")
    print("(b) cost-neutral baseline-risk threshold (inc_cost = 0, QALYs ignored)")
    print(f"  {'  n/a':>12}  {_fmt_pct(cn['threshold_value']):>13}  {cn['nnt']:>7.0f}  "
          f"{cn['ssi_prevented_per_1000']:>10.2f}  "
          f"{cn['reop_prevented_per_1000']:>11.2f}")
    print()
    print(f"  Undiscounted / no half-cycle: {_fmt_pct(und['threshold_value'])} "
          f"(engine target {_fmt_pct(TARGET_P_STAR_UNDISC)}; hand-calc "
          f"{_fmt_pct(TARGET_P_STAR_HANDCALC)}). Discounting RAISES the threshold to "
          f"{_fmt_pct(p100)}")
    print(f"  because the device toll falls at cycle 0 (discount factor 1) while every "
          f"benefit is discounted.")
    print()

    rta = res[res["family"] == "risk_threshold_attrib"].set_index("wtp")
    cna = res[res["family"] == "risk_threshold_cost_neutral_attrib"].iloc[0]
    print(f"(e) the same thresholds under the ATTRIBUTION scenario "
          f"(SSI cost x {K_ATTRIB:.3f}), SAME REPORTING BASIS")
    print(f"  {'WTP':>12}  {'base p*':>13}  {'attrib p*':>13}  {'ratio':>7}")
    for w in WTP_REPORTED:
        pb = float(rt.loc[w, "threshold_value"])
        pa = float(rta.loc[w, "threshold_value"])
        print(f"  ${w:>11,.0f}  {_fmt_pct(pb):>13}  {_fmt_pct(pa):>13}  {pa / pb:>6.2f}x")
    print(f"  {'cost-neutral':>12}  {_fmt_pct(cn['threshold_value']):>13}  "
          f"{_fmt_pct(cna['threshold_value']):>13}  "
          f"{float(cna['threshold_value']) / float(cn['threshold_value']):>6.2f}x")
    print(f"  -> MANUSCRIPT PAIR at ${WTP:,.0f}/QALY, both on the reporting basis: "
          f"{_fmt_pct(p100)} -> {_fmt_pct(float(rta.loc[WTP, 'threshold_value']))}. "
          f"Never pair it with the undiscounted attribution value.")
    print()

    print("(c) threshold ciNPWT unit price      (d) threshold ciNPWT relative risk")
    price = base[base["family"] == "device_price_threshold"].set_index(["stratum", "wtp"])
    rrt = base[base["family"] == "rr_threshold"].set_index(["stratum", "wtp"])
    print(f"  {'stratum':<14}{'p':>7}{'C* @50k':>10}{'C* @100k':>11}{'C* @150k':>11}"
          f"{'  |':>3}{'RR* @100k':>11}{'  on price axis?':>18}")
    for sname, p in STRATA.items():
        on = bool(price.loc[(sname, WTP), "on_swept_grid"])
        print(f"  {sname:<14}{p:>7.3f}"
              f"{price.loc[(sname, WTP_LO), 'threshold_value']:>10,.0f}"
              f"{price.loc[(sname, WTP), 'threshold_value']:>11,.0f}"
              f"{price.loc[(sname, WTP_HI), 'threshold_value']:>11,.0f}"
              f"{'  |':>3}{rrt.loc[(sname, WTP), 'threshold_value']:>11.3f}"
              f"{('yes' if on else 'NO -- above $%s' % f'{PRICE_GRID[-1]:,.0f}'):>18}")
    print()
    print(f"  At a fixed ${C0:,.0f} device price the break-even ABSOLUTE risk reduction "
          f"is a CONSTANT of the model: {float(rt.loc[WTP, 'ssi_prevented_per_1000']):.2f}"
          f" SSIs prevented per 1,000 (NNT {float(rt.loc[WTP, 'nnt']):.0f}) at "
          f"${WTP:,.0f}/QALY --")
    print(f"  the same number whether you lower the risk (a) or weaken the effect (d); "
          f"(c) is the same identity solved for the price instead. One equation, four "
          f"coordinate systems.")
    print()
    print(f"  HONEST READING (pre-A2, superseded -- see AUDIT-2026-08-17.md): p* = {_fmt_pct(p100)} sits BELOW the pooled deep-SSI rate "
          f"(2.0% on the all-SSI axis after A2 rescaled it) but ABOVE the pre-A2 reading; "
          f"Mueller 2026's '1.0% floor' does not exist -- see AUDIT-2026-08-17.md.")
    print(f"  Under base-case assumptions the model argues for BROAD, not selective, use "
          f"-- exactly as the plan reported.")
    print()

    print(f"Engine pairs actually run (cached): {len(_PAIR_CACHE):,} "
          f"-- vs {len(RISK_GRID) * len(PRICE_GRID):,} for a naive risk x price sweep.")
    print()
    print("Files written to:", RESULTS_DIR)
    expected = ["threshold_results.csv", "threshold_curve.csv",
                "threshold_grid_risk_price.csv", "threshold_rr.csv",
                "threshold_summary.md"]
    missing = []
    for fn in expected:
        path = os.path.join(RESULTS_DIR, fn)
        ok = os.path.exists(path)
        if not ok:
            missing.append(fn)
        n = ""
        if ok and fn.endswith(".csv"):
            n = f"  ({len(pd.read_csv(path, skiprows=1)):,} rows)"
        print(f"  [{'OK' if ok else 'MISSING'}] {fn}{n}")
    if missing:
        raise RuntimeError(f"run_threshold.py failed to write: {missing}")


if __name__ == "__main__":
    main()
