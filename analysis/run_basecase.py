"""
run_basecase.py -- Deterministic base-case run for the SSI / ciNPWT risk-threshold
Markov cost-utility analysis.

Imports the engine (`inputs.py` / `model.py`) READ-ONLY and writes the six base-case
deliverables required by BUILD-CONTRACT section 8 to `v2/results/`:

    basecase_results.csv   one row per (risk stratum x horizon x arm), plus the
                           B-minus-A incremental block, NNT, SSIs prevented / 1,000 and
                           reoperations prevented / 1,000
    basecase_summary.md    verification statement, per-stratum narrative, and the
                           explicit per-horizon toll table (section 6.2) so the 90-day
                           and 1-year numbers cannot be mistaken for errors
    trace_A_standard.csv   24-cycle base-case cohort trace at the INTERMEDIATE stratum,
    trace_B_cinpwt.csv     collapsed to the 5 CLINICAL_STATES (never the raw slots)
    accrual.csv            share of the DISCOUNTED total accruing after 6 months
    cum_events.csv         cumulative expected event counts per 1,000 patients

Every CSV carries `inputs.RESULTS_BANNER` as a '#'-prefixed LINE 1, followed by the
header row -- so downstream readers use `pd.read_csv(path, skiprows=1)` (BUILD-CONTRACT
section 7; do NOT use `comment='#'` on a results CSV).

Run with cwd = v2/analysis/ :  python3 run_basecase.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import inputs as I
import model

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V1_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
RESULTS_DIR = os.path.join(V1_DIR, "results")

STRATA = I.SETTINGS["strata"]                     # low 0.015 / intermediate 0.066 / high 0.114
HORIZONS = tuple(I.SETTINGS["horizons"])          # (3, 12, 24, 60)
BASE_HORIZON = int(I.SETTINGS["horizon_cycles"])  # 24
BASE_STRATUM = "intermediate"
WTP = float(I.SETTINGS["wtp_primary"])            # 100,000
WTP_LO, WTP_HI = (float(w) for w in I.SETTINGS["wtp_report"])   # 50,000 / 150,000

HORIZON_LABELS = {3: "90 d", 12: "1 y", 24: "2 y (base case)", 60: "5 y"}


# --------------------------------------------------------------------------- #
# I/O helpers                                                                  #
# --------------------------------------------------------------------------- #
def write_results_csv(df: pd.DataFrame, filename: str) -> str:
    """Write a results CSV with `inputs.RESULTS_BANNER` as a '#'-prefixed line 1."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", newline="") as fh:
        fh.write("# " + I.RESULTS_BANNER + "\n")
        df.to_csv(fh, index=False)
    return path


def _horizon_label(h: int) -> str:
    return HORIZON_LABELS.get(int(h), f"{int(h)} cycles")


def _fmt_money(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    sign = "-" if x < 0 else ""
    a = abs(x)
    return f"{sign}${a:,.0f}" if a >= 1 else f"{sign}${a:,.2f}"


def _fmt_icer(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a (dominant)"
    return f"${x:,.0f}/QALY"


def _fmt_share(x) -> str:
    """A share, or an explicit '--' when the incremental denominator vanished."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a (increment ~ 0)"
    return f"{x:.1%}"


# --------------------------------------------------------------------------- #
# Accrual arithmetic (plan Decision 7)                                         #
# --------------------------------------------------------------------------- #
#: The engine's `share_*_after_6mo` convention: strictly AFTER cycle 6, i.e. the slice
#: `stream[7:]`. A structural index (month 6 of a monthly cycle), not a parameter.
AFTER_6MO_FROM_CYCLE = 7

#: Below this the incremental total is indistinguishable from zero relative to the arms'
#: own scale, and "the share of the increment accruing after 6 months" is a ratio of two
#: rounding errors. Emit NaN and label it, rather than a huge or sign-flipped share.
INC_DENOM_REL_EPS = 1e-9


def _inc_share_after_6mo(stream_b, stream_a) -> float:
    """Share of the INCREMENTAL (B minus A) discounted stream accruing strictly after
    cycle 6.

    Plan Decision 7 asks for "the share of **incremental** cost and QALY accruing after
    6 months". The engine reports only the per-arm shares (`share_cost_after_6mo` /
    `share_qaly_after_6mo`), and BUILD-CONTRACT section 9 forbids figure/table scripts
    from re-running the model, so this is the only place the incremental version can be
    computed -- from the two arms' `cost_stream` / `qaly_stream`, which are already in
    hand here. Both streams are DISCOUNTED and post-half-cycle and sum exactly to their
    arm's `cost` / `qaly`, so their difference sums exactly to the incremental total.

    Returns NaN when the incremental total is within `INC_DENOM_REL_EPS` of zero relative
    to the arms' own scale (a near-zero denominator would otherwise produce a meaningless
    ratio); callers must format it as "n/a", never as a number.
    """
    b = np.asarray(stream_b, dtype=float)
    a = np.asarray(stream_a, dtype=float)
    inc = b - a
    total = float(inc.sum())
    scale = float(np.abs(a).sum() + np.abs(b).sum())
    if abs(total) <= INC_DENOM_REL_EPS * max(scale, 1.0):
        return float("nan")
    return float(inc[AFTER_6MO_FROM_CYCLE:].sum() / total)


# --------------------------------------------------------------------------- #
# Deliverable 1 -- basecase_results.csv                                        #
# --------------------------------------------------------------------------- #
def build_basecase_results():
    """One row per (stratum x horizon x arm).

    The incremental block is B_cinpwt MINUS A_standard and is therefore a property of the
    (stratum x horizon) CELL, not of an arm; it is repeated identically on both arm rows
    so a downstream filter on either arm still sees it. Never sum an incremental column
    across arms.
    """
    rows = []
    cache = {}
    for sname, p in STRATA.items():
        for h in HORIZONS:
            res_a, res_b = model.run_pair(p_base=p, horizon_cycles=h)
            cache[(sname, h)] = (res_a, res_b)
            inc = model.incremental(res_b, res_a, wtp=WTP)
            inc_lo = model.incremental(res_b, res_a, wtp=WTP_LO)
            inc_hi = model.incremental(res_b, res_a, wtp=WTP_HI)

            d_ssi = res_a["n_ssi"] - res_b["n_ssi"]
            d_deep = res_a["n_ssi_deep"] - res_b["n_ssi_deep"]
            d_reop = res_a["n_reop"] - res_b["n_reop"]
            d_pers = res_a["n_persistent"] - res_b["n_persistent"]
            nnt = (1.0 / d_ssi) if d_ssi > 0 else np.nan

            for res in (res_a, res_b):
                rows.append({
                    "stratum": sname,
                    "p_base": p,
                    "horizon_cycles": int(h),
                    "horizon_label": _horizon_label(h),
                    "arm": res["arm"],
                    "arm_label": I.ARM_LABELS[res["arm"]],
                    "rr_used": res["rr_used"],
                    "device_cost_used": res["device_cost_used"],
                    "cost": res["cost"],
                    "qaly": res["qaly"],
                    "cost_undisc": res["cost_undisc"],
                    "qaly_undisc": res["qaly_undisc"],
                    "life_years": res["life_years"],
                    "n_ssi": res["n_ssi"],
                    "n_ssi_superficial": res["n_ssi_superficial"],
                    "n_ssi_deep": res["n_ssi_deep"],
                    "n_reop": res["n_reop"],
                    "n_persistent": res["n_persistent"],
                    # ---- incremental block: B_cinpwt MINUS A_standard ----
                    "inc_cost": inc["inc_cost"],
                    "inc_qaly": inc["inc_qaly"],
                    "icer": inc["icer"],
                    "inmb_at_100k": inc["inmb"],
                    "inmb_at_50k": inc_lo["inmb"],
                    "inmb_at_150k": inc_hi["inmb"],
                    "dominance": inc["dominance"],
                    "quadrant": inc["quadrant"],
                    # ---- clinical translation ----
                    "nnt": nnt,
                    "ssi_prevented_per_1000": 1000.0 * d_ssi,
                    "ssi_deep_prevented_per_1000": 1000.0 * d_deep,
                    "reop_prevented_per_1000": 1000.0 * d_reop,
                    "persistent_prevented_per_1000": 1000.0 * d_pers,
                })
    df = pd.DataFrame(rows)
    write_results_csv(df, "basecase_results.csv")
    return df, cache


# --------------------------------------------------------------------------- #
# Deliverable 2 -- trace_<arm>.csv (base horizon, intermediate stratum)        #
# --------------------------------------------------------------------------- #
def build_traces(cache):
    """24-cycle base-case traces at the INTERMEDIATE stratum, collapsed to the 5
    `inputs.CLINICAL_STATES`. Raw slot traces are never written: traces of different
    tunnel widths must never be compared, and the 5-state form is the only comparable
    one (BUILD-CONTRACT section 2 amendment)."""
    res_a, res_b = cache[(BASE_STRATUM, BASE_HORIZON)]
    age0 = float(I.mortality.age_start)
    cyc_years = float(I.SETTINGS["cycle_years"])
    out = {}
    for res in (res_a, res_b):
        coll = model.collapse_trace(res["trace"])
        n = coll.shape[0]
        cyc = np.arange(n)
        df = pd.DataFrame(coll, columns=list(I.CLINICAL_STATES))
        df.insert(0, "alive", 1.0 - coll[:, I.CLINICAL_STATES.index("Dead")])
        df.insert(0, "age", age0 + cyc * cyc_years)
        df.insert(0, "month", cyc)
        df.insert(0, "cycle", cyc)
        out[res["arm"]] = write_results_csv(df, f"trace_{res['arm']}.csv")
    return out


# --------------------------------------------------------------------------- #
# Deliverable 3 -- accrual.csv (data-coverage disclosure)                      #
# --------------------------------------------------------------------------- #
def build_accrual(cache):
    """Share of the base-horizon DISCOUNTED total accruing strictly AFTER cycle 6
    (mirct's `share_after_k` idiom: `stream[k+1:].sum() / stream.sum()`).

    The plan's data-coverage disclosure: the pivotal SSI hazard is confined to the first
    90 days and needs no extrapolation, but Edmiston's payment windows push COST accrual
    out to 24 months, so the reader is told how much of each total lands beyond the
    6-month mark.

    TWO SHARES ARE WRITTEN AND THEY ARE NOT INTERCHANGEABLE:

    * `share_cost_after_6mo` / `share_qaly_after_6mo` -- PER ARM, straight from the
      engine, of that arm's own discounted total.
    * `share_inc_cost_after_6mo` / `share_inc_qaly_after_6mo` -- of the INCREMENTAL
      (B minus A) stream. **This is the one plan Decision 7 asks for.** It is a property
      of the (stratum x horizon) CELL, not of an arm, so it is repeated identically on
      both arm rows -- the same convention as the incremental block in
      `basecase_results.csv`. NEVER sum an incremental column across arms.

    The incremental share cannot be recovered downstream: the engine exposes only the
    per-arm shares, and BUILD-CONTRACT section 9 forbids figure/table scripts from
    re-running the model. It is computed here from the two arms' `cost_stream` /
    `qaly_stream`, which are already in hand.
    """
    rows = []
    for sname, p in STRATA.items():
        res_a, res_b = cache[(sname, BASE_HORIZON)]
        share_inc_cost = _inc_share_after_6mo(res_b["cost_stream"], res_a["cost_stream"])
        share_inc_qaly = _inc_share_after_6mo(res_b["qaly_stream"], res_a["qaly_stream"])
        for res in (res_a, res_b):
            rows.append({
                "stratum": sname,
                "p_base": p,
                "horizon_cycles": BASE_HORIZON,
                "arm": res["arm"],
                "arm_label": I.ARM_LABELS[res["arm"]],
                "cost_disc_total": res["cost"],
                "qaly_disc_total": res["qaly"],
                "share_cost_after_6mo": res["share_cost_after_6mo"],
                "share_qaly_after_6mo": res["share_qaly_after_6mo"],
                # ---- incremental (B_cinpwt MINUS A_standard) -- plan Decision 7 ----
                # Cell-level, repeated on both arm rows. NaN = the incremental total was
                # numerically zero; see `_inc_share_after_6mo`.
                "share_inc_cost_after_6mo": share_inc_cost,
                "share_inc_qaly_after_6mo": share_inc_qaly,
            })
    df = pd.DataFrame(rows)
    write_results_csv(df, "accrual.csv")
    return df


# --------------------------------------------------------------------------- #
# Deliverable 4 -- cum_events.csv                                              #
# --------------------------------------------------------------------------- #
def build_cum_events(cache):
    """Cumulative expected event counts per 1,000 patients at the base horizon, per arm
    per stratum. Reoperations and persistent infections are COUNTERS (plan Decision 3):
    they carry no incremental cost anywhere in the engine and exist to translate the
    result into clinical units and to feed `validation.py`."""
    rows = []
    for sname, p in STRATA.items():
        res_a, res_b = cache[(sname, BASE_HORIZON)]
        for res in (res_a, res_b):
            rows.append({
                "stratum": sname,
                "p_base": p,
                "horizon_cycles": BASE_HORIZON,
                "arm": res["arm"],
                "arm_label": I.ARM_LABELS[res["arm"]],
                "ssi_per_1000": 1000.0 * res["n_ssi"],
                "ssi_superficial_per_1000": 1000.0 * res["n_ssi_superficial"],
                "ssi_deep_per_1000": 1000.0 * res["n_ssi_deep"],
                "reoperations_per_1000": 1000.0 * res["n_reop"],
                "persistent_infections_per_1000": 1000.0 * res["n_persistent"],
            })
    df = pd.DataFrame(rows)
    write_results_csv(df, "cum_events.csv")
    return df


# --------------------------------------------------------------------------- #
# Deliverable 5 -- basecase_summary.md                                         #
# --------------------------------------------------------------------------- #
def build_summary_md(df, accrual_df, events_df, cache):
    per_case = model.ssi_cost_per_case()
    dq = (float(I.tp.frac_deep) * float(I.util.du_deep) * (I.DEEP_LEN / 12.0)
          + float(I.tp.frac_sup) * float(I.util.du_sup) * (I.SUP_LEN / 12.0))

    L = []
    L.append("# Base-case results -- ciNPWT after instrumented posterior lumbar fusion")
    L.append("")
    L.append("> **" + I.RESULTS_BANNER + "**")
    L.append("")
    L.append("*Generated by `run_basecase.py` from the read-only engine "
             "(`inputs.py` / `model.py`). Model-based result; not a substitute for "
             "clinical evidence.*")
    L.append("")

    # ---- model summary ----
    L.append("## Model at a glance")
    L.append("")
    L.append(f"- **Arms.** `A_standard` = {I.ARM_LABELS['A_standard']}; "
             f"`B_cinpwt` = {I.ARM_LABELS['B_cinpwt']}. Universal *S. aureus* "
             "decolonization is background in **both** arms and cancels from every "
             "base-case increment.")
    L.append(f"- **Structure.** 5 clinical states expanded to an {I.N_SLOTS}-slot engine "
             f"vector (superficial tunnel {I.SUP_LEN} cycles, deep tunnel "
             f"{I.DEEP_LEN} cycles); monthly cycles; death absorbing.")
    L.append(f"- **Cohort.** Age {float(I.mortality.age_start):.0f}, "
             f"{float(I.mortality.frac_male):.0%} male (both sourced to Urquhart 2021 [21] at A2); background mortality from the 2023 US life table; "
             f"perioperative mortality {float(I.mortality.p_periop_death):.1%} in "
             "cycle 0 only, identically in both arms.")
    L.append(f"- **Time and money.** Base horizon {BASE_HORIZON} cycles (2 years); "
             f"scenarios {', '.join(str(h) for h in HORIZONS if h != BASE_HORIZON)} "
             f"cycles. {I.SETTINGS['disc_cost']:.0%}/yr discounting on costs and QALYs; "
             "trapezoidal half-cycle correction on **dwell** streams only (tolls are "
             f"point events, attributed whole). {I.PRICE_YEAR} USD; "
             f"WTP ${WTP:,.0f}/QALY primary (${WTP_LO:,.0f} and ${WTP_HI:,.0f} also "
             "reported).")
    L.append(f"- **Effect.** RR = {float(I.tp.rr_cinpwt):.2f} "
             f"({float(I.tp.rr_cinpwt.low):.2f}-{float(I.tp.rr_cinpwt.high):.2f}), applied "
             "to the onset hazard **inside the engine** (`p_ciNPWT = p_control x RR`), so "
             "the arms are correlated by construction and arm ordering holds in every "
             "draw.")
    L.append(f"- **Device.** ciNPWT ${float(I.cost.c_device_cinpwt):,.0f} "
             f"(`derived_assumption`; the single largest evidence gap, and the reason it "
             "is swept as an axis rather than fixed), charged once at cycle 0 in arm B. "
             f"Standard-dressing differential ${float(I.cost.c_dressing_standard):,.0f} "
             "in the base case, charged at cycle 0 in arm A.")
    L.append("")
    L.append("**There is no index-surgery cost and no well-state cost, by design.** The "
             "model is purely incremental: the index fusion, implant, ward stay and "
             "routine follow-up are identical in both arms and cancel exactly. Every "
             "\"cost\" below is therefore *incremental-versus-no-SSI* cost, **not** total "
             "cost of care, and must not be read as one.")
    L.append("")
    L.append(f"Weighted 24-month incremental cost per SSI = "
             f"**${per_case['weighted']:,.0f}** (2025 USD; "
             f"${per_case['deep']:,.0f} deep / ${per_case['superficial']:,.0f} "
             f"superficial at a deep fraction of {per_case['frac_deep']:.3f}). "
             f"Implied QALY loss per SSI = **{dq:.4f}**, i.e. "
             f"${WTP * dq:,.0f} at ${WTP:,.0f}/QALY. Those two numbers are the whole "
             "threshold arithmetic.")
    L.append("")

    # ---- the toll table: the thing most likely to be mistaken for an error ----
    L.append("## Which SSI cost tranches are charged at each horizon")
    L.append("")
    L.append("SSI cost enters as **three marginal Edmiston tranches** charged relative to "
             "the cycle of onset (`0`, `+6`, `+12` cycles). Onset is confined to the "
             f"first 90 days, so entrants occupy `SUP_0`/`DEEP_0` at cycles "
             f"1-{model.ONSET_CYCLES}; the three tranches therefore land at cycles "
             "1-3, 7-9 and 13-15. **A toll falling beyond the horizon is not charged.** "
             "This is by design (BUILD-CONTRACT section 6.2), not a truncation error:")
    L.append("")
    L.append("| Horizon | Cycles | Tranche 1 (onset, cycles 1-3) | Tranche 2 (onset+6, cycles 7-9) "
             "| Tranche 3 (onset+12, cycles 13-15) | Cost per SSI charged (2025 USD) |")
    L.append("|---|---:|---|---|---|---:|")
    for h in HORIZONS:
        tc = model.tranches_charged(h)
        cells = []
        for d in tc:
            cells.append({"full": "**charged**", "partial": "*partially charged*",
                          "none": "not charged"}[d["charged"]])
        # `horizon_cycles=`, not a hand-rolled `charged == "full"` filter: the full-only
        # convention is valid only where no tranche is clipped, and this call RAISES on a
        # clipped horizon instead of quietly under-reporting the cost per SSI. All four
        # contract horizons are clean, so the printed values are unchanged.
        amt = model.ssi_cost_per_case(horizon_cycles=h)["weighted"]
        L.append(f"| {_horizon_label(h)} | {h} | {cells[0]} | {cells[1]} | {cells[2]} "
                 f"| ${amt:,.0f} |")
    L.append("")
    # Tranche 2 is `c_ssi_*_6_12mo`. Interpolated from inputs.py (BUILD-CONTRACT section
    # 9: only inputs.py may own a number) so this sentence cannot go stale if Phase A2
    # revises Edmiston. Both tranches are stored in the same price year -- asserted, so a
    # future divergence fails loudly instead of mislabelling one of them.
    t2_deep, t2_sup = I.cost.c_ssi_deep_6_12mo, I.cost.c_ssi_sup_6_12mo
    assert int(t2_deep.price_year) == int(t2_sup.price_year), \
        "tranche-2 cost parameters are stored in different price years"
    L.append("**The 1-year row is the trap.** Tranche 2 lands at cycles 7-9, comfortably "
             "*inside* a 12-cycle horizon, and is charged. Dropping it would silently "
             f"remove ${float(t2_deep):,.0f} (deep) / ${float(t2_sup):,.0f} "
             f"(superficial) in {int(t2_deep.price_year)} USD from every 1-year "
             "scenario, and the loss would look like a plausible horizon effect rather "
             "than a bug. Only the **90-day** horizon loses tranches 2 and 3 -- which is "
             "why the 90-day incremental cost is small and ciNPWT looks least attractive "
             "there. Note also that tranche 3 is charged whole at onset+12 but represents "
             "spending through months **25-27** post-index, so cost accrual reaches "
             "slightly past the nominal 2-year horizon; this is by design and is "
             "disclosed rather than truncated.")
    L.append("")

    # ---- headline: base horizon, per stratum ----
    base = df[(df["horizon_cycles"] == BASE_HORIZON)]
    L.append(f"## Base case -- {_horizon_label(BASE_HORIZON)}, {BASE_HORIZON} cycles")
    L.append("")
    L.append("| Risk stratum | Baseline 90-d SSI risk | Arm | Cost | QALYs | Life-years "
             "| Expected SSIs |")
    L.append("|---|---:|---|---:|---:|---:|---:|")
    for sname, p in STRATA.items():
        for arm in I.ARMS:
            r = base[(base["stratum"] == sname) & (base["arm"] == arm)].iloc[0]
            L.append(f"| {sname} | {p:.1%} | `{arm}` | {_fmt_money(r['cost'])} "
                     f"| {r['qaly']:.4f} | {r['life_years']:.3f} | {r['n_ssi']:.4f} |")
    L.append("")
    L.append("| Risk stratum | Incremental cost | Incremental QALY | ICER | "
             f"INMB @ ${WTP:,.0f} | CE quadrant | Dominance | NNT | SSIs prevented / 1,000 "
             "| Reoperations prevented / 1,000 |")
    L.append("|---|---:|---:|---|---:|---|---|---:|---:|---:|")
    for sname in STRATA:
        r = base[(base["stratum"] == sname) & (base["arm"] == "B_cinpwt")].iloc[0]
        L.append(f"| {sname} | {_fmt_money(r['inc_cost'])} | {r['inc_qaly']:+.5f} "
                 f"| {_fmt_icer(r['icer'])} | {_fmt_money(r['inmb_at_100k'])} "
                 f"| {r['quadrant']} | {r['dominance']} | {r['nnt']:.1f} "
                 f"| {r['ssi_prevented_per_1000']:.1f} "
                 f"| {r['reop_prevented_per_1000']:.1f} |")
    L.append("")
    L.append("Incrementals are **`B_cinpwt` minus `A_standard`**. A negative incremental "
             "cost means ciNPWT *saves* money; with a positive incremental QALY that is "
             "**dominance** (CE quadrant `NW`), and the ICER is reported as `n/a` because "
             "a ratio is uninformative under strict dominance.")
    L.append("")

    # ---- per-stratum narrative ----
    L.append("## Reading the strata")
    L.append("")
    for sname, p in STRATA.items():
        r = base[(base["stratum"] == sname) & (base["arm"] == "B_cinpwt")].iloc[0]
        ra = base[(base["stratum"] == sname) & (base["arm"] == "A_standard")].iloc[0]
        anchor = {a[0]: f"{a[1]} {a[2]}" for a in I.SETTINGS["risk_anchors"]}.get(p, "")
        saving = -r["inc_cost"]
        L.append(f"**{sname.capitalize()} risk (p = {p:.1%}"
                 + (f", the {anchor} anchor" if anchor else "") + ").** "
                 f"Standard care accrues {_fmt_money(ra['cost'])} of SSI-attributable "
                 f"cost per patient over two years and "
                 f"{ra['n_ssi'] * 1000:.1f} SSIs per "
                 f"1,000. ciNPWT prevents {r['ssi_prevented_per_1000']:.1f} SSIs per "
                 f"1,000 (NNT {r['nnt']:.0f}) and "
                 f"{r['reop_prevented_per_1000']:.1f} reoperations per 1,000, gains "
                 f"{r['inc_qaly']:+.5f} QALYs, and "
                 + (f"**saves {_fmt_money(saving)}** per patient" if saving > 0
                    else f"costs {_fmt_money(-saving)} more per patient")
                 + f" -- INMB {_fmt_money(r['inmb_at_100k'])} at ${WTP:,.0f}/QALY "
                 f"(**{r['dominance']}**).")
        L.append("")

    # ---- horizon panel ----
    L.append("## Horizon scenarios")
    L.append("")
    L.append("Same cohort, same parameters; only the horizon changes. The step from "
             "90 days to 1 year to 2 years is dominated by **which SSI cost tranches "
             "fall inside the horizon** (table above), not by any change in the "
             "underlying hazard -- the pivotal onset hazard is zero after 90 days in "
             "every column.")
    L.append("")
    L.append("| Horizon | Stratum | Incremental cost | Incremental QALY | "
             f"INMB @ ${WTP:,.0f} | Dominance |")
    L.append("|---|---|---:|---:|---:|---|")
    for h in HORIZONS:
        for sname in STRATA:
            r = df[(df["horizon_cycles"] == h) & (df["stratum"] == sname)
                   & (df["arm"] == "B_cinpwt")].iloc[0]
            L.append(f"| {_horizon_label(h)} | {sname} | {_fmt_money(r['inc_cost'])} "
                     f"| {r['inc_qaly']:+.5f} | {_fmt_money(r['inmb_at_100k'])} "
                     f"| {r['dominance']} |")
    L.append("")

    # ---- accrual ----
    L.append("## Accrual / data-coverage disclosure")
    L.append("")
    L.append("Share of the base-horizon **discounted** total accruing strictly after "
             "cycle 6 (6 months). The pivotal SSI hazard needs **no extrapolation** -- it "
             "is zero from day 91 -- but Edmiston's payment windows carry cost out to 24 "
             "months, so the post-6-month cost share is a real disclosure while the "
             "post-6-month QALY share simply reflects continued survival of the cohort.")
    L.append("")
    L.append("**Two different shares are reported below and they are not "
             "interchangeable.** The plan's Decision 7 asks for the share of "
             "***incremental*** cost and QALY accruing after 6 months -- that is "
             "**Table B**. Table A is the per-arm share of each arm's own total, which "
             "is descriptive only. The two differ substantially on the cost side: the "
             "ciNPWT device toll falls whole at cycle 0 and cancels much of the *early* "
             "SSI saving, so a larger fraction of the **net** increment accrues after 6 "
             "months than of either arm's own total -- most visibly at low risk, where "
             "the early saving is smallest. Reading the per-arm share as if it answered "
             "Decision 7 would understate how much of the decision-relevant cost lands "
             "late.")
    L.append("")
    L.append("**Table A -- per arm** (share of that arm's own discounted total):")
    L.append("")
    L.append("| Stratum | Arm | Discounted cost | Discounted QALYs | "
             "Cost after 6 mo (per arm) | QALYs after 6 mo (per arm) |")
    L.append("|---|---|---:|---:|---:|---:|")
    for _, r in accrual_df.iterrows():
        L.append(f"| {r['stratum']} | `{r['arm']}` | {_fmt_money(r['cost_disc_total'])} "
                 f"| {r['qaly_disc_total']:.4f} | {r['share_cost_after_6mo']:.1%} "
                 f"| {r['share_qaly_after_6mo']:.1%} |")
    L.append("")
    L.append("**Table B -- incremental (`B_cinpwt` minus `A_standard`) -- the plan's "
             "Decision 7 disclosure:**")
    L.append("")
    L.append("| Stratum | Incremental cost | Incremental QALY | "
             "Incremental cost after 6 mo | Incremental QALYs after 6 mo |")
    L.append("|---|---:|---:|---:|---:|")
    for sname in STRATA:
        acc = accrual_df[(accrual_df["stratum"] == sname)
                         & (accrual_df["arm"] == "B_cinpwt")].iloc[0]
        inc = base[(base["stratum"] == sname) & (base["arm"] == "B_cinpwt")].iloc[0]
        L.append(f"| {sname} | {_fmt_money(inc['inc_cost'])} | {inc['inc_qaly']:+.5f} "
                 f"| {_fmt_share(acc['share_inc_cost_after_6mo'])} "
                 f"| {_fmt_share(acc['share_inc_qaly_after_6mo'])} |")
    L.append("")
    L.append("Both tables come from `accrual.csv`, which carries all four columns "
             "(`share_cost_after_6mo`, `share_qaly_after_6mo`, "
             "`share_inc_cost_after_6mo`, `share_inc_qaly_after_6mo`). The incremental "
             "shares are a property of the **stratum cell**, not of an arm, so "
             "`accrual.csv` repeats them identically on both arm rows -- filter to one "
             "arm before reading them and never sum them across arms. Where the "
             "incremental total is numerically zero the share is written as `NaN` and "
             "printed as `n/a` here: a share of a vanishing denominator is meaningless, "
             "so it is declared rather than reported as a large number.")
    L.append("")

    # ---- events ----
    L.append("## Cumulative expected events per 1,000 patients "
             f"({_horizon_label(BASE_HORIZON)})")
    L.append("")
    L.append("Reoperations and persistent infections are **counters** (plan Decision 3): "
             "Edmiston's incremental payment is an all-cause spend that already contains "
             "readmission, I&D, revision and antibiotics, so these events carry **no** "
             "incremental cost anywhere in the engine. They exist to translate the result "
             "into clinical units and to feed the validation checks.")
    L.append("")
    L.append("| Stratum | Arm | SSIs | Superficial | Deep | Reoperations | "
             "Persistent infections |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for _, r in events_df.iterrows():
        L.append(f"| {r['stratum']} | `{r['arm']}` | {r['ssi_per_1000']:.1f} "
                 f"| {r['ssi_superficial_per_1000']:.1f} | {r['ssi_deep_per_1000']:.1f} "
                 f"| {r['reoperations_per_1000']:.1f} "
                 f"| {r['persistent_infections_per_1000']:.1f} |")
    L.append("")

    # ---- caveats ----
    L.append("## What these numbers do and do not carry")
    L.append("")
    L.append("- **Verified.** Phase A2 full-text verification completed 2026-08-17: 46 of 48 parameters are closed out against the 14 source full texts and 43 carry a table/page locator. Only 2 still carry `verify=True` -- the ciNPWT institutional net acquisition price and the superficial tunnel length -- because no available source resolves either.")
    L.append("- **Cost is incremental-versus-no-SSI**, not total cost of care. Do not "
             "compare it with a published cost-per-fusion figure.")
    L.append(f"- **The absolute well-state utility ({float(I.util.u_well):.2f}) is an "
             "explicitly labelled assumption.** It is identical in both arms and enters "
             "the incremental QALY only through the small fraction of cohort-time that "
             "differs, so it scales each arm's QALY *level* but very nearly cancels from "
             "the *difference*.")
    L.append(f"- **Conservative-bias register** (all five bias against ciNPWT): no "
             f"SSI-attributable mortality difference is assumed; the weaker all-surgery "
             f"RR {float(I.tp.rr_cinpwt):.2f} is used rather than the favourable "
             f"spine-specific RR {float(I.tp.rr_cinpwt_spine):.3f}; "
             f"superficial-to-deep progression is omitted; the standard-dressing "
             f"differential is set to $0 so arm B pays the full device cost with no "
             f"offset; and the base case charges Pennington's matched attributable cost "
             f"(m_ssi_cost = {float(I.cost.m_ssi_cost):.3f}) rather than Edmiston's "
             f"larger all-cause increment. One anti-conservative item is disclosed: "
             f"`du_sup` is McFarland's *any-SSI* decrement applied to superficial SSI, "
             f"and A2 upheld that objection rather than retiring it -- the only "
             f"superficial-specific estimate in McFarland is a single null study at "
             f"-0.03.")
    L.append("- **Competing mortality.** The cumulative 90-day onset *risk* reproduces "
             "the axis value `p` exactly, but the expected *count* of SSIs in the cohort "
             "sits ~0.3% below it because perioperative (cycle 0) and background "
             "mortality remove people from the at-risk pool. That is correct "
             "competing-risks behaviour, not a rounding error.")
    L.append("")
    L.append("---")
    L.append("")
    L.append("*Files written: `basecase_results.csv`, `basecase_summary.md`, "
             "`trace_A_standard.csv`, `trace_B_cinpwt.csv`, `accrual.csv`, "
             "`cum_events.csv` -- all in `v2/results/`, each CSV carrying the results "
             "banner as line 1 (read with `pd.read_csv(path, skiprows=1)`).*")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "basecase_summary.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    return path


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df, cache = build_basecase_results()
    build_traces(cache)
    accrual_df = build_accrual(cache)
    events_df = build_cum_events(cache)
    build_summary_md(df, accrual_df, events_df, cache)

    # ---- stdout headline ----
    print("=" * 78)
    print("SSI / ciNPWT risk-threshold Markov CUA -- BASE CASE")
    print(I.RESULTS_BANNER)
    print("=" * 78)
    per_case = model.ssi_cost_per_case()
    print(f"Cohort age {float(I.mortality.age_start):.0f}, "
          f"{float(I.mortality.frac_male):.0%} male | monthly cycles | base horizon "
          f"{BASE_HORIZON} | disc {I.SETTINGS['disc_cost']:.0%} | WTP ${WTP:,.0f}/QALY | "
          f"{I.PRICE_YEAR} USD")
    print(f"RR {float(I.tp.rr_cinpwt):.2f} | device ${float(I.cost.c_device_cinpwt):,.0f} | "
          f"weighted 24-mo cost per SSI ${per_case['weighted']:,.0f}")
    print()

    base = df[df["horizon_cycles"] == BASE_HORIZON]
    print(f"Base case ({BASE_HORIZON} cycles), B_cinpwt minus A_standard:")
    print(f"  {'stratum':<14}{'p':>7}{'cost A':>11}{'cost B':>11}{'inc cost':>11}"
          f"{'inc QALY':>10}{'INMB':>11}{'NNT':>7}  dominance")
    for sname in STRATA:
        rb = base[(base["stratum"] == sname) & (base["arm"] == "B_cinpwt")].iloc[0]
        ra = base[(base["stratum"] == sname) & (base["arm"] == "A_standard")].iloc[0]
        print(f"  {sname:<14}{rb['p_base']:>7.3f}{ra['cost']:>11,.0f}{rb['cost']:>11,.0f}"
              f"{rb['inc_cost']:>11,.0f}{rb['inc_qaly']:>10.5f}{rb['inmb_at_100k']:>11,.0f}"
              f"{rb['nnt']:>7.1f}  {rb['dominance']}")
    print()
    print("Horizon scenarios (INMB @ $%s/QALY, B minus A):" % f"{WTP:,.0f}")
    print(f"  {'horizon':<18}" + "".join(f"{s:>14}" for s in STRATA))
    for h in HORIZONS:
        cells = []
        for sname in STRATA:
            r = df[(df["horizon_cycles"] == h) & (df["stratum"] == sname)
                   & (df["arm"] == "B_cinpwt")].iloc[0]
            cells.append(f"{r['inmb_at_100k']:>14,.0f}")
        print(f"  {_horizon_label(h):<18}" + "".join(cells))
    print()
    tc = model.tranches_charged(12)
    print("Toll check (BUILD-CONTRACT 6.2): at a 12-cycle horizon tranche 2 (cycles "
          f"{tc[1]['first_cycle']}-{tc[1]['last_cycle']}) is {tc[1]['charged'].upper()}, "
          f"tranche 3 (cycles {tc[2]['first_cycle']}-{tc[2]['last_cycle']}) is "
          f"{tc[2]['charged'].upper()}.")
    print()
    print("Files written to:", RESULTS_DIR)
    expected = ["basecase_results.csv", "basecase_summary.md", "trace_A_standard.csv",
                "trace_B_cinpwt.csv", "accrual.csv", "cum_events.csv"]
    missing = []
    for fn in expected:
        path = os.path.join(RESULTS_DIR, fn)
        ok = os.path.exists(path)
        if not ok:
            missing.append(fn)
        print(f"  [{'OK' if ok else 'MISSING'}] {fn}")
    if missing:
        raise RuntimeError(f"run_basecase.py failed to write: {missing}")


if __name__ == "__main__":
    main()
