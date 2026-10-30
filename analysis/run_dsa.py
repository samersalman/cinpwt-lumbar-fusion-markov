"""
run_dsa.py -- One-way deterministic sensitivity analysis (tornado) for the SSI / ciNPWT
risk-threshold Markov cost-utility analysis.

Imports the engine (`inputs.py` / `model.py`) READ-ONLY and writes the three
BUILD-CONTRACT section 8 deliverables to `v2/results/`:

    dsa_results.csv    one row per varied parameter PER STRATUM -- base/low/high value,
                       INMB at each bound, |INMB swing|, incremental cost and QALY at each
                       bound, dominance at each bound, the constant `stratum` / `p_base`
                       columns, and a `note` saying how the row was varied
    dsa_tornado.csv    the plot-ready subset, swing-sorted descending WITHIN each stratum;
                       carries the same `stratum` / `p_base` columns -- filter
                       `stratum == "intermediate"` for the Fig 5 panel
    dsa_summary.md     results banner, the ranked table at the Fig 5 stratum, the
                       three-stratum ranking, the crossing point, and the narrative

Every CSV carries `inputs.RESULTS_BANNER` as a '#'-prefixed LINE 1 followed by the header
row, so downstream readers use `pd.read_csv(path, skiprows=1)` (BUILD-CONTRACT section 7).

Run with cwd = v2/analysis/ :  python3 run_dsa.py

=============================================================================
WHERE THE ANALYSIS SITS
=============================================================================
Base 24-cycle horizon, 3%/yr discounting, WTP $100,000/QALY, 2025 USD. The tornado is
sorted by **|INMB swing|**, descending -- NOT by ICER: ciNPWT is strictly dominant over
most of the range, so `icer` is `np.nan` on most rows and is reported only as a diagnostic
(BUILD-CONTRACT section 5; `run_log.d/model.md` gotcha 5).

**The full tornado is run and emitted at ALL THREE reported strata** -- low (2.0%),
intermediate (6.6%, the Edmiston [12] anchor) and high (11.4%) -- because a one-way DSA is
a *local* analysis and BUILD-CONTRACT section 10, as amended 2026-08-09, is explicitly
stratum-conditional: `m_ssi_cost` must top the tornado "at the intermediate (6.6%) and
high (11.4%) strata", and at the low stratum the top bar is `c_device_cinpwt` ($500)
rather than `m_ssi_cost` ($489). That is a result, not a defect: every SSI-cost bar scales
with `p` while the device bar is a fixed $500 swing, so they cross at p = 1.533%.
The crossing is computed live here and reported in `dsa_summary.md`.

**The intermediate stratum remains the Fig 5 panel** (plan Phase C); low and high are
supplementary. Both CSVs therefore carry constant `stratum` and `p_base` columns -- same
idiom as the constant `inmb_base` column -- so a supplementary tornado can never be
byte-indistinguishable from the shipped one. Downstream figure code MUST filter
`stratum == "intermediate"` rather than reading the first rows.

=============================================================================
THE FIVE RULES THIS MODULE EXISTS TO GET RIGHT
=============================================================================
1. **SSI-attributable cost is ONE bar, driven by `m_ssi_cost` (0.266-1.201).**
   BUILD-CONTRACT section 10 requires SSI-attributable cost to be the TOP tornado bar.
   `m_ssi_cost` is a dimensionless multiplier on all six cost tranches at once and exists
   precisely so the pivotal parameter appears as one interpretable bar. The six tranches
   are therefore NOT additionally swung as separate bars: that would double-count the same
   uncertainty across six small bars and could displace the pivotal one. The requirement is
   asserted at the end of `main()` -- this module refuses to ship a tornado that violates it.

2. **`p_base_ssi` is the study's x-axis, not an uncertain input.** It is excluded from the
   tornado entirely. Its 0.005-0.15 span produces a larger INMB swing than any real
   parameter, so including it would displace the pivotal bar and answer a question the
   paper asks elsewhere (`run_threshold.py`, Figs 2-4). The risk-stratum spread IS
   reported in `dsa_summary.md`, in its own section, explicitly labelled as the analysis
   axis and deliberately kept OUT of both CSVs so no figure can plot it as a bar.

3. **`n_tunnel_sup` / `n_tunnel_deep` are swept through the `n_sup=` / `n_deep=` KWARGS**,
   never through `sample`. Tunnel width is a LAYOUT, not a sampled Param: the engine reads
   the layout from the kwargs and RAISES if a `sample` dict carries a tunnel Param that
   disagrees with it (model.py post-review fix 5, added because this module was about to
   walk into it). Sampling them would have produced two silent zero-swing rows that read as
   "duration does not matter" when in fact nothing was ever varied.

4. **`frac_deep` / `frac_sup` are a Dirichlet PAIR** (`group="ssi_split"`). They are varied
   ONCE, driven by `frac_deep`, with the partner set to `1 - member` so the split always
   sums to 1. Moving one alone would hand the engine an unnormalized split.

5. **The relative effect enters as one CORRELATED "gap" row.** `rr_cinpwt` is applied to the
   onset hazard INSIDE the engine (`p_ciNPWT = p_control x RR`), so varying it moves arm B
   relative to arm A while arm A stays put -- which is exactly what the TLIF "gap" idiom
   constructs by hand when an engine reads two independent arm probabilities. No second
   arm-B parameter exists to vary, and no arm ordering can invert.

Two further traps, both handled and both visible in the output:

* **Counters carry no cost (plan Decision 3).** `p_persistent_deep`, `p_reop_ssi`,
  `n_debridements` and `rr_readmit_sa` produce an INMB swing of EXACTLY zero. That is the
  correct result, not a bug -- Edmiston's payment is an all-cause incremental spend that
  already contains reoperation, readmission, I&D and antibiotics. They are kept as explicit
  zero rows with a note, rather than dropped silently, so a reader can see Decision 3 is
  real and enforced. `main()` asserts the zeros.
* **Costs are passed in their OWN price year.** A `sample` value for a cost is expressed in
  that parameter's stored price year and is inflated exactly once inside the engine
  (`model._cost` -> `inputs.cost_2025`). This module never calls `val()` on a cost and never
  calls `inflate()`. The only two cost parameters it varies (`c_device_cinpwt`,
  `c_dressing_standard`) are already 2025 USD and must not be inflated at all.

See: v2/BUILD-CONTRACT.md sections 5 / 8 / 9 / 10, v2/run_log.d/model.md
("FOR THE NEXT TASK"), v2/run_log.d/inputs.md (objections 1-3),
v2/run_log.d/run_dsa.md (this module's own run-log fragment).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import inputs as I
import model

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V1_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
RESULTS_DIR = os.path.join(V1_DIR, "results")

# --------------------------------------------------------------------------- #
# Where the DSA is run. Every one of these comes from inputs.py -- BUILD-CONTRACT
# section 9 forbids a run script from owning a number.
# --------------------------------------------------------------------------- #
STRATA = I.SETTINGS["strata"]
DSA_STRATUM = "intermediate"                        # plan Phase C: the Fig 5 panel
P_DSA = float(STRATA[DSA_STRATUM])                  # 0.066, the Edmiston [12] anchor
HORIZON = int(I.SETTINGS["horizon_cycles"])         # 24 cycles = 2 years
WTP = float(I.SETTINGS["wtp_primary"])              # $100,000/QALY

#: Every stratum the tornado is emitted at, **PRIMARY FIRST**. The Fig 5 stratum leads both
#: CSVs so that a naive reader who does not filter still gets the specified panel; the
#: `stratum` / `p_base` columns exist so that filtering is possible at all.
DSA_STRATA: list[tuple[str, float]] = (
    [(DSA_STRATUM, P_DSA)]
    + [(n, float(p)) for n, p in STRATA.items() if n != DSA_STRATUM]
)

#: BUILD-CONTRACT section 10 as amended 2026-08-09: `m_ssi_cost` must be the top bar at the
#: intermediate and high strata, "which is where the DSA is specified to run". The low
#: stratum is deliberately NOT in this tuple -- the device price leads there, and the
#: contract says so explicitly ("Do not 'fix' a low-stratum tornado to force `m_ssi_cost`
#: to the top"). The requirement is asserted for these two and REPORTED for the third.
SECTION10_STRATA: tuple[str, ...] = ("intermediate", "high")

#: The two bars whose ranking crosses with `p` (section 10's amendment). Used by
#: `crossing_point()`; named here rather than inline so the crossing can never be computed
#: on a different pair than the one the summary describes.
CROSSING_PAIR = ("m_ssi_cost", "c_device_cinpwt")

#: Sanity: a 2025-USD parameter must never be inflated (inputs.py trap 1).
assert I.cost_2025(I.cost.c_device_cinpwt) == float(I.cost.c_device_cinpwt.value), \
    "c_device_cinpwt is already 2025 USD and must not be inflated"


# --------------------------------------------------------------------------- #
# 1. The variation registry                                                    #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Variation:
    """One tornado row.

    `param` is the DRIVER parameter's `Param.name` and is the first column of
    `dsa_results.csv`; its `value` / `low` / `high` come straight from `inputs.py`.
    `handling` says how the bound reaches the engine, and it is the whole point of this
    registry -- three different routes exist and picking the wrong one fails SILENTLY:

        "sample"  -> {param: bound}                     the ordinary case
        "pair"    -> {param: bound, partner: 1 - bound} Dirichlet pair (ssi_split)
        "kwarg"   -> run_pair(..., n_sup=/n_deep=)      LAYOUT, not a sampled Param
    """

    param: str
    label: str            # human-readable; first column of dsa_tornado.csv (Fig 5 y-axis)
    group: str
    handling: str         # "sample" | "pair" | "kwarg"
    note: str
    partner: str = ""     # handling == "pair"
    kwarg: str = ""       # handling == "kwarg"


#: EXPLICIT, not a filter over `dist != "fixed"`.
#: A dist-based filter would be wrong in both directions here: it would DROP `m_ssi_cost`
#: (dist='fixed', and the bar BUILD-CONTRACT section 10 requires to be on top),
#: `c_dressing_standard`, `du_recovered_residual`, both tunnel widths, `age_start` and
#: `frac_male`; and it would ADD a dozen scenario/secondary parameters the engine never
#: reads, each of which would report a swing of exactly zero that means "not in the model"
#: rather than "does not matter". Every exclusion is recorded in EXCLUSIONS below and every
#: one of the 48 parameters is accounted for (asserted in `_check_accounting`).
VARIATIONS: list[Variation] = [
    # ---- the pivotal bar (BUILD-CONTRACT section 10) --------------------------------- #
    Variation(
        "m_ssi_cost",
        "SSI-attributable cost multiplier (m_ssi_cost) x all six tranches",
        "cost", "sample",
        "PIVOTAL. Dimensionless multiplier scaling ALL SIX Edmiston tranches together, so "
        "SSI-attributable cost is ONE bar; the six tranches are deliberately NOT swung "
        "separately (that would double-count the same uncertainty). Range spans Pennington's "
        "matched increment (0.266) to the deep-SSI value applied to every SSI (1.201).",
    ),
    # ---- correlated relative-effect 'gap' row ---------------------------------------- #
    Variation(
        "rr_cinpwt",
        "ciNPWT relative risk (rr_cinpwt) -- correlated arm gap",
        "effect", "sample",
        "Correlated 'gap' row: RR is applied to the onset hazard INSIDE the engine "
        "(p_ciNPWT = p_control x RR), so this single row moves arm B relative to a fixed "
        "arm A and arm ordering can never invert. Lower RR = larger effect = higher INMB, "
        "so inmb_low > inmb_high on this row.",
    ),
    # ---- Dirichlet pair --------------------------------------------------------------- #
    Variation(
        "frac_deep",
        "Deep fraction of SSI (frac_deep; frac_sup = 1 - frac_deep)",
        "ssi_split", "pair",
        "Varied as a Dirichlet PAIR (group ssi_split): the partner frac_sup is set to "
        "1 - frac_deep at each bound so the split always sums to 1. Total incidence is "
        "invariant by construction (the engine sets the total hazard first), so this row "
        "moves SSI COMPOSITION only -- cost per SSI and QALY loss per SSI.",
        partner="frac_sup",
    ),
    # ---- the two cost tolls that are genuinely their own parameters ------------------- #
    Variation(
        "c_device_cinpwt",
        "ciNPWT device unit cost (c_device_cinpwt), 2025 USD",
        "cost", "sample",
        "2025 USD -- already in the model price year and NOT inflated. One-time cycle-0 "
        "toll, arm B only, so INMB moves dollar-for-dollar with it. Also swept in full as "
        "analysis axis 2 by run_threshold.py; this bar is the one-way version.",
    ),
    Variation(
        "c_dressing_standard",
        "Standard-dressing differential (c_dressing_standard), 2025 USD",
        "cost", "sample",
        "2025 USD -- not inflated. Charged as a cycle-0 toll in ARM A, so it moves the "
        "result in the OPPOSITE direction to the device price. $0 base case is the "
        "conservative-bias choice (arm B pays the full device cost with no offset).",
    ),
    # ---- utilities -------------------------------------------------------------------- #
    Variation(
        "du_deep", "Deep-SSI utility decrement (du_deep)", "utility", "sample",
        "Positive magnitude subtracted from u_well for the whole deep tunnel.",
    ),
    Variation(
        "du_sup", "Superficial-SSI utility decrement (du_sup)", "utility", "sample",
        "Positive magnitude subtracted from u_well for the whole superficial tunnel.",
    ),
    Variation(
        "du_recovered_residual",
        "Residual decrement after recovery (du_recovered_residual)", "utility", "sample",
        "Applied in RECOV to the horizon. Non-zero only in the residual-decrement "
        "scenario; arm A accumulates more RECOV time (more SSIs), so a residual decrement "
        "favours ciNPWT.",
    ),
    Variation(
        "u_well", "Well-state utility (u_well)", "utility", "sample",
        "Expected to swing by ~0: u_well is identical in both arms and the arms' live mass "
        "is identical cycle by cycle (mortality is arm-independent), so it cancels from the "
        "incremental QALY to machine precision. It scales each arm's QALY LEVEL only. This "
        "row is the numerical proof of the BUILD-CONTRACT section 6.7 claim.",
    ),
    # ---- tunnel widths: LAYOUT, swept through the kwargs ------------------------------ #
    Variation(
        "n_tunnel_deep", "Deep-SSI tunnel duration (n_tunnel_deep), months",
        "structure", "kwarg",
        "Swept via the n_deep= kwarg, NOT via sample -- tunnel width is a layout and the "
        "engine RAISES on a disagreeing sample key. Evidence conflict: Urquhart [21] "
        "reports PROM recovery by 1 year, McFarland [19] a persisting decrement. Moves the "
        "QALY side only (the cost tolls attach to the onset cohort, not tunnel occupancy).",
        kwarg="n_deep",
    ),
    Variation(
        "n_tunnel_sup", "Superficial-SSI tunnel duration (n_tunnel_sup), months",
        "structure", "kwarg",
        "Swept via the n_sup= kwarg, NOT via sample (see n_tunnel_deep). No source reports "
        "a superficial-SSI duration; the 2-6 month range is an assumed span.",
        kwarg="n_sup",
    ),
    # ---- cohort / mortality ----------------------------------------------------------- #
    Variation(
        "p_periop_death", "Perioperative mortality (p_periop_death)", "mortality", "sample",
        "Cycle 0 only, IDENTICAL in both arms, so it enters the increment only through the "
        "competing-risks attenuation of onset -- a very small swing is the correct result.",
    ),
    Variation(
        "age_start", "Cohort age (age_start), years", "mortality", "sample",
        "Indexes the 2023 US life table. now sourced to Urquhart 2021 [21] at A2. Enters "
        "both arms identically; only the competing-risks attenuation differs.",
    ),
    Variation(
        "frac_male", "Cohort fraction male (frac_male)", "mortality", "sample",
        "Sex mix for background mortality. now sourced to Urquhart 2021 [21] at A2. Enters "
        "both arms identically.",
    ),
    # ---- COUNTERS AND CROSS-CHECKS: plan Decision 3, EXACT ZERO expected --------------- #
    Variation(
        "p_reop_ssi", "Reoperation among SSI patients (p_reop_ssi) -- counter",
        "counter", "sample",
        "COUNTER -- no cost by plan Decision 3. Read by the engine (it populates n_reop) "
        "but multiplies nothing, so the INMB swing is EXACTLY zero. That is the correct "
        "result: Edmiston's payment is an all-cause increment that already contains "
        "reoperation, and costing this would double-count.",
    ),
    Variation(
        "p_persistent_deep", "Persistent deep infection (p_persistent_deep) -- counter",
        "counter", "sample",
        "COUNTER -- no cost by plan Decision 3. Read by the engine (populates n_persistent) "
        "but multiplies nothing; INMB swing is EXACTLY zero by design.",
    ),
    Variation(
        "n_debridements", "Debridements per SSI (n_debridements) -- cross-check",
        "counter", "sample",
        "CROSS-CHECK -- never costed (Decision 3) and not read by the engine at all. INMB "
        "swing is EXACTLY zero. Kept as an explicit zero row so the exclusion is visible "
        "rather than silent.",
    ),
    Variation(
        "rr_readmit_sa", "Readmission risk ratio (rr_readmit_sa) -- cross-check",
        "counter", "sample",
        "CROSS-CHECK -- never added to any cost (Decision 3) and not read by the engine. "
        "Readmission is already inside Edmiston's all-cause increment. INMB swing EXACTLY "
        "zero, by design.",
    ),
]

#: The rows whose INMB swing MUST be exactly zero (plan Decision 3). Asserted in `main()`:
#: if one of these ever moves the result, a counter has started carrying cost and the model
#: is double-counting against Edmiston's all-cause increment.
COUNTER_ROWS = tuple(v.param for v in VARIATIONS if v.group == "counter")

#: Every parameter deliberately NOT given a tornado bar, with the reason. Printed in
#: `dsa_summary.md` so the reader can audit the coverage of all 48 parameters, and checked
#: against `inputs.all_params()` at run time so a newly added parameter cannot slip through
#: unexamined.
EXCLUSIONS: dict[str, str] = {
    # -- the analysis axis -- #
    "p_base_ssi":
        "ANALYSIS AXIS, not an uncertain input. Its 0.005-0.15 span would produce the "
        "largest swing in the file and would displace the pivotal bar. Swept in full by "
        "run_threshold.py; the risk-stratum spread is reported separately in this summary.",
    # -- Dirichlet partner -- #
    "frac_sup":
        "Varied as the PARTNER of the frac_deep Dirichlet pair (group ssi_split), set to "
        "1 - frac_deep at each bound. Moving it independently would break the split.",
    # -- the six commercial tranches: subsumed in m_ssi_cost -- #
    **{name:
       "Subsumed in the m_ssi_cost bar, which scales all six tranches together "
       "(BUILD-CONTRACT section 10). Swinging it separately would double-count the same "
       "uncertainty across six small bars and could displace the pivotal bar."
       for name in ("c_ssi_sup_0_6mo", "c_ssi_sup_6_12mo", "c_ssi_sup_12_24mo",
                    "c_ssi_deep_0_6mo", "c_ssi_deep_6_12mo", "c_ssi_deep_12_24mo")},
    # -- Medicare perspective scenario -- #
    **{name:
       "Medicare-perspective scenario input; not read by the base-case engine. Handled as "
       "a perspective scenario in run_scenarios.py, not as a tornado bar."
       for name in ("c_ssi_sup_0_6mo_mcr", "c_ssi_sup_6_12mo_mcr", "c_ssi_sup_12_24mo_mcr",
                    "c_ssi_deep_0_6mo_mcr", "c_ssi_deep_6_12mo_mcr",
                    "c_ssi_deep_12_24mo_mcr")},
    "k_attrib_pennington":
        "Scenario constant. A2 promoted it: m_ssi_cost now TAKES this value in the base "
        "case, and the all-cause increment (1.000) is the sensitivity. Its "
        "uncertainty is already carried by the m_ssi_cost bar -- and since A2 it IS the "
        "m_ssi_cost base case, not a bound on it.",
    # -- alternative / scenario relative effects -- #
    **{name:
       "Alternative or scenario relative effect; the engine reads rr_cinpwt only. Handled "
       "as an effect scenario in run_scenarios.py -- a tornado bar would report a swing of "
       "exactly zero meaning 'not in the model', not 'does not matter'."
       for name in ("rr_cinpwt_spine", "rr_cinpwt_lambrechts", "rr_cinpwt_mueller",
                    "rr_cinpwt_attenuated")},
    # -- decolonization: background in both arms, cancels -- #
    **{name:
       "Decolonization is background in BOTH arms and cancels exactly from every base-case "
       "increment; secondary analysis only, so it carries no tornado bar."
       for name in ("or_decol", "rr_decol_broad", "f_sa_ssi", "m_decol_residual",
                    "c_decol_bundle")},
    # -- onset weights: constrained to sum to 1 -- #
    **{name:
       "Conditional onset weight constrained so the triple sums to 1. Moving one alone "
       "changes TOTAL incidence (cumulative onset risk = p x sum(w)) and would conflate "
       "onset TIMING with onset TOTAL. The structural scenario (uniform "
       "1/3, 1/3, 1/3) belongs to run_scenarios.py."
       for name in ("w_onset_deep_m1", "w_onset_deep_m2", "w_onset_deep_m3",
                    "w_onset_sup_m1", "w_onset_sup_m2", "w_onset_sup_m3")},
}


def _check_accounting() -> None:
    """Every one of `inputs.all_params()` must be either varied or explicitly excluded.

    This is the guard that keeps the DSA honest as `inputs.py` evolves: a parameter added
    after Phase A2 cannot quietly acquire no tornado bar and no recorded reason.
    """
    varied = [v.param for v in VARIATIONS]
    assert len(varied) == len(set(varied)), "a parameter is varied twice"
    known = {p.name for p in I.all_params()}
    unknown = sorted((set(varied) | set(EXCLUSIONS)) - known)
    assert not unknown, f"registry names not in inputs.all_params(): {unknown}"
    overlap = sorted(set(varied) & set(EXCLUSIONS))
    assert not overlap, f"parameter both varied and excluded: {overlap}"
    missing = sorted(known - set(varied) - set(EXCLUSIONS))
    assert not missing, (
        f"{len(missing)} parameter(s) are neither varied nor explicitly excluded: "
        f"{missing}. Add each to VARIATIONS or to EXCLUSIONS with a reason -- a DSA that "
        "silently ignores a parameter is not a DSA."
    )
    for v in VARIATIONS:
        p = I.param_by_name(v.param)
        assert p.low <= p.value <= p.high, f"{v.param}: value outside [low, high]"
        if v.handling == "pair":
            assert v.partner and I.param_by_name(v.partner).group == p.group, \
                f"{v.param}: Dirichlet partner missing or in a different group"
        if v.handling == "kwarg":
            assert v.kwarg in ("n_sup", "n_deep"), f"{v.param}: bad kwarg {v.kwarg!r}"


_check_accounting()


# --------------------------------------------------------------------------- #
# 2. Running one bound                                                         #
# --------------------------------------------------------------------------- #
def _route(v: Variation, value: float) -> tuple[dict, dict]:
    """Turn one bound into `(sample, run_kwargs)` for `model.run_pair`.

    THE ONE PLACE THE THREE ROUTES ARE CHOSEN. In particular the tunnel widths go out as
    `n_sup=` / `n_deep=` and NEVER into `sample`: `model.run_arm` raises on a sample key
    that disagrees with the resolved layout, and a sample key that agreed would still set
    nothing -- either way the swing would not be the parameter's.
    """
    if v.handling == "kwarg":
        return {}, {v.kwarg: int(round(float(value)))}
    if v.handling == "pair":
        return {v.param: float(value), v.partner: 1.0 - float(value)}, {}
    if v.handling == "sample":
        return {v.param: float(value)}, {}
    raise ValueError(f"{v.param}: unknown handling {v.handling!r}")


def _incremental_at(sample: dict | None = None, *, p_base: float = P_DSA,
                    **run_kwargs) -> dict:
    """B_cinpwt MINUS A_standard at a DSA operating point, with overrides applied.

    `p_base` defaults to the Fig 5 stratum but is passed explicitly by `compute()` so the
    same registry runs unchanged at all three reported strata.

    A cost value inside `sample` is expressed in that parameter's OWN stored price year and
    is inflated exactly once inside the engine (`model._cost` -> `inputs.cost_2025`).
    This module never inflates anything itself and never reads a cost through `val()`.
    """
    res_a, res_b = model.run_pair(
        p_base=float(p_base), horizon_cycles=HORIZON,
        sample=(sample or None), **run_kwargs,
    )
    return model.incremental(res_b, res_a, wtp=WTP)


# --------------------------------------------------------------------------- #
# 3. Deliverables 1 and 2 -- dsa_results.csv / dsa_tornado.csv                  #
# --------------------------------------------------------------------------- #
def compute(stratum: str = DSA_STRATUM,
            p_base: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the one-way DSA at ONE stratum.

    Returns `(results_df, tornado_df, base_incremental)`. Both frames carry the CONSTANT
    `stratum` and `p_base` columns -- the same idiom as the constant `inmb_base` column --
    so that the three emitted tornados are distinguishable in the file itself and not only
    by the order they happen to be written in.
    """
    p_base = float(STRATA[stratum]) if p_base is None else float(p_base)
    base = _incremental_at(p_base=p_base)

    rows = []
    for v in VARIATIONS:
        p = I.param_by_name(v.param)
        lo_sample, lo_kw = _route(v, p.low)
        hi_sample, hi_kw = _route(v, p.high)
        lo = _incremental_at(lo_sample, p_base=p_base, **lo_kw)
        hi = _incremental_at(hi_sample, p_base=p_base, **hi_kw)

        rows.append({
            # `parameter` is column 0 on purpose: downstream audits grep this column for
            # 'tunnel' and for the counter names. Do NOT insert `rank` in front of it.
            "parameter": v.param,
            "label": v.label,
            "group": v.group,
            "handling": v.handling,
            "base_value": float(p.value),
            "low_value": float(p.low),
            "high_value": float(p.high),
            "inmb_base": base["inmb"],
            "inmb_low": lo["inmb"],
            "inmb_high": hi["inmb"],
            "inmb_swing": abs(hi["inmb"] - lo["inmb"]),
            "inc_cost_base": base["inc_cost"],
            "inc_cost_low": lo["inc_cost"],
            "inc_cost_high": hi["inc_cost"],
            "inc_qaly_base": base["inc_qaly"],
            "inc_qaly_low": lo["inc_qaly"],
            "inc_qaly_high": hi["inc_qaly"],
            "icer_low": lo["icer"],
            "icer_high": hi["icer"],
            "dominance_low": lo["dominance"],
            "dominance_high": hi["dominance"],
            "quadrant_low": lo["quadrant"],
            "quadrant_high": hi["quadrant"],
            "note": v.note,
        })

    df = pd.DataFrame(rows)
    # Sort by |INMB swing| descending. `icer` is NaN wherever ciNPWT is dominant -- which is
    # nearly everywhere at this stratum -- so it can never be the sort key. The secondary
    # key makes the ordering of the exact-zero counter rows deterministic run to run.
    df = df.sort_values(["inmb_swing", "parameter"], ascending=[False, True])
    df = df.reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)          # rank RESTARTS within each stratum

    # The two constant identity columns. Appended at the END so no positional reader of the
    # existing columns is disturbed and the documented "column 0 is `parameter` / `label`"
    # contract still holds; read them BY NAME.
    df["stratum"] = stratum
    df["p_base"] = p_base

    # Plot-ready subset for Fig 5. Column 0 is the human-readable LABEL (the y-axis
    # category); `inmb_base` is carried so the figure can draw the base-case reference line.
    tornado = df[["label", "low_value", "high_value", "inmb_low", "inmb_high",
                  "inmb_swing", "inmb_base", "rank", "parameter", "note",
                  "stratum", "p_base"]].copy()
    return df, tornado, base


def swing_at(param: str, p_base: float) -> float:
    """|INMB swing| of ONE registry row at an arbitrary baseline risk `p_base`.

    Runs the row exactly as `compute()` does -- same registry entry, same `_route`, same
    bounds from `inputs.py` -- so a swing quoted here can never be built differently from
    the swing in the CSV.
    """
    v = next(v for v in VARIATIONS if v.param == param)
    par = I.param_by_name(param)
    lo_sample, lo_kw = _route(v, par.low)
    hi_sample, hi_kw = _route(v, par.high)
    lo = _incremental_at(lo_sample, p_base=p_base, **lo_kw)["inmb"]
    hi = _incremental_at(hi_sample, p_base=p_base, **hi_kw)["inmb"]
    return abs(hi - lo)


def crossing_point(pair: tuple[str, str] = CROSSING_PAIR,
                   n_bisect: int = 60) -> dict:
    """The baseline risk at which the two bars in `pair` swap rank.

    BUILD-CONTRACT section 10's amendment states the mechanism: every SSI-cost bar scales
    with the number of SSIs prevented, i.e. roughly with `p`, while the device-price bar is
    a fixed `high - low` swing independent of `p` (a single additive cycle-0 toll at a
    discount factor of exactly 1). So `swing(m_ssi_cost) - swing(c_device_cinpwt)` is
    increasing in `p` and has one root. It is found by BISECTION ON THE ENGINE rather than
    by scaling a single measured swing, so the number reported in the manuscript is the
    model's, not an extrapolation.

    Bracketed by the reported risk axis `SETTINGS['risk_grid']`. Returns
    `{"p_star": None, ...}` if the pair never crosses inside it.
    """
    a, b = pair
    grid = np.asarray(I.SETTINGS["risk_grid"], dtype=float)
    lo, hi = float(grid[0]), float(grid[-1])

    def f(p: float) -> float:
        return swing_at(a, p) - swing_at(b, p)

    f_lo, f_hi = f(lo), f(hi)
    if not (f_lo < 0.0 < f_hi):
        return {"pair": pair, "p_star": None, "p_lo": lo, "p_hi": hi,
                "f_lo": f_lo, "f_hi": f_hi, "swing_a": None, "swing_b": None}
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    p_star = 0.5 * (lo + hi)
    return {"pair": pair, "p_star": p_star,
            "p_lo": float(grid[0]), "p_hi": float(grid[-1]),
            "f_lo": f_lo, "f_hi": f_hi,
            "swing_a": swing_at(a, p_star), "swing_b": swing_at(b, p_star)}


# --------------------------------------------------------------------------- #
# 4. I/O helpers                                                               #
# --------------------------------------------------------------------------- #
def _write_banner_csv(df: pd.DataFrame, filename: str) -> str:
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
    a = abs(float(x))
    return f"{sign}${a:,.0f}" if a >= 1 else f"{sign}${a:,.2f}"


def _fmt_icer(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a (dominant)"
    return f"${float(x):,.0f}/QALY"


def _fmt_val(v: Variation, x: float) -> str:
    """Format a parameter bound in its own units."""
    p = I.param_by_name(v.param)
    if p.price_year is not None:
        return f"${float(x):,.0f}"
    if v.handling == "kwarg":
        return f"{int(round(float(x)))} mo"
    if v.param == "age_start":
        return f"{float(x):.0f} y"
    return f"{float(x):g}"


# --------------------------------------------------------------------------- #
# 5. Deliverable 3 -- dsa_summary.md                                           #
# --------------------------------------------------------------------------- #
def build_summary_md(frames: dict, bases: dict, axis_rows: list[dict],
                     cross: dict) -> str:
    """`frames` / `bases` are keyed by stratum name and hold every emitted tornado; the
    ranked table and the narrative are written at `DSA_STRATUM` (the Fig 5 panel), and the
    three-stratum ranking + crossing point get their own section."""
    by_name = {v.param: v for v in VARIATIONS}
    df = frames[DSA_STRATUM]
    base = bases[DSA_STRATUM]
    top = df.iloc[0]
    m_row = df[df["parameter"] == "m_ssi_cost"].iloc[0]
    top_is_m = str(top["parameter"]) == "m_ssi_cost"
    ok_strata = [s for s in SECTION10_STRATA
                 if str(frames[s].iloc[0]["parameter"]) == "m_ssi_cost"]

    per_case_base = model.ssi_cost_per_case()
    per_case_lo = model.ssi_cost_per_case(
        sample={"m_ssi_cost": float(I.cost.m_ssi_cost.low)})
    per_case_hi = model.ssi_cost_per_case(
        sample={"m_ssi_cost": float(I.cost.m_ssi_cost.high)})
    # Deep-vs-superficial ratios, derived live from inputs.py (BUILD-CONTRACT section 9:
    # a run script may not own a number, not even one used only in prose).
    cost_ratio = per_case_base["deep"] / per_case_base["superficial"]
    qaly_ratio = ((float(I.util.du_deep) * I.DEEP_LEN)
                  / (float(I.util.du_sup) * I.SUP_LEN))

    L: list[str] = []
    L.append("# One-way deterministic sensitivity analysis (tornado) -- ciNPWT after "
             "instrumented posterior lumbar fusion")
    L.append("")
    L.append("> **" + I.RESULTS_BANNER + "**")
    L.append("")
    L.append("*Generated by `run_dsa.py` from the read-only engine (`inputs.py` / "
             "`model.py`). Model-based result; not a substitute for clinical evidence.*")
    L.append("")

    # ---- operating point ----
    L.append("## Where this analysis sits")
    L.append("")
    L.append(f"- **Risk stratum.** The ranked table and the narrative below are the "
             f"`{DSA_STRATUM}` stratum, baseline 90-day all-SSI risk **p = {P_DSA:.1%}** "
             "(the Edmiston [12] anchor) -- the panel the plan specifies for **Fig 5**. "
             "One-way DSA is a *local* analysis: every bar below is the swing **at this "
             "p**, and the ranking does change at other strata.")
    L.append(f"- **All three reported strata are emitted.** The full tornado is run at "
             + ", ".join(f"**{s} (p = {p:.1%})**" for s, p in DSA_STRATA)
             + " and all "
             f"{len(DSA_STRATA) * len(VARIATIONS)} rows are written to "
             "`dsa_results.csv` and `dsa_tornado.csv`, each carrying constant **`stratum`** "
             "and **`p_base`** columns. Low and high are **supplementary**; "
             f"`{DSA_STRATUM}` is Fig 5. **Filter on `stratum`** -- do not read the first "
             "rows of the file. See \"The same tornado at all three strata\" below.")
    L.append(f"- **Horizon.** {HORIZON} cycles ({HORIZON // 12} years), "
             f"{I.SETTINGS['disc_cost']:.0%}/yr discounting on costs and QALYs, "
             "trapezoidal half-cycle correction on dwell streams only, "
             f"{I.PRICE_YEAR} USD.")
    L.append(f"- **Decision rule.** INMB at WTP **${WTP:,.0f}/QALY**, `B_cinpwt` minus "
             "`A_standard`. Base case: incremental cost "
             f"**{_fmt_money(base['inc_cost'])}**, incremental QALY "
             f"**{base['inc_qaly']:+.5f}**, **INMB {_fmt_money(base['inmb'])}** "
             f"(*{base['dominance']}*, CE quadrant `{base['quadrant']}`).")
    L.append("- **Sort key: `|INMB swing|`, descending -- not ICER.** ciNPWT is strictly "
             "dominant across essentially the whole range at this stratum, so `icer` is "
             "`NaN` on almost every row. It is reported in `dsa_results.csv` as a "
             "diagnostic and must never be used to rank or to plot.")
    L.append("")

    # ---- the headline ----
    L.append("## Does SSI-attributable cost top the tornado, at the intermediate and high "
             "strata where the DSA is specified to run? (BUILD-CONTRACT section 10)")
    L.append("")
    if top_is_m and len(ok_strata) == len(SECTION10_STRATA):
        second = df.iloc[1]
        L.append(f"**YES -- at the intermediate and high strata, which is the whole of "
                 f"what section 10 requires.** At the `{DSA_STRATUM}` stratum "
                 f"(p = {P_DSA:.1%}) `m_ssi_cost` is the top bar with an INMB swing of "
                 f"**{_fmt_money(m_row['inmb_swing'])}** -- "
                 f"{m_row['inmb_swing'] / max(float(second['inmb_swing']), 1e-9):.2f}x the "
                 f"next bar (`{second['parameter']}`, "
                 f"{_fmt_money(second['inmb_swing'])}); at the high stratum "
                 f"(p = {float(STRATA['high']):.1%}) it leads by "
                 f"{_fmt_money(frames['high'].iloc[0]['inmb_swing'])} against "
                 f"{_fmt_money(frames['high'].iloc[1]['inmb_swing'])} for "
                 f"`{frames['high'].iloc[1]['parameter']}`. "
                 "**BUILD-CONTRACT section 10, as amended 2026-08-09, is satisfied at both "
                 "strata where it applies.**")
        L.append("")
        L.append(f"**It is NOT satisfied at the low stratum, and it is not supposed to "
                 f"be.** At p = {float(STRATA['low']):.1%} the top bar is "
                 f"`{frames['low'].iloc[0]['parameter']}` "
                 f"({_fmt_money(frames['low'].iloc[0]['inmb_swing'])}) against "
                 f"`m_ssi_cost` at "
                 f"{_fmt_money(float(frames['low'][frames['low']['parameter'] == 'm_ssi_cost'].iloc[0]['inmb_swing']))}. "
                 "The contract's amendment states this outcome in advance and forbids "
                 "\"fixing\" it: it is a **result**, quantified in the next section.")
    else:
        L.append(f"**NO -- and this file should not have shipped.** At `{DSA_STRATUM}` the "
                 f"top bar is `{top['parameter']}` ({_fmt_money(top['inmb_swing'])}) "
                 f"against `m_ssi_cost` at {_fmt_money(m_row['inmb_swing'])}; section 10 "
                 f"holds at {len(ok_strata)} of {len(SECTION10_STRATA)} required strata "
                 f"({', '.join(ok_strata) or 'none'}). BUILD-CONTRACT section 10 is "
                 "violated; investigate before using these numbers.")
    L.append("")
    L.append("**Why one bar and not six.** SSI-attributable cost is carried by the single "
             "dimensionless multiplier `m_ssi_cost` "
             f"({float(I.cost.m_ssi_cost.low):.3f}-{float(I.cost.m_ssi_cost.high):.3f}, "
             f"base {float(I.cost.m_ssi_cost):.3f}), which scales **all six** Edmiston "
             "tranches together. The six "
             "tranches are deliberately **not** additionally swung as separate bars. Two "
             "reasons, and they are the same reason from two directions:")
    L.append("")
    L.append("1. **It would double-count.** The tranches are six windows of one quantity. "
             "The real uncertainty is not \"is the 6-12-month deep tranche $15,109 or "
             "$22,664\" -- it is *whether Edmiston's regression-adjusted all-cause "
             "increment is the right attribution at all*, against Pennington's matched "
             "increment (a 4.5-fold span). That is a single structural question, and it "
             "moves all six tranches in the same direction at once.")
    L.append("2. **It would hide the pivotal parameter.** Six independent bars each carry "
             "a fraction of the swing, so the parameter that actually decides the answer "
             "would appear as six mid-table bars and something less important would top "
             "the chart. BUILD-CONTRACT section 10 exists to prevent exactly that.")
    L.append("")
    L.append(f"Weighted 24-month cost per SSI moves from "
             f"**${per_case_lo['weighted']:,.0f}** at `m_ssi_cost` = "
             f"{float(I.cost.m_ssi_cost.low):.3f} (Pennington's matched increment) through "
             f"**${per_case_base['weighted']:,.0f}** at base to "
             f"**${per_case_hi['weighted']:,.0f}** at "
             f"{float(I.cost.m_ssi_cost.high):.3f} (the deep-SSI value applied to every "
             f"SSI) -- all in {I.PRICE_YEAR} USD.")
    L.append("")

    # ---- the three strata, and the crossing ----
    L.append("## The same tornado at all three strata -- the ranking rotates with risk")
    L.append("")
    L.append("The full registry was run at every reported stratum and all three tornados "
             "are in `dsa_results.csv` / `dsa_tornado.csv` (filter on `stratum`). The four "
             "leading bars, in order:")
    L.append("")
    L.append("| Stratum | p | 1st | 2nd | 3rd | 4th |")
    L.append("|---|---:|---|---|---|---|")
    for sname, p in sorted(DSA_STRATA, key=lambda kv: kv[1]):
        f = frames[sname]
        cells = [f"`{f.iloc[k]['parameter']}` {_fmt_money(f.iloc[k]['inmb_swing'])}"
                 for k in range(4)]
        cells[0] = f"**{cells[0]}**"
        L.append(f"| {sname} | {p:.1%} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("**The mechanism is exact, not a numerical accident.** Every SSI-related bar "
             "scales with the number of SSIs prevented, i.e. roughly with `p`, while the "
             "device-price bar is a fixed "
             f"{_fmt_money(float(I.cost.c_device_cinpwt.high) - float(I.cost.c_device_cinpwt.low))} "
             "swing at every `p`: the device is a single additive toll at cycle 0, where "
             "the discount factor is exactly 1 and no half-cycle weight applies, so INMB "
             "moves dollar-for-dollar with it and the bar length is exactly `high - low`, "
             "independent of the risk. The two therefore cross once.")
    L.append("")
    if cross["p_star"] is not None:
        a, b = cross["pair"]
        L.append(f"**They cross at p = {cross['p_star'] * 100:.3f}%** "
                 f"(bisection on the engine; at that risk both bars are "
                 f"{_fmt_money(cross['swing_a'])}). Below it `{b}` is the top bar, above "
                 f"it `{a}` is -- which is why BUILD-CONTRACT section 10 is written as a "
                 "**stratum-conditional** requirement and why the low-stratum tornado must "
                 "not be \"fixed\" to force `m_ssi_cost` to the top.")
    else:                                                    # pragma: no cover
        L.append(f"`{cross['pair'][0]}` and `{cross['pair'][1]}` do **not** cross inside "
                 f"the reported risk axis ({cross['p_lo']:.1%}-{cross['p_hi']:.1%}); "
                 "the ranking is stable across it.")
    L.append("")
    # Which bounds, at which stratum, actually take INMB below zero. At the intermediate
    # stratum nothing does; at the low stratum the device price alone does, which is the
    # same finding as the crossing seen from the decision side rather than the ranking side.
    flips = []
    for sname, p in sorted(DSA_STRATA, key=lambda kv: kv[1]):
        f = frames[sname]
        for _, r in f[(f["inmb_low"] < 0) | (f["inmb_high"] < 0)].iterrows():
            bound = "low" if float(r["inmb_low"]) < 0 else "high"
            flips.append((sname, p, str(r["parameter"]), bound,
                          float(r[f"inmb_{bound}"]), float(r[f"{bound}_value"]),
                          by_name[str(r["parameter"])]))
    if flips:
        L.append("**And it shows up in the decision, not only in the ranking.** Across all "
                 f"{len(DSA_STRATA) * len(VARIATIONS)} rows, "
                 + (f"exactly {len(flips)} bound takes" if len(flips) == 1
                    else f"{len(flips)} bounds take")
                 + " INMB below $0:")
        L.append("")
        for sname, p, name, bound, inmb, val, v in flips:
            L.append(f"- **{sname} stratum (p = {p:.1%})**, `{name}` at its **{bound}** "
                     f"bound ({_fmt_val(v, val)}): INMB **{_fmt_money(inmb)}**.")
        L.append("")
        flip_strata = sorted({f[0] for f in flips}, key=lambda s: float(STRATA[s]))
        clean = [s for s, _ in sorted(DSA_STRATA, key=lambda kv: kv[1])
                 if s not in flip_strata]
        L.append(f"At the {' and '.join(clean)} "
                 + ("stratum" if len(clean) == 1 else "strata")
                 + " **no** single parameter's one-way range can overturn the conclusion. "
                 f"At the {' and '.join(flip_strata)} "
                 + ("stratum" if len(flip_strata) == 1 else "strata")
                 + " the bounds listed above can -- and note *which* they are: "
                 + " and ".join(f"`{n}`" for n in
                                dict.fromkeys(f[2] for f in flips))
                 + ". That is the crossing above seen from the decision side rather than "
                 "the ranking side: at low risk the answer is decided by what the device "
                 "costs against how much of the infection cost is really attributable, "
                 "and everything else is second order.")
        L.append("")
    # A2 2026-08-17: this sentence carried a hard-coded 1.5% -- the pre-audit crossing,
    # and also the pre-audit low stratum. Both moved. It is now derived from the crossing
    # actually computed above, and the direction it reports is checked against the strata
    # rather than assumed, because the whole point of the post-A2 sentence is that the
    # crossing now sits ABOVE the low stratum instead of below it.
    if cross["p_star"] is not None:
        _x = cross["p_star"]
        _lo = float(I.SETTINGS["strata"]["low"])
        _rel = ("above" if _x > _lo else "below")
        L.append(
            f"**The reading, in Results language:** *below roughly "
            f"{_x * 100:.3f}% baseline 90-day SSI risk the decision is governed by the "
            f"price of the device; above it, by the cost of an infection.* That crossing "
            f"sits **{_rel}** the low stratum ({_lo * 100:.1f}%), so across the whole of "
            f"that stratum the decision turns on the device price. This belongs in the "
            f"manuscript, not only in a sensitivity appendix -- it is the same boundary "
            f"the threshold analysis reaches from the other direction, and it tells a "
            f"surgeon which number to argue about for a given patient. **Note the "
            f"crossing depends on the width of the device bar, which is the reported "
            f"price axis ($100-$900 after A2 widened it to cover Mueller 2026 [9]'s two "
            f"published list prices), not an evidence-derived range** -- a wider axis "
            f"moves the crossing right.")
    L.append("")
    L.append("Two things this table is **not**. It is not three independent analyses: the "
             "bars move because `p` moves, not because any parameter was re-estimated. And "
             "it is not a licence to read the low-stratum panel as the headline -- **Fig 5 "
             f"is the `{DSA_STRATUM}` stratum** (plan Phase C); low and high are "
             "supplementary and must be labelled as such wherever they appear.")
    L.append("")

    # ---- ranked table ----
    L.append(f"## Ranked tornado at the `{DSA_STRATUM}` stratum, p = {P_DSA:.1%} "
             "(sorted by |INMB swing|, descending) -- the Fig 5 panel")
    L.append("")
    L.append(f"| # | Parameter | Low -> high | INMB at low | INMB at high | "
             f"**\\|INMB swing\\|** | Inc. cost (low / high) | Dominance (low / high) |")
    L.append("|---:|---|---|---:|---:|---:|---|---|")
    for _, r in df.iterrows():
        v = by_name[str(r["parameter"])]
        rng = f"{_fmt_val(v, r['low_value'])} -> {_fmt_val(v, r['high_value'])}"
        dom = (f"{r['dominance_low']} / {r['dominance_high']}"
               if r["dominance_low"] != r["dominance_high"]
               else str(r["dominance_low"]))
        L.append(f"| {int(r['rank'])} | {r['label']} | {rng} "
                 f"| {_fmt_money(r['inmb_low'])} | {_fmt_money(r['inmb_high'])} "
                 f"| **{_fmt_money(r['inmb_swing'])}** "
                 f"| {_fmt_money(r['inc_cost_low'])} / {_fmt_money(r['inc_cost_high'])} "
                 f"| {dom} |")
    L.append("")
    L.append(f"Base-case INMB is **{_fmt_money(base['inmb'])}**; every bar is centred on "
             "that value in Fig 5. A bar whose *whole* span stays above $0 is a parameter "
             "that cannot overturn the conclusion on its own.")
    L.append("")

    # ---- does anything flip the sign? ----
    flippers = df[(df["inmb_low"] < 0) | (df["inmb_high"] < 0)]
    L.append("### Does any single parameter overturn the conclusion?")
    L.append("")
    if flippers.empty:
        L.append(f"**No.** At p = {P_DSA:.1%} every parameter's *entire* one-way range "
                 f"leaves INMB positive -- the minimum INMB anywhere in this table is "
                 f"**{_fmt_money(df['inmb_low'].combine(df['inmb_high'], min).min())}**, "
                 "and ciNPWT is **dominant or cost-effective at every bound of every "
                 "parameter**. The intermediate-risk conclusion is robust to one-way "
                 "variation; what moves it is the *joint* configuration explored by "
                 "`run_scenarios.py` and `run_psa.py`, and above all the baseline risk "
                 "itself.")
    else:
        L.append("**Yes -- these parameters push INMB below $0 at one of their bounds:**")
        L.append("")
        for _, r in flippers.iterrows():
            bound = "low" if r["inmb_low"] < 0 else "high"
            L.append(f"- `{r['parameter']}` at its **{bound}** bound: INMB "
                     f"{_fmt_money(r[f'inmb_{bound}'])}.")
    L.append("")

    # ---- narrative ----
    L.append("## Reading the drivers")
    L.append("")
    ordered = list(df["parameter"])
    rank_of = {n: i + 1 for i, n in enumerate(ordered)}
    L.append(f"**1. SSI-attributable cost (`m_ssi_cost`), "
             f"{_fmt_money(m_row['inmb_swing'])}.** The whole result is an arbitrage "
             "between a $250 device and the cost of the SSIs it prevents, so the size of "
             "the avoided cost is the size of the answer. At the Pennington end of the "
             "attribution envelope the INMB falls to "
             f"{_fmt_money(m_row['inmb_low'])} and at the Edmiston-deep end it rises to "
             f"{_fmt_money(m_row['inmb_high'])}. **This is a structural disagreement "
             "between sources, not sampling noise** -- which is why it is characterized "
             f"here and in the all-cause x1.000 sensitivity, and *not* in the "
             "PSA (`m_ssi_cost` is `dist='fixed'`; see the limitation `run_psa.py` is "
             "required to state).")
    L.append("")
    rr_row = df[df["parameter"] == "rr_cinpwt"].iloc[0]
    L.append(f"**{rank_of['rr_cinpwt']}. ciNPWT relative risk (`rr_cinpwt`), "
             f"{_fmt_money(rr_row['inmb_swing'])} -- the correlated arm gap.** RR is "
             "applied to the onset hazard *inside* the engine (`p_ciNPWT = p_control x "
             "RR`), so this one row is the relative-effect uncertainty in its correct "
             "correlated form: arm A is held fixed and arm B moves relative to it. The "
             f"direction is inverted relative to most rows -- the *low* bound "
             f"(RR = {float(I.tp.rr_cinpwt.low):.2f}, a larger effect) gives the *higher* "
             f"INMB ({_fmt_money(rr_row['inmb_low'])}) and the high bound "
             f"(RR = {float(I.tp.rr_cinpwt.high):.2f}) the lower "
             f"({_fmt_money(rr_row['inmb_high'])}).")
    L.append("")
    fd_row = df[df["parameter"] == "frac_deep"].iloc[0]
    L.append(f"**{rank_of['frac_deep']}. Deep fraction of SSI (`frac_deep`), "
             f"{_fmt_money(fd_row['inmb_swing'])} -- the Dirichlet pair.** Varied as a "
             "**pair**: the partner `frac_sup` is set to `1 - frac_deep` at each bound, so "
             "the destination split always sums to 1. Total incidence is invariant by "
             "construction (the engine sets the total hazard first and splits afterwards), "
             "so this bar moves **composition** only -- but composition is expensive: a "
             f"deep SSI costs **{cost_ratio:.1f}x** a superficial one over 24 months and "
             f"carries **{qaly_ratio:.1f}x** the QALY loss. The wide "
             f"{float(I.tp.frac_deep.low):.2f}-{float(I.tp.frac_deep.high):.2f} range is "
             "the unresolved conflict between Edmiston's 74%-deep composition [12] and "
             "Freire-Archer's 1.5% pooled deep rate [17].")
    L.append("")
    dev_row = df[df["parameter"] == "c_device_cinpwt"].iloc[0]
    L.append(f"**{rank_of['c_device_cinpwt']}. ciNPWT device unit cost "
             f"(`c_device_cinpwt`), {_fmt_money(dev_row['inmb_swing'])}.** Exactly the "
             f"width of the price range (${float(I.cost.c_device_cinpwt.low):,.0f}-"
             f"${float(I.cost.c_device_cinpwt.high):,.0f}), because the device is a single "
             "additive toll at cycle 0 where the discount factor is 1 and no half-cycle "
             "weight applies: INMB moves dollar-for-dollar. It is the largest evidence gap "
             "in the model and is reported in full as analysis axis 2 "
             "(`run_threshold.py`, Fig 3), not only as this bar.")
    L.append("")

    # ---- tunnel rows ----
    tun = df[df["parameter"].isin(["n_tunnel_deep", "n_tunnel_sup"])]
    L.append("## Tunnel duration -- swept through the layout kwargs, not through `sample`")
    L.append("")
    L.append("`n_tunnel_sup` (2-6 months) and `n_tunnel_deep` (6-24 months) carry DSA "
             "ranges, but tunnel width is a **layout**, not a sampled parameter: it is set "
             "by `run_arm(n_sup=..., n_deep=...)`. A `sample` dict carrying them would "
             "have set nothing, and the engine now **raises** on a tunnel key that "
             "disagrees with the resolved layout precisely so that failure cannot be "
             "silent. Both rows below therefore carry a **real, non-zero** swing:")
    L.append("")
    L.append("| Parameter | Range | INMB at low | INMB at high | \\|INMB swing\\| | "
             "Inc. QALY (low -> high) | Inc. cost (low -> high) |")
    L.append("|---|---|---:|---:|---:|---|---|")
    for _, r in tun.iterrows():
        v = by_name[str(r["parameter"])]
        L.append(f"| `{r['parameter']}` "
                 f"| {_fmt_val(v, r['low_value'])} -> {_fmt_val(v, r['high_value'])} "
                 f"| {_fmt_money(r['inmb_low'])} | {_fmt_money(r['inmb_high'])} "
                 f"| {_fmt_money(r['inmb_swing'])} "
                 f"| {r['inc_qaly_low']:+.5f} -> {r['inc_qaly_high']:+.5f} "
                 f"| {_fmt_money(r['inc_cost_low'])} -> {_fmt_money(r['inc_cost_high'])} |")
    L.append("")
    L.append("Note that the incremental **cost** is identical at both bounds of both rows. "
             "That is correct and is a property of the accounting, not an artefact: the "
             "SSI cost tolls are attached to the **onset cohort**, not to tunnel occupancy, "
             "so tunnel width moves the **QALY side only**. This also resolves objection 3 "
             "in `run_log.d/inputs.md`, which warned that these two parameters would "
             "otherwise appear as dead zero-swing rows.")
    L.append("")

    # ---- counters ----
    L.append("## Counters -- an INMB swing of exactly zero, and that is the finding")
    L.append("")
    L.append("Plan **Decision 3**: `p_persistent_deep`, `p_reop_ssi`, `n_debridements` and "
             "`rr_readmit_sa` are counters and cross-checks that carry **no** incremental "
             "cost. Edmiston's payment is an *all-cause* incremental spend that already "
             "contains reoperation, readmission, I&D, revision and antibiotics; costing "
             "any of them again would double-count. They are reported here as explicit "
             "**exact-zero** rows rather than dropped, so a reader can see the decision is "
             "real and enforced -- `main()` asserts the zeros and fails loudly if a "
             "counter ever starts moving the result.")
    L.append("")
    L.append("| Parameter | Range | INMB swing | Why zero |")
    L.append("|---|---|---:|---|")
    for name in COUNTER_ROWS:
        r = df[df["parameter"] == name].iloc[0]
        v = by_name[name]
        read = {
            "p_reop_ssi": "read by the engine (populates `n_reop`) but multiplies no cost",
            "p_persistent_deep":
                "read by the engine (populates `n_persistent`) but multiplies no cost",
        }.get(name, "not read by the engine at all -- a published cross-check only")
        L.append(f"| `{name}` | {_fmt_val(v, r['low_value'])} -> "
                 f"{_fmt_val(v, r['high_value'])} "
                 f"| **{_fmt_money(r['inmb_swing'])}** | {read} |")
    L.append("")
    uw = df[df["parameter"] == "u_well"].iloc[0]
    L.append(f"A fifth near-zero row is **not** a Decision-3 counter and should be read "
             f"differently: `u_well` swings {_fmt_money(uw['inmb_swing'])} across "
             f"{float(I.util.u_well.low):.2f}-{float(I.util.u_well.high):.2f}. The "
             "absolute well-state utility is an explicitly labelled `derived_assumption` "
             "(BUILD-CONTRACT section 6.7), and this row is its numerical vindication: "
             "because mortality is arm-independent, the two arms hold identical live mass "
             "in every cycle, so `u_well` multiplies a quantity that differences to zero "
             "and cancels from the incremental QALY to machine precision. It scales each "
             "arm's QALY **level**, not the **difference**.")
    L.append("")

    # ---- the analysis axis ----
    L.append("## The analysis axis -- reported here, deliberately NOT a tornado bar")
    L.append("")
    L.append("`p_base_ssi` is the study's **x-axis**, not an uncertain input. It is "
             "excluded from `dsa_results.csv` and `dsa_tornado.csv` on purpose: its "
             f"{float(I.tp.p_base_ssi.low):g}-{float(I.tp.p_base_ssi.high):g} span "
             "produces a larger INMB swing than any genuine parameter, so including it "
             "would displace the pivotal bar and violate BUILD-CONTRACT section 10 while "
             "answering a question the paper reports properly elsewhere. The spread across "
             "the three reported strata, for orientation only:")
    L.append("")
    L.append("| Stratum | p | Incremental cost | Incremental QALY | INMB | Dominance |")
    L.append("|---|---:|---:|---:|---:|---|")
    for a in axis_rows:
        L.append(f"| {a['stratum']} | {a['p']:.1%} | {_fmt_money(a['inc_cost'])} "
                 f"| {a['inc_qaly']:+.5f} | {_fmt_money(a['inmb'])} | {a['dominance']} |")
    L.append("")
    L.append("**Do not plot these as tornado bars.** The full 291-point sweep, the "
             "break-even risk and the two-way risk x price map are `run_threshold.py`'s "
             "deliverables (Figs 2-3).")
    L.append("")

    # ---- coverage audit ----
    L.append("## Coverage -- every parameter accounted for")
    L.append("")
    L.append(f"`inputs.all_params()` holds **{len(I.all_params())}** parameters. "
             f"**{len(VARIATIONS)}** carry a tornado bar **at each of the "
             f"{len(DSA_STRATA)} strata**; **{len(EXCLUSIONS)}** are excluded with a "
             "recorded reason. The split is asserted at import, so a parameter added "
             "after Phase A2 must be given either a bar or a written reason -- it cannot "
             "be silently ignored.")
    L.append("")
    L.append("| Excluded parameter | Reason |")
    L.append("|---|---|")
    for name in [p.name for p in I.all_params() if p.name in EXCLUSIONS]:
        L.append(f"| `{name}` | {EXCLUSIONS[name]} |")
    L.append("")

    # ---- caveats ----
    L.append("## What this tornado does and does not carry")
    L.append("")
    L.append("- **Verified.** Phase A2 full-text verification completed 2026-08-17: 46 of 48 parameters are closed out against the 14 source full texts and 43 carry a table/page locator. Only 2 still carry `verify=True` -- the ciNPWT institutional net acquisition price and the superficial tunnel length -- because no available source resolves either. Every bound above is a "
             "verified range except those two.")
    L.append("- **One-way and local.** Each bar holds every other parameter at base. "
             "Interactions are not captured here -- `run_scenarios.py` handles the "
             "defined joint configurations and `run_psa.py` the joint probabilistic "
             "spread.")
    L.append("- **The ranking is stratum-specific -- and all three strata are shipped.** "
             f"The table above is computed at p = {P_DSA:.1%}. Bars that scale with the "
             "number of SSIs prevented (`m_ssi_cost`, `rr_cinpwt`, `frac_deep`) shrink as "
             "p falls, while the device-price bar does not move at all, so at low risk the "
             "device price takes the top of the chart"
             + (f" (crossing at p = {cross['p_star'] * 100:.3f}%)"
                if cross["p_star"] is not None else "")
             + ". Quantified in \"The same tornado at all three strata\" above; the low "
             "and high panels are in the CSVs under `stratum`.")
    L.append("- **Ranges are plausible spans, not confidence intervals.** Several are "
             "explicitly assumed (the +/-50% cost-tranche spans -- read as approximate 95% "
             "CIs, i.e. a per-tranche CV of about 0.26, not 0.5 -- the 2-6-month "
             "superficial tunnel, the 55-70 age span, the 0.60-0.80 `u_well` span); they "
             "are ranges for a sensitivity analysis, and a bar length is not an "
             "inferential statement.")
    L.append("- **The top two bars are not commensurable, and the Fig 5 caption must say "
             "so.** `m_ssi_cost` spans a **full structural envelope** (Edmiston's "
             "regression-adjusted increment against Pennington's matched increment, a "
             "4.5-fold span) while `rr_cinpwt` spans a **95% CI**. Two different kinds of "
             "statement are being rendered on one axis; the ranking should not be read as "
             "implying they carry comparable precision (BUILD-CONTRACT section 10).")
    L.append("- **`icer` is `NaN` almost everywhere** because ciNPWT is dominant. Rank and "
             "plot on `inmb_swing`; use `dominance` / `quadrant` as the reporting fields.")
    L.append("")
    L.append("---")
    L.append("")
    L.append(f"*Both CSVs are LONG-FORMAT over {len(DSA_STRATA)} strata "
             f"({len(VARIATIONS)} rows each, {len(DSA_STRATA) * len(VARIATIONS)} in total), "
             f"with the Fig 5 stratum (`{DSA_STRATUM}`) written first. **Filter on the "
             "`stratum` column** -- `stratum` and `p_base` are constant within a block, in "
             "the same idiom as `inmb_base`, and `rank` restarts at 1 in each block.*")
    L.append("")
    L.append("*Files written: `dsa_results.csv`, `dsa_tornado.csv`, `dsa_summary.md` -- all "
             "in `v2/results/`, each CSV carrying the results banner as line 1 (read "
             "with `pd.read_csv(path, skiprows=1)`).*")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "dsa_summary.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    return path


# --------------------------------------------------------------------------- #
# 6. Main                                                                      #
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- the full tornado at every reported stratum, PRIMARY FIRST ---- #
    frames, tornados, bases = {}, {}, {}
    for sname, p in DSA_STRATA:
        frames[sname], tornados[sname], bases[sname] = compute(sname, p)
    df, base = frames[DSA_STRATUM], bases[DSA_STRATUM]
    results_all = pd.concat([frames[s] for s, _ in DSA_STRATA], ignore_index=True)
    tornado_all = pd.concat([tornados[s] for s, _ in DSA_STRATA], ignore_index=True)

    # Where the pivotal bar and the device-price bar swap rank (section 10's amendment).
    cross = crossing_point()

    # The analysis axis, for the summary only -- never written to either CSV, so no figure
    # can pick it up as a bar (it would outswing the pivotal parameter and break section 10).
    axis_rows = []
    for sname, p in STRATA.items():
        res_a, res_b = model.run_pair(p_base=float(p), horizon_cycles=HORIZON)
        inc = model.incremental(res_b, res_a, wtp=WTP)
        axis_rows.append({"stratum": sname, "p": float(p), **inc})

    p_results = _write_banner_csv(results_all, "dsa_results.csv")
    p_tornado = _write_banner_csv(tornado_all, "dsa_tornado.csv")
    p_summary = build_summary_md(frames, bases, axis_rows, cross)

    # ---- stdout summary ---- #
    print("=" * 78)
    print("SSI / ciNPWT risk-threshold Markov CUA -- ONE-WAY DSA (TORNADO)")
    print(I.RESULTS_BANNER)
    print("=" * 78)
    print(f"Strata emitted: "
          + ", ".join(f"{s} (p = {p:.1%})" for s, p in DSA_STRATA)
          + f" | Fig 5 panel = {DSA_STRATUM}")
    print(f"Horizon {HORIZON} cycles | disc {I.SETTINGS['disc_cost']:.0%} | "
          f"WTP ${WTP:,.0f}/QALY | {I.PRICE_YEAR} USD")
    print(f"Ranked table below is the {DSA_STRATUM} stratum (p = {P_DSA:.1%}); "
          f"{len(results_all)} rows written across {len(DSA_STRATA)} strata "
          f"(filter on the `stratum` column).")
    print(f"Base case: inc cost {_fmt_money(base['inc_cost'])} | inc QALY "
          f"{base['inc_qaly']:+.5f} | INMB {_fmt_money(base['inmb'])} | "
          f"{base['dominance']} | ICER {_fmt_icer(base['icer'])}")
    print(f"{len(VARIATIONS)} parameters varied, {len(EXCLUSIONS)} excluded with a reason "
          f"({len(I.all_params())} in the store).")
    print("-" * 78)
    print(f"{'#':>3}  {'parameter':<24s}{'low':>12s}{'high':>12s}"
          f"{'INMB low':>12s}{'INMB high':>12s}{'|swing|':>12s}")
    print("-" * 78)
    for _, r in df.iterrows():
        print(f"{int(r['rank']):>3}  {r['parameter']:<24s}"
              f"{r['low_value']:>12g}{r['high_value']:>12g}"
              f"{r['inmb_low']:>12,.0f}{r['inmb_high']:>12,.0f}"
              f"{r['inmb_swing']:>12,.2f}")
    print("-" * 78)

    # ---- the three-stratum ranking and the crossing point ---- #
    print()
    print("THE SAME TORNADO AT ALL THREE STRATA (top four bars; the ranking rotates "
          "with p):")
    for sname, p in sorted(DSA_STRATA, key=lambda kv: kv[1]):
        f = frames[sname]
        cells = "  ".join(f"{f.iloc[k]['parameter']} ${f.iloc[k]['inmb_swing']:,.0f}"
                          for k in range(4))
        print(f"    {sname:<14s} p={p:.3f} : {cells}")
    if cross["p_star"] is not None:
        a, b = cross["pair"]
        print(f"    CROSSING: {a} overtakes {b} at p = {cross['p_star'] * 100:.3f}% "
              f"(both bars ${cross['swing_a']:,.0f} there). Below it the device price "
              f"governs the decision; above it, the cost of an infection.")
    else:                                                    # pragma: no cover
        print(f"    NO CROSSING inside the reported risk axis "
              f"({cross['p_lo']:.1%}-{cross['p_hi']:.1%}).")
    print()

    # ---- BUILD-CONTRACT section 10: fail fast rather than ship a wrong tornado ---- #
    # AMENDED 2026-08-09: the requirement is STRATUM-CONDITIONAL -- m_ssi_cost must top the
    # tornado at the intermediate and high strata, "which is where the DSA is specified to
    # run". At the low stratum the device price leads, and the contract explicitly forbids
    # "fixing" that. So SECTION10_STRATA is asserted and the low stratum is REPORTED.
    for sname in SECTION10_STRATA:
        f = frames[sname]
        top_s = f.iloc[0]
        if str(top_s["parameter"]) != "m_ssi_cost":
            m = f[f["parameter"] == "m_ssi_cost"].iloc[0]
            raise AssertionError(
                "BUILD-CONTRACT section 10 VIOLATION: SSI-attributable cost is not the top "
                f"tornado bar at the {sname!r} stratum (p = {float(STRATA[sname]):.1%}), "
                f"where section 10 requires it. Top bar is {top_s['parameter']!r} with a "
                f"swing of ${float(top_s['inmb_swing']):,.0f} against m_ssi_cost at "
                f"${float(m['inmb_swing']):,.0f}. Investigate before shipping: check that "
                "m_ssi_cost reached the engine (it is read through val(), not cost_2025()), "
                "that p_base_ssi has not leaked into the tornado, and that the six cost "
                "tranches are not being swung separately."
            )
        second_s = f.iloc[1]
        print(f"[OK] BUILD-CONTRACT section 10 at the {sname} stratum "
              f"(p = {float(STRATA[sname]):.1%}): m_ssi_cost IS the top bar, "
              f"${float(top_s['inmb_swing']):,.0f} "
              f"({float(top_s['inmb_swing']) / max(float(second_s['inmb_swing']), 1e-9):.2f}x "
              f"the next bar, {second_s['parameter']}).")
    for sname, _ in DSA_STRATA:
        if sname in SECTION10_STRATA:
            continue
        f = frames[sname]
        m = f[f["parameter"] == "m_ssi_cost"].iloc[0]
        print(f"[--] {sname} stratum (p = {float(STRATA[sname]):.1%}) is OUTSIDE section "
              f"10's requirement: top bar is {f.iloc[0]['parameter']} "
              f"${float(f.iloc[0]['inmb_swing']):,.0f} vs m_ssi_cost "
              f"${float(m['inmb_swing']):,.0f}. Reported as a RESULT, not a failure "
              "(section 10 amendment: do not 'fix' the low-stratum tornado).")

    # ---- plan Decision 3: the counters must be EXACTLY zero, at EVERY stratum ---- #
    for sname, _ in DSA_STRATA:
        for name in COUNTER_ROWS:
            s = float(frames[sname][frames[sname]["parameter"] == name].iloc[0]["inmb_swing"])
            if s != 0.0:
                raise AssertionError(
                    f"plan Decision 3 VIOLATION: counter {name!r} moved INMB by ${s:,.6f} "
                    f"at the {sname!r} stratum. Counters carry NO incremental cost -- if "
                    "one now does, the model is double-counting against Edmiston's "
                    "all-cause increment."
                )
    print(f"[OK] plan Decision 3: {len(COUNTER_ROWS)} counter/cross-check rows "
          f"({', '.join(COUNTER_ROWS)}) all swing EXACTLY $0 at all "
          f"{len(DSA_STRATA)} strata.")

    # ---- the tunnel rows must NOT be zero (proof the kwargs were used) ---- #
    for sname, _ in DSA_STRATA:
        for name in ("n_tunnel_sup", "n_tunnel_deep"):
            s = float(frames[sname][frames[sname]["parameter"] == name].iloc[0]["inmb_swing"])
            if not s > 0.0:
                raise AssertionError(
                    f"{name!r} produced a swing of exactly ${s:,.6f} at the {sname!r} "
                    "stratum, which means the tunnel width never reached the engine. "
                    "Tunnel widths are a LAYOUT: pass n_sup= / n_deep= to "
                    "run_arm/run_pair, never a `sample` key."
                )
    n_sup_s = float(df[df["parameter"] == "n_tunnel_sup"].iloc[0]["inmb_swing"])
    n_deep_s = float(df[df["parameter"] == "n_tunnel_deep"].iloc[0]["inmb_swing"])
    print(f"[OK] tunnel widths swept via the n_sup= / n_deep= kwargs: non-zero swings "
          f"n_tunnel_sup ${n_sup_s:,.2f}, n_tunnel_deep ${n_deep_s:,.2f}.")

    print()
    print("The analysis axis (p_base_ssi) is reported in dsa_summary.md and is "
          "DELIBERATELY absent from both CSVs:")
    for a in axis_rows:
        print(f"    {a['stratum']:<14s} p={a['p']:.3f}  INMB {a['inmb']:>10,.0f}  "
              f"{a['dominance']}")
    print()
    print("Files written to:", RESULTS_DIR)
    expected = ["dsa_results.csv", "dsa_tornado.csv", "dsa_summary.md"]
    missing = []
    for fn in expected:
        ok = os.path.exists(os.path.join(RESULTS_DIR, fn))
        if not ok:
            missing.append(fn)
        print(f"  [{'OK' if ok else 'MISSING'}] {fn}")
    if missing:
        raise RuntimeError(f"run_dsa.py failed to write: {missing}")
    assert p_results and p_tornado and p_summary


if __name__ == "__main__":
    main()
