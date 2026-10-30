"""
run_scenarios.py -- Defined scenario analyses and budget impact for the
SSI / ciNPWT risk-threshold Markov cost-utility analysis.

Imports the engine (`inputs.py` / `model.py`) READ-ONLY and never mutates the parameter
store: every scenario is expressed as a fresh `sample` dict (`P(**overrides)`) plus a
fresh run-kwargs dict (`S(**overrides)`), both copies. A snapshot of every
`inputs.all_params()` value is taken at import and re-checked at the end of `main()`, so
an accidental in-place write fails loudly instead of silently contaminating later rows.

Writes the three BUILD-CONTRACT section-8 deliverables to `v2/results/`:

    scenarios_results.csv   one row per scenario x risk stratum, plus explicit `not_run`
                            rows for the two defined analyses that cannot be run
    budget_impact.csv       500- and 1,000-case programs x proportion treated x stratum
    scenarios_summary.md    results banner, the scenario/decision table, the
                            decision-stability statement, and the narrative

Every CSV carries `inputs.RESULTS_BANNER` as a '#'-prefixed LINE 1 followed by the header
row, so downstream readers use `pd.read_csv(path, skiprows=1)` (BUILD-CONTRACT section 7).

Run with cwd = v2/analysis/ :  python3 run_scenarios.py

-----------------------------------------------------------------------------
THE FOUR THINGS THAT ARE EASY TO GET WRONG HERE
-----------------------------------------------------------------------------
1. **The 1-year horizon charges tranche 2.** Onset sits at cycles 1-3, so the three
   marginal Edmiston tranches land at cycles 1-3, 7-9 and 13-15. At a 12-cycle horizon
   tranche 2 (cycles 7-9) is comfortably INSIDE the horizon and IS charged; only tranche
   3 is not. Dropping it would silently remove $15,109 (deep) / $6,137 (superficial) in
   2019 USD from every 1-year row and look like a plausible horizon effect. This module
   never hand-rolls that selection -- it drives off `model.tranches_charged()` and
   `model.ssi_cost_per_case(horizon_cycles=T)`, which RAISES on a clipped horizon.

2. **The 5-year row is IDENTICAL to the 2-year row in the base case, by construction.**
   The SSI hazard is zero after 90 days, all three cost tranches are charged by cycle 15,
   and both tunnels have drained into RECOV by cycle 16; from there on both arms occupy
   states of identical utility (RECOV = WELL in the base case) and identical alive mass,
   so neither the incremental cost nor the incremental QALY can move. It is asserted
   below rather than asserted in prose. The 5-year horizon only becomes informative under
   the residual-decrement scenario, where RECOV utility differs from WELL -- which is why
   `horizon_5y_residual_decrement` is a scenario in its own right.

3. **Tunnel duration is a LAYOUT, not a `sample` key.** `sample={"n_tunnel_deep": 24}`
   RAISES in the engine (it would otherwise be silently ignored and report a swing of
   exactly zero). Tunnel scenarios pass `n_sup=` / `n_deep=` to `run_arm`/`run_pair`.

4. **The standard-dressing cost is an ARM-A toll.** Charging $50 makes the comparator
   more expensive and therefore moves the threshold in the OPPOSITE direction to the
   device price: it makes ciNPWT look better, not worse.

See: v2/BUILD-CONTRACT.md sections 2, 5, 6.2, 8, 10; v2/run_log.d/model.md;
v2/run_log.d/inputs.md; v2/run_log.d/run_scenarios.md (this module's own fragment).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402
from scipy.optimize import brentq                                    # noqa: E402

import inputs as I                                                   # noqa: E402  engine, read-only
import model                                                         # noqa: E402  engine, read-only

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V1_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
RESULTS_DIR = os.path.join(V1_DIR, "results")

STRATA: dict = dict(I.SETTINGS["strata"])          # low 0.015 / intermediate 0.066 / high 0.114
WTP = float(I.SETTINGS["wtp_primary"])             # 100,000
WTP_LO, WTP_HI = (float(w) for w in I.SETTINGS["wtp_report"])        # 50,000 / 150,000
BASE_HORIZON = int(I.SETTINGS["horizon_cycles"])   # 24 cycles = 2 years

#: Budget-impact programme sizes and the proportion-treated axis (plan Phase D, Table 5).
PROGRAM_SIZES = (500, 1000)
PROPORTIONS_TREATED = (0.0, 0.25, 0.50, 0.75, 1.00)

#: Horizons used by the horizon family, in cycles. All four are "clean" -- no cost
#: tranche is clipped mid-window -- which is what makes `ssi_cost_per_case(horizon_cycles=)`
#: safe to call on them (it raises otherwise).
HORIZON_LABELS = {3: "90 d", 6: "6 mo", 12: "1 y", 24: "2 y (base case)", 60: "5 y"}


# --------------------------------------------------------------------------- #
# 0. Guard: the parameter store must never be mutated in place                 #
# --------------------------------------------------------------------------- #
_PARAM_SNAPSHOT = {p.name: float(p.value) for p in I.all_params()}


def _assert_store_unmutated() -> None:
    """Fail loudly if any `inputs.Param.value` moved during this run.

    Scenario panels are the classic place an analysis quietly corrupts itself: one
    `inputs.cost.m_ssi_cost.value = 1.000` and every subsequent row -- including the base
    case -- is silently wrong. Every scenario here is a COPY (`P()` / `S()`), and this
    check proves it.
    """
    drift = {
        name: (before, float(I.param_by_name(name).value))
        for name, before in _PARAM_SNAPSHOT.items()
        if float(I.param_by_name(name).value) != before
    }
    if drift:
        raise RuntimeError(
            "inputs.py parameter values were MUTATED IN PLACE during run_scenarios.py "
            f"(scenarios must use the P()/S() copy helpers): {drift}"
        )


# --------------------------------------------------------------------------- #
# 1. Copy helpers -- P() builds a `sample`, S() builds run kwargs              #
#    NEITHER ever writes to `inputs`.                                          #
# --------------------------------------------------------------------------- #
#: Every `run_arm` keyword this module is allowed to vary. Anything else is a typo, and a
#: typo in a scenario panel produces a row that silently equals the base case.
_ALLOWED_RUN_KW = frozenset({
    "horizon_cycles", "device_cost", "rr", "half_cycle",
    "disc_cost", "disc_qaly", "n_sup", "n_deep",
})

#: The base-case run configuration, read from `inputs.SETTINGS` (BUILD-CONTRACT section 9:
#: only inputs.py may own a number). `None` means "let the engine read the `inputs.py`
#: value through val()/cost_2025(), so a PSA draw could still override it".
BASE_RUN_KW = {
    "horizon_cycles": BASE_HORIZON,
    "device_cost": None,            # -> cost.c_device_cinpwt ($250, 2025 USD, arm B only)
    "rr": None,                     # -> tp.rr_cinpwt (0.61)
    "half_cycle": bool(I.SETTINGS["half_cycle"]),
    "disc_cost": float(I.SETTINGS["disc_cost"]),
    "disc_qaly": float(I.SETTINGS["disc_qaly"]),
    "n_sup": None,                  # -> inputs.SUP_LEN  (3)
    "n_deep": None,                 # -> inputs.DEEP_LEN (12)
}


def S(**overrides) -> dict:
    """A FRESH copy of the base-case run kwargs with `overrides` applied.

    Mirrors the mirct `S(**overrides)` idiom. Returns a new dict every call, so a
    scenario can never reach into another scenario's configuration. Unknown keys raise --
    a mis-spelled kwarg would otherwise be dropped by `**kw` forwarding and the row would
    silently reproduce the base case.
    """
    bad = sorted(set(overrides) - _ALLOWED_RUN_KW)
    if bad:
        raise KeyError(
            f"S(): unknown run kwarg(s) {bad}; allowed = {sorted(_ALLOWED_RUN_KW)}. "
            "Tunnel widths are `n_sup` / `n_deep` (a LAYOUT), never a sampled Param."
        )
    out = dict(BASE_RUN_KW)
    out.update(overrides)
    return out


def P(**overrides) -> dict:
    """A FRESH `sample` dict (`{Param.name: float}`) carrying `overrides` only.

    Mirrors the mirct `P(**overrides)` idiom, adapted to this build's `Param`-object
    store: the engine's override hook is a `sample` dict rather than a parameter
    dictionary, and every name absent from the sample falls through to `Param.value`. So
    `P()` is the base case and `P(m_ssi_cost=1.0)` is the all-cause upper bound --
    **without touching `inputs.cost.m_ssi_cost`**.

    Two guards, both of which catch a scenario that would otherwise report a swing of
    exactly zero and look like a genuine null result:

    * an unknown parameter name RAISES (the engine only warns, because an over-broad but
      correctly-spelled sample is legitimate for PSA; a hand-written scenario panel has no
      such excuse);
    * `n_tunnel_sup` / `n_tunnel_deep` RAISE with a pointer to `S(n_sup=..., n_deep=...)`
      -- tunnel widths are a layout, and the engine refuses them in `sample` for exactly
      this reason.

    COST VALUES GO IN IN THEIR OWN STORED PRICE YEAR. The engine inflates exactly once
    inside `inputs.cost_2025()`. Never pre-inflate, and never pass a `val()`-read cost.
    """
    for name in overrides:
        if name in ("n_tunnel_sup", "n_tunnel_deep"):
            raise KeyError(
                f"P(): {name!r} is a LAYOUT, not a sampled parameter -- the engine raises "
                "on it because it would otherwise be silently ignored. Use "
                "S(n_sup=..., n_deep=...) instead."
            )
        if name not in model.KNOWN_PARAM_NAMES:
            raise KeyError(
                f"P(): {name!r} is not an inputs.all_params() parameter name; a scenario "
                "keyed on a name the engine never reads would report a swing of exactly "
                "zero. Check the spelling against [p.name for p in inputs.all_params()]."
            )
    return {str(k): float(v) for k, v in overrides.items()}


# --------------------------------------------------------------------------- #
# 2. Scenario constants, all read from inputs.py (no literal parameter values) #
# --------------------------------------------------------------------------- #
K_ATTRIB = float(I.cost.k_attrib_pennington)          # A2: 0.492 (was 0.427) -- now the BASE case
RR_SPINE = float(I.tp.rr_cinpwt_spine)                # A2: 0.434 (OR 0.399 converted to RR)  Chen 2022 [8]
RR_ATTEN = float(I.tp.rr_cinpwt_attenuated)           # 0.85   scenario constant
FRAC_DEEP_LO = float(I.tp.frac_deep.low)              # 0.30
FRAC_DEEP_HI = float(I.tp.frac_deep.high)             # 0.80
DU_RESIDUAL = float(I.util.du_recovered_residual.high)  # 0.02  McFarland 2023 [19]
DRESSING_HI = float(I.cost.c_dressing_standard.high)  # $50 (2025 USD -- NOT inflated)
N_SUP_SHORT, N_DEEP_SHORT = int(I.tp.n_tunnel_sup.low), int(I.tp.n_tunnel_deep.low)   # 2 / 6
N_SUP_LONG, N_DEEP_LONG = int(I.tp.n_tunnel_sup.high), int(I.tp.n_tunnel_deep.high)   # 6 / 24

#: Decolonization secondary analysis (plan, "Secondary analysis (decolonization)"):
#:     residual-risk multiplier = (1 - f_SA) + f_SA x OR_decol
#:                              = 0.54 + 0.46 x 0.56 = 0.7976
#: Recomputed here from the two source Params and cross-checked against the stored
#: `tp.m_decol_residual` so the derivation is visible rather than trusted.
F_SA = float(I.tp.f_sa_ssi)                           # 0.46  Freire-Archer 2023 [17]
OR_DECOL = float(I.tp.or_decol)                       # 0.56  Joshi 2026 [2]
M_DECOL = (1.0 - F_SA) + F_SA * OR_DECOL              # 0.7976
assert abs(M_DECOL - float(I.tp.m_decol_residual)) < 5e-4, (
    f"decolonization residual multiplier recomputed as {M_DECOL:.6f} but "
    f"inputs.tp.m_decol_residual = {float(I.tp.m_decol_residual)}"
)

#: Medicare perspective. The engine reads the SIX COMMERCIAL tranche Params by name, so
#: the Medicare perspective is run by substituting the `*_mcr` VALUES onto the commercial
#: NAMES in a `sample`. That is price-year-safe **only** because both sets are stored in
#: the same price year (2019 USD) and `cost_2025()` inflates whatever value it is handed
#: using the *commercial* Param's `price_year` -- asserted below, so a future divergence
#: fails loudly instead of mixing 2019 and 2025 dollars.
_MEDICARE_SUBSTITUTION = {
    "c_ssi_sup_0_6mo": I.cost.c_ssi_sup_0_6mo_mcr,
    "c_ssi_sup_6_12mo": I.cost.c_ssi_sup_6_12mo_mcr,
    "c_ssi_sup_12_24mo": I.cost.c_ssi_sup_12_24mo_mcr,
    "c_ssi_deep_0_6mo": I.cost.c_ssi_deep_0_6mo_mcr,
    "c_ssi_deep_6_12mo": I.cost.c_ssi_deep_6_12mo_mcr,
    "c_ssi_deep_12_24mo": I.cost.c_ssi_deep_12_24mo_mcr,
}
for _commercial_name, _mcr_param in _MEDICARE_SUBSTITUTION.items():
    _commercial_param = I.param_by_name(_commercial_name)
    assert int(_commercial_param.price_year) == int(_mcr_param.price_year), (
        f"Medicare substitution is price-year-unsafe: {_commercial_name} is "
        f"{_commercial_param.price_year} USD but {_mcr_param.name} is "
        f"{_mcr_param.price_year} USD"
    )
MEDICARE_SAMPLE = {k: float(v.value) for k, v in _MEDICARE_SUBSTITUTION.items()}

#: A2 2026-08-17: this is now EMPTY, and that is the point. This build asserted that
#: Edmiston reports the Medicare SUPERFICIAL payment at 6 months only and extrapolated the
#: 12- and 24-month tranches. Full-text verification found all three reported -- $11,044,
#: $17,967 and $24,096 -- in the Abstract, the Results, Table 3 and the Discussion. Both
#: tranches were corrected ($3,259 -> $6,923 and $3,124 -> $6,129, each roughly 2x) and
#: their evidence_status moved off `derived_assumption`, so nothing matches this filter
#: any more. The filter is retained so the Medicare notes stay derived from the parameter
#: store rather than from a claim in prose.
MEDICARE_EXTRAPOLATED = tuple(
    v.name for v in _MEDICARE_SUBSTITUTION.values()
    if v.evidence_status == "derived_assumption"
)

#: Uniform onset weights -- the structural alternative `inputs.py` defines for the
#: median-matched Edmiston weights ("at 3%/yr the choice is immaterial"; tested, not assumed).
UNIFORM_ONSET = {name: 1.0 / 3.0 for name in
                 ("w_onset_deep_m1", "w_onset_deep_m2", "w_onset_deep_m3",
                  "w_onset_sup_m1", "w_onset_sup_m2", "w_onset_sup_m3")}


# --------------------------------------------------------------------------- #
# 3. The scenario panel                                                        #
#    (key, description, sample, run_kw, note, p_transform)                     #
#    `p_transform` is None (= run at the stratum's own p) or a callable p -> p  #
#    for a scenario that is itself a RISK statement (decolonization).           #
# --------------------------------------------------------------------------- #
SCENARIOS = [
    # ---- 1. Base case (reference row) -------------------------------------- #
    ("basecase",
     f"Base case: 2 y (24 cycles), 3%/yr discounting, commercial payer, RR 0.61, "
     f"device $250, deep fraction {float(I.tp.frac_deep):.3f}, SSI cost at the matched "
     f"attributable anchor (m_ssi_cost = {float(I.cost.m_ssi_cost):.3f}), full recovery",
     P(), S(),
     "Reference row. Every sign-flip flag in this file is relative to this row at the "
     "same risk stratum.",
     None),

    # ---- 2. Attribution-adjusted SSI cost ---------------------------------- #
    ("allcause_edmiston",
     "All-cause SSI cost (upper bound): m_ssi_cost = 1.000, i.e. Edmiston's "
     "regression-adjusted ALL-CAUSE incremental payment charged at face value",
     P(m_ssi_cost=1.0), S(),
     "**A2 INVERTED THIS ROW (2026-08-17).** It used to be the attribution-adjusted "
     "SENSITIVITY, run at m_ssi_cost = k_attrib_pennington while the base case charged "
     "Edmiston's all-cause increment. Verification of Edmiston's full text established "
     "that its payments are least-squares means from a GLM with log link and gamma error, "
     "covariate-adjusted only, over ALL-CAUSE total payments, with NO MATCHING anywhere "
     "in the paper -- an upper bound on SSI-attributable cost by construction, because "
     "the residual gap is confounding by indication. The matched anchor therefore became "
     "the BASE CASE, and this row now carries the all-cause UPPER BOUND. Keeping it at "
     "the old value would have made it a numerical duplicate of the base case. It scales "
     "all six marginal tranches together and remains the single largest swing in the "
     "model; because the averted cost is larger here, ciNPWT appears to pay off at a "
     "LOWER baseline risk -- which is exactly why it is a bound and not the headline.",
     None),

    # ---- 2b. Effect confined to SUPERFICIAL SSI (A2, from Zwanenburg's own data) -- #
    ("effect_superficial_only",
     "ciNPWT prevents SUPERFICIAL SSI only: no benefit on deep SSI "
     "(Zwanenburg 2020 [11] RCT subgroup)",
     P(c_ssi_deep_0_6mo=0.0, c_ssi_deep_6_12mo=0.0, c_ssi_deep_12_24mo=0.0, du_deep=0.0),
     S(),
     "**ADDED BY THE A2 AUDIT, AND IT IS THE MOST ADVERSE SCENARIO IN THIS FILE.** "
     "Zwanenburg 2020 [11] -- the source of the base-case RR 0.61 -- reports NO "
     "significant benefit for deep SSI in its randomized trials, and its authors write "
     "that 'iNPWT efficacy may potentially be limited to superficial SSIs only, and such "
     "SSIs are of less clinical relevance than deep SSIs.' This model draws the large "
     "majority of its saving from averted DEEP SSI, so that subgroup result is a "
     "first-order threat to its logic and is tested here rather than left in prose.\n"
     "HOW IT IS IMPLEMENTED, AND WHY THIS IS EXACT RATHER THAN AN APPROXIMATION. The "
     "engine applies RR to the TOTAL SSI risk and splits the result by frac_deep / "
     "frac_sup, so it cannot be told to spare deep SSI directly. But INMB is a "
     "DIFFERENCE between arms. If ciNPWT does not prevent deep SSI then deep cases are "
     "equinumerous in both arms and every deep cost and every deep QALY loss CANCELS from "
     "the incremental. Setting the three deep cost tranches and du_deep to zero in BOTH "
     "arms makes them cancel too, and leaves the superficial contribution untouched: "
     "superficial cases averted are p x frac_sup x (1 - RR) either way. The two "
     "formulations therefore give a NEARLY identical INMB. Measured against the exact "
     "counterfactual (arm B run at RR_eff = RR x f_sup + f_deep with the split "
     "re-weighted so the deep hazard is unchanged and the superficial hazard is scaled "
     "by RR), this emulation agrees to within 1.3% and ERRS AGAINST ciNPWT's critics -- "
     "i.e. the exact counterfactual is slightly MORE adverse at every stratum "
     "(-$209.26/-$115.57/-$17.79 exact vs -$209.22/-$115.43/-$17.57 here). The residual "
     "gap is competing risk: the two depths draw from the same Well pool over the "
     "three-cycle onset window, so sparing deep SSI in arm B leaves marginally less mass "
     "available to generate superficial onsets than this emulation retains. Exactness "
     "would additionally require du_recovered_residual = 0 (it is 0.00 in the base case "
     "but 0.06 in the residual-decrement scenario, and the two tunnels differ in length, "
     "so arrival into RECOVERED is depth-dependent) and depth-independent mortality "
     "(which does hold here). The sign is negative at every stratum under BOTH "
     "formulations, so the reversal finding is safe. This row also distorts each arm's "
     "absolute cost and QALY LEVELS, and its nnt / ssi_prevented_per_1000 / "
     "reop_prevented_per_1000 columns are the BASE-CASE counts because the trace is "
     "untouched -- only the incremental columns of this row may be quoted.",
     None),

    # ---- 3. Medicare perspective ------------------------------------------- #
    ("medicare",
     "Medicare perspective: the six *_mcr cost tranches replace the commercial set "
     "(2 y horizon)",
     P(**MEDICARE_SAMPLE), S(),
     "**A2 CORRECTED THIS ROW (2026-08-17) AND RETIRED ITS FOOTNOTE.** This build stated "
     "that Edmiston reports the Medicare SUPERFICIAL payment at 6 months only, and "
     "extrapolated the 12- and 24-month tranches by transferring the commercial growth "
     "ratio onto the 6-month value ($3,259 and $3,124). All three Medicare superficial "
     "payments are reported -- $11,044, $17,967 and $24,096 -- in four places, so the "
     "correct marginal tranches are $6,923 and $6,129, each roughly TWICE what was used. "
     "The extrapolation also failed in principle: Medicare payments grow FASTER than "
     "commercial from 6 to 12 months (1.627 vs 1.295), so a transferred commercial ratio "
     "understates them. **Every dollar in this row is now a reported Edmiston value**, "
     "the four-tranche identity 11,044 + 6,923 + 6,129 = 24,096 closes exactly, and no "
     "extrapolation footnote is required.",
     None),
    ("medicare_6mo_horizon",
     "Medicare perspective, 6-month horizon (6 cycles) -- a horizon sensitivity",
     P(**MEDICARE_SAMPLE), S(horizon_cycles=6),
     "**A2 CHANGED WHAT THIS ROW IS FOR (2026-08-17).** It was the clean fallback that "
     "avoided the extrapolated Medicare superficial tranches; A2 removed that rationale: the 12- and 24-month Medicare superficial tranches are reported, not "
     "extrapolated, so the 2-year Medicare row is now built entirely from reported values "
     "too and this row is no longer needed to avoid an assumption. It is retained as a "
     "genuine horizon sensitivity. At a 6-cycle horizon only tranche 1 (cycles 1-3) is "
     "charged; no tranche is clipped, so the horizon is 'clean' in the BUILD-CONTRACT 6.2 "
     "sense. It is a shorter horizon, not a like-for-like alternative to the 2-year "
     "Medicare row: read it as a floor.",
     None),

    # ---- 4-5. Relative effect ---------------------------------------------- #
    ("rr_spine",
     f"Spine-specific effect (favorable): RR = {RR_SPINE:.3f} (Chen 2022 [8])",
     P(), S(rr=RR_SPINE),
     "Chen pools 5 small retrospective spine studies (n=994) and reports an ODDS ratio, "
     "applied here as a risk ratio; at p of 0.5-15% the two differ by at most a few "
     "percent and the direction (an OR sits further from 1 than the RR) makes this "
     "favorable scenario slightly MORE favorable to ciNPWT. Recorded as an inference, "
     "not silently absorbed. The base case deliberately uses the weaker all-surgery "
     "RR 0.61, which raises the threshold and errs in the harder direction.",
     None),
    ("rr_attenuated",
     f"Attenuated effect: RR = {RR_ATTEN:.2f} (scenario constant)",
     P(), S(rr=RR_ATTEN),
     "Not an estimate -- a stress test of the assumption that ciNPWT's "
     "relative effect applies equally to the residual (relatively more gram-negative and "
     "polymicrobial) SSI left after universal decolonization. RR 0.85 sits above the "
     "upper 95% bound of the pooled Zwanenburg estimate (0.76).",
     None),

    # ---- 6. Deep fraction --------------------------------------------------- #
    ("deep_frac_030",
     f"Deep fraction {FRAC_DEEP_LO:.2f} (frac_sup = {1 - FRAC_DEEP_LO:.2f})",
     P(frac_deep=FRAC_DEEP_LO, frac_sup=1.0 - FRAC_DEEP_LO), S(),
     "frac_deep and frac_sup are a Dirichlet PAIR (group='ssi_split') and must be set "
     f"together -- the engine renormalizes, but leaving frac_sup at "
     f"{float(I.tp.frac_sup):.3f} would silently "
     "rescale the split. The wide 0.30-0.80 range exists because Edmiston's 74%-deep "
     "claims composition and Freire-Archer's 1.5% pooled deep rate [17] are hard to "
     "reconcile; claims coding likely over-captures 'deep'. Deep SSI costs ~2.9x a "
     "superficial one, so this moves the cost per SSI substantially.",
     None),
    ("deep_frac_080",
     f"Deep fraction {FRAC_DEEP_HI:.2f} (frac_sup = {1 - FRAC_DEEP_HI:.2f})",
     P(frac_deep=FRAC_DEEP_HI, frac_sup=1.0 - FRAC_DEEP_HI), S(),
     "Upper bound of the deep-fraction range; the Dirichlet partner frac_sup is set to "
     "1 - frac_deep. Favourable to ciNPWT because it raises the cost per SSI averted.",
     None),

    # ---- 7. Horizons (BUILD-CONTRACT 6.2 -- read the toll table) ------------ #
    ("horizon_90d",
     "Horizon 90 days (3 cycles)",
     P(), S(horizon_cycles=3),
     "ONLY tranche 1 (cycles 1-3) is charged; tranches 2 (cycles 7-9) and 3 (cycles "
     "13-15) fall beyond the horizon. This is deliberate truncation of COST accrual, "
     "not a change in the hazard -- the SSI hazard is already zero from day 91 in every "
     "column of this family. ciNPWT necessarily looks least attractive here.",
     None),
    ("horizon_1y",
     "Horizon 1 year (12 cycles)",
     P(), S(horizon_cycles=12),
     "THE TRAP ROW. Tranche 2 lands at cycles 7-9, comfortably INSIDE a 12-cycle "
     "horizon, and IS charged; only tranche 3 (cycles 13-15) is not. Dropping tranche 2 "
     "would silently remove $15,109 (deep) / $6,137 (superficial) in 2019 USD per SSI "
     "and would look like a plausible horizon effect rather than a bug. Verified against "
     "model.tranches_charged(12), which reports tranche 2 as 'full'.",
     None),
    ("horizon_2y_base",
     "Horizon 2 years (24 cycles) -- the base horizon, repeated so the horizon family is "
     "complete",
     P(), S(horizon_cycles=BASE_HORIZON),
     "Identical to `basecase` by construction; included so the horizon family can be "
     "read as one panel without cross-referencing another row. All three tranches are "
     "charged.",
     None),
    ("horizon_5y",
     "Horizon 5 years (60 cycles)",
     P(), S(horizon_cycles=60),
     "IDENTICAL TO THE 2-YEAR ROW, AND CORRECTLY SO -- asserted in this script, not "
     "assumed. The SSI hazard is zero after 90 days, all three cost tranches are charged "
     "by cycle 15, and both tunnels have drained into RECOV by cycle 16; from there both "
     "arms occupy states of identical utility (RECOV = WELL in the base case) with "
     "identical alive mass, so neither increment can move. The 5-year horizon becomes "
     "informative ONLY under the residual-decrement scenario -- see "
     "`horizon_5y_residual_decrement`.",
     None),

    # ---- 8. Discount rate ---------------------------------------------------- #
    ("discount_0pct",
     "Discount rate 0%/yr on costs and QALYs",
     P(), S(disc_cost=0.0, disc_qaly=0.0),
     "Undiscounted. Favours ciNPWT slightly: the device toll falls at cycle 0 and is "
     "undiscounted in every scenario, while the SSI cost it averts accrues over months "
     "1-15 and is therefore discounted in the base case.",
     None),
    ("discount_5pct",
     "Discount rate 5%/yr on costs and QALYs",
     P(), S(disc_cost=0.05, disc_qaly=0.05),
     "Higher discount rate. Immaterial over a 2-year horizon -- the whole model runs "
     "inside 24 monthly cycles -- which is itself worth reporting.",
     None),

    # ---- 9. Residual decrement in RECOV -------------------------------------- #
    ("residual_decrement",
     f"Residual decrement: du_recovered_residual = {DU_RESIDUAL:.2f} in RECOV to the "
     "horizon (2 y)",
     P(du_recovered_residual=DU_RESIDUAL), S(),
     "A GENUINE EVIDENCE CONFLICT, which is why it is a scenario and not a DSA row. "
     "Urquhart 2021 [21] reports PROM recovery by 1 year (the base case: RECOV = WELL); "
     "McFarland 2023 [19] reports a persisting decrement, and Freire-Archer [17] reports "
     "up-to-25% persistent infection. This conflict is the only reason RECOV is kept as "
     "a state distinct from WELL. Arm A sends more patients through the tunnels into "
     "RECOV, so a residual decrement penalises the comparator and favours ciNPWT -- and "
     "the effect GROWS with the horizon.",
     None),
    ("horizon_5y_residual_decrement",
     f"Horizon 5 y (60 cycles) x residual decrement {DU_RESIDUAL:.2f} -- the interaction",
     P(du_recovered_residual=DU_RESIDUAL), S(horizon_cycles=60),
     "THE ONLY WAY THE 5-YEAR HORIZON IS INFORMATIVE. With RECOV = WELL the 5-year row "
     "is an exact copy of the 2-year row; with a residual decrement in RECOV the two "
     "arms' utilities differ for every cycle after the tunnels drain, so the incremental "
     "QALY keeps growing for the whole 60 cycles. Reported explicitly so the 5-year row "
     "is not read as redundant copy-paste.",
     None),

    # ---- 10. Tunnel duration (a LAYOUT change, not a sample) ----------------- #
    ("tunnel_short",
     f"Short tunnels: superficial {N_SUP_SHORT} cycles, deep {N_DEEP_SHORT} cycles "
     f"({1 + N_SUP_SHORT + N_DEEP_SHORT + 2} slots)",
     P(), S(n_sup=N_SUP_SHORT, n_deep=N_DEEP_SHORT),
     "The Urquhart-vs-McFarland conflict expressed STRUCTURALLY rather than through a "
     "utility value. Passed as n_sup/n_deep kwargs, never as a sample -- the engine "
     "RAISES on n_tunnel_* in a sample because it would otherwise be silently ignored "
     "and report a swing of exactly zero. Tunnel width moves the QALY side and ONLY the "
     "QALY side: the cost tolls are attached to the onset cohort, not to tunnel "
     "occupancy, so the incremental cost is unchanged. The 90-day onset window remains "
     "3 cycles at every width.",
     None),
    ("tunnel_long",
     f"Long tunnels: superficial {N_SUP_LONG} cycles, deep {N_DEEP_LONG} cycles "
     f"({1 + N_SUP_LONG + N_DEEP_LONG + 2} slots)",
     P(), S(n_sup=N_SUP_LONG, n_deep=N_DEEP_LONG),
     "McFarland-leaning structural variant (deep decrement carried for 24 months). Same "
     "layout caveats as `tunnel_short`; traces of differing widths are never compared "
     "without collapsing to the 5 CLINICAL_STATES first, and this module reports only "
     "scalars, so no trace of a non-base width is written anywhere.",
     None),

    # ---- 11. Standard-dressing differential cost ----------------------------- #
    ("dressing_50",
     f"Standard-dressing differential ${DRESSING_HI:,.0f} in arm A (base case $0)",
     P(c_dressing_standard=DRESSING_HI), S(),
     "MOVES THE THRESHOLD IN THE OPPOSITE DIRECTION TO THE DEVICE PRICE. The dressing "
     "toll is charged at cycle 0 in ARM A (the comparator), so charging $50 makes "
     "standard care more expensive and ciNPWT relatively cheaper: the incremental cost "
     "falls by exactly $50 and the INMB rises by exactly $50 at every risk stratum. The "
     "$0 base case is the conservative choice -- arm B pays the full device price with "
     "no offset. c_dressing_standard is stored in 2025 USD and is NOT inflated.",
     None),

    # ---- 12. Decolonization secondary analysis (a RISK statement) ------------ #
    ("decol_residual",
     f"Decolonization secondary analysis: residual-risk multiplier "
     f"(1 - f_SA) + f_SA x OR_decol = {1 - F_SA:.2f} + {F_SA:.2f} x {OR_DECOL:.2f} = "
     f"{M_DECOL:.3f}; the stratum p is treated as the PRE-decolonization risk",
     P(), S(),
     "Universal S. aureus decolonization is background in BOTH arms (SHEA/IDSA essential "
     "practice for implant procedures), so its cost and its effect CANCEL from every "
     "base-case increment. It survives only here, as one transparent line: it lowers the "
     "risk each patient brings to the ciNPWT decision, and the reported question is how "
     "far that shifts patients BELOW the ciNPWT threshold. p shown is the RESIDUAL risk "
     "actually run (= stratum p x 0.798); p_stratum is the pre-decolonization risk. "
     "LIMITATION: ciNPWT's relative risk is assumed to apply equally to the residual SSI "
     "pool, which is relatively more gram-negative and polymicrobial after S. aureus is "
     "selectively removed. `decol_residual_attenuated` relaxes exactly that assumption.",
     lambda p: p * M_DECOL),
    ("decol_residual_attenuated",
     f"Decolonization (residual risk x {M_DECOL:.3f}) PLUS attenuated ciNPWT effect "
     f"RR = {RR_ATTEN:.2f} on the residual pool",
     P(), S(rr=RR_ATTEN),
     "The stated limitation of `decol_residual`, made a scenario: the residual SSI pool "
     "after decolonization is relatively more gram-negative and polymicrobial, and "
     "ciNPWT's pooled all-surgery RR may not transport to it. Both effects are applied "
     "together, which is the pessimistic corner of the secondary analysis.",
     lambda p: p * M_DECOL),

    # ---- Supplementary: structural alternative ----------------- #
    ("onset_uniform",
     "Uniform 90-day onset weights (1/3, 1/3, 1/3) instead of the Edmiston "
     "median-matched weights",
     P(**UNIFORM_ONSET), S(),
     "The structural alternative `inputs.py` defines for the onset weights ('at "
     "3%/yr the choice is immaterial'). Run so that claim is tested rather than "
     "asserted. The weights remain CONDITIONAL-probability inputs, so the cumulative "
     "90-day onset risk still equals p exactly; only the within-window timing moves, "
     "which shifts the cost tolls by at most a couple of cycles.",
     None),
]

#: Defined analyses that CANNOT be run, recorded explicitly rather than omitted.
NOT_RUN = [
    ("societal_perspective",
     "Societal perspective (productivity loss, patient/caregiver time, travel)",
     "NOT RUN -- no societal cost input exists anywhere in the parameter store. The "
     "declared perspective is the US commercial payer on a payments basis (Medicare is "
     "the alternative payer scenario above); adding productivity or caregiver costs "
     "would require inventing them. Direction if it were run is unambiguous: SSI carries "
     "substantial productivity loss that the payer perspective ignores, so a societal "
     "perspective would raise the cost per SSI averted and move the threshold DOWN, "
     "making ciNPWT look better. The payer perspective is therefore the conservative "
     "one. Documented as unrun and data-limited."),
    ("risk_targeted_arm",
     "'Risk-targeted ciNPWT' as a separately modelled third arm",
     "NOT RUN BY DESIGN -- and this is a plan decision, not a limitation of this script. "
     "A third arm that treats only patients above a risk threshold requires the "
     "POPULATION DISTRIBUTION of predicted 90-day SSI risk, which does not exist in the "
     "evidence base. Rather than invent one, risk-targeted ciNPWT is reported as a "
     "DERIVED result: the per-stratum rows in this file plus `budget_impact.csv`, which "
     "is parameterized by the proportion of the programme treated. That is "
     "MATHEMATICALLY IDENTICAL to a third arm whose targeting rule selects the same "
     "proportion at the same risk, and it avoids fabricating a distribution."),
]


# --------------------------------------------------------------------------- #
# 4. Running the panel                                                         #
# --------------------------------------------------------------------------- #
def _sign(x, tol: float = 1e-9) -> int:
    """Sign of a float with a dead zone, so |INMB| of a few cents is not a 'flip'."""
    if x is None or not np.isfinite(x) or abs(x) < tol:
        return 0
    return 1 if x > 0 else -1


def _scenario_rows(key, desc, sample, run_kw, note, p_transform, base_inmb) -> list[dict]:
    """One row per risk stratum for a single scenario.

    `base_inmb` maps stratum -> the base-case INMB at $100k, used only for the sign-flip
    flag. The flag is the decision-relevant field: the point estimates move constantly,
    but a change in the SIGN of INMB is what changes what a surgeon should do.
    """
    rows = []
    for sname, p_stratum in STRATA.items():
        p = float(p_stratum) if p_transform is None else float(p_transform(p_stratum))
        res_a, res_b = model.run_pair(p_base=p, sample=(sample or None), **run_kw)
        inc = model.incremental(res_b, res_a, wtp=WTP)
        inc_lo = model.incremental(res_b, res_a, wtp=WTP_LO)
        inc_hi = model.incremental(res_b, res_a, wtp=WTP_HI)

        d_ssi = res_a["n_ssi"] - res_b["n_ssi"]
        d_reop = res_a["n_reop"] - res_b["n_reop"]
        sgn = _sign(inc["inmb"])
        rows.append({
            "scenario": key,
            "description": desc,
            "stratum": sname,
            "p_stratum": float(p_stratum),
            "p": p,
            "horizon_cycles": int(res_b["horizon_cycles"]),
            "disc_rate": float(run_kw["disc_cost"]),
            "rr_used": float(res_b["rr_used"]),
            "device_cost_used": float(res_b["device_cost_used"]),
            "n_sup": int(I.SUP_LEN if run_kw["n_sup"] is None else run_kw["n_sup"]),
            "n_deep": int(I.DEEP_LEN if run_kw["n_deep"] is None else run_kw["n_deep"]),
            "cost_A": res_a["cost"],
            "cost_B": res_b["cost"],
            "qaly_A": res_a["qaly"],
            "qaly_B": res_b["qaly"],
            "inc_cost": inc["inc_cost"],
            "inc_qaly": inc["inc_qaly"],
            "icer": inc["icer"],                      # NaN under dominance -- see `dominance`
            "inmb_50k": inc_lo["inmb"],
            "inmb_100k": inc["inmb"],
            "inmb_150k": inc_hi["inmb"],
            "dominance": inc["dominance"],
            "quadrant": inc["quadrant"],
            "nnt": (1.0 / d_ssi) if d_ssi > 0 else np.nan,
            "ssi_prevented_per_1000": 1000.0 * d_ssi,
            "reop_prevented_per_1000": 1000.0 * d_reop,
            "inmb_sign": {1: "positive", -1: "negative", 0: "zero"}[sgn],
            # Two DIFFERENT decision changes, kept apart on purpose:
            #   flips_vs_basecase -- the sign of INMB moved (ciNPWT stops being worth
            #                        using at $100k/QALY). The decision-relevant one.
            #   cost_saving       -- ciNPWT is still cheaper in absolute terms. Losing
            #                        this while INMB stays positive means ciNPWT remains
            #                        COST-EFFECTIVE but is no longer COST-SAVING, which
            #                        is a different conversation with a payer.
            "cost_saving": "Y" if inc["inc_cost"] < 0 else "N",
            # Sentinels are chosen to SURVIVE `pd.read_csv` -- "n/a" is in pandas'
            # default NA list and would come back as NaN, silently turning a
            # deliberately-labelled cell into missing data for make_tables.py.
            "flips_vs_basecase": ("reference" if key == "basecase"
                                  else ("Y" if sgn != _sign(base_inmb[sname]) else "N")),
            "note": note,
        })
    return rows


def build_scenarios():
    """Run the whole panel and return (DataFrame, base-case INMB by stratum)."""
    # The base case first: every flip flag is relative to it.
    base_key, base_desc, base_sample, base_kw, base_note, base_tf = SCENARIOS[0]
    assert base_key == "basecase", "the base case must be the first panel entry"
    seed_rows = _scenario_rows(base_key, base_desc, base_sample, base_kw, base_note,
                               base_tf, {s: 0.0 for s in STRATA})
    base_inmb = {r["stratum"]: r["inmb_100k"] for r in seed_rows}

    rows = _scenario_rows(base_key, base_desc, base_sample, base_kw, base_note, base_tf,
                          base_inmb)
    for key, desc, sample, run_kw, note, p_tf in SCENARIOS[1:]:
        rows.extend(_scenario_rows(key, desc, sample, run_kw, note, p_tf, base_inmb))

    for key, desc, note in NOT_RUN:
        rows.append({
            "scenario": key, "description": desc, "stratum": "all",
            "p_stratum": np.nan, "p": np.nan, "horizon_cycles": pd.NA,
            "disc_rate": np.nan, "rr_used": np.nan, "device_cost_used": np.nan,
            "n_sup": pd.NA, "n_deep": pd.NA,
            "cost_A": np.nan, "cost_B": np.nan, "qaly_A": np.nan, "qaly_B": np.nan,
            "inc_cost": np.nan, "inc_qaly": np.nan, "icer": np.nan,
            "inmb_50k": np.nan, "inmb_100k": np.nan, "inmb_150k": np.nan,
            "dominance": "not_run", "quadrant": "not_run",
            "nnt": np.nan, "ssi_prevented_per_1000": np.nan,
            "reop_prevented_per_1000": np.nan,
            "inmb_sign": "not_run", "cost_saving": "not_run",
            "flips_vs_basecase": "not_run", "note": note,
        })

    cols = ["scenario", "description", "stratum", "p_stratum", "p", "horizon_cycles",
            "disc_rate", "rr_used", "device_cost_used", "n_sup", "n_deep",
            "cost_A", "cost_B", "qaly_A", "qaly_B", "inc_cost", "inc_qaly", "icer",
            "inmb_50k", "inmb_100k", "inmb_150k", "dominance", "quadrant", "nnt",
            "ssi_prevented_per_1000", "reop_prevented_per_1000", "inmb_sign",
            "cost_saving", "flips_vs_basecase", "note"]
    return pd.DataFrame(rows, columns=cols), base_inmb


# --------------------------------------------------------------------------- #
# 5. The 5-year / 2-year structural identity -- asserted, not asserted in prose #
# --------------------------------------------------------------------------- #
def check_five_year_identity(df: pd.DataFrame) -> dict:
    """The base-case 5-year result MUST equal the 2-year result exactly.

    Not a coincidence and not a copy-paste bug: the SSI hazard is zero after 90 days, all
    three cost tranches are charged by cycle 15, both tunnels drain into RECOV by cycle
    16, and from there both arms hold identical alive mass in states of identical utility
    (RECOV = WELL in the base case). Neither increment can move. Checked here so the
    claim in `scenarios_summary.md` is verified rather than asserted, and so that a future
    engine change that breaks it fails loudly.

    Returns the max absolute deviation in inc_cost / inc_qaly / INMB across strata.
    """
    two = df[df["scenario"] == "horizon_2y_base"].set_index("stratum")
    five = df[df["scenario"] == "horizon_5y"].set_index("stratum")
    dev = {
        "inc_cost": float(np.max(np.abs(five["inc_cost"] - two["inc_cost"]))),
        "inc_qaly": float(np.max(np.abs(five["inc_qaly"] - two["inc_qaly"]))),
        "inmb_100k": float(np.max(np.abs(five["inmb_100k"] - two["inmb_100k"]))),
    }
    assert dev["inc_cost"] < 1e-8 and dev["inmb_100k"] < 1e-6, (
        "the base-case 5-year result is NOT identical to the 2-year result "
        f"(max deviations {dev}); either a cost toll now falls after cycle 24, a tunnel "
        "no longer drains by cycle 24, or RECOV no longer shares WELL's utility"
    )
    return dev


# --------------------------------------------------------------------------- #
# 6. Threshold solver (local; run_threshold.py is the authoritative module)     #
# --------------------------------------------------------------------------- #
def _inmb_at(p, sample=None, wtp=WTP, **run_kw) -> float:
    res_a, res_b = model.run_pair(p_base=float(p), sample=(sample or None), **run_kw)
    return model.incremental(res_b, res_a, wtp=wtp)["inmb"]


def solve_threshold(sample=None, wtp=WTP, run_kw=None, lo=1e-6, hi=0.5) -> float:
    """Baseline 90-day SSI risk `p` at which INMB = 0 (ciNPWT breaks even).

    Solved with `brentq`; INMB is smooth and monotone increasing in `p` over the whole
    grid, so no bracketing tricks are needed. Returns NaN if the interval does not
    bracket a root (e.g. ciNPWT is cost-effective even at p -> 0).

    THIS IS A LOCAL SOLVE, and it is here only because the decolonization secondary
    analysis is defined in terms of the threshold ("how far does decolonization shift
    patients below it?"). `run_threshold.py` owns `threshold_results.csv` and is the
    authoritative source; it also reports the UNDISCOUNTED, no-half-cycle variant used for
    the BUILD-CONTRACT section-10 hand-calculation cross-check, which is a slightly
    different number from the discounted, half-cycle-corrected value solved here under
    the base-case REPORTING settings. Both are printed in the summary so they cannot be
    confused.
    """
    kw = dict(BASE_RUN_KW) if run_kw is None else dict(run_kw)
    f = lambda p: _inmb_at(p, sample=sample, wtp=wtp, **kw)     # noqa: E731
    try:
        f_lo, f_hi = f(lo), f(hi)
    except Exception:
        return float("nan")
    if not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_lo * f_hi > 0:
        return float("nan")
    return float(brentq(f, lo, hi, xtol=1e-14, rtol=8.9e-16, maxiter=200))


# --------------------------------------------------------------------------- #
# 7. Budget impact                                                             #
# --------------------------------------------------------------------------- #
def build_budget_impact(df: pd.DataFrame) -> pd.DataFrame:
    """500- and 1,000-case programmes x proportion treated x risk stratum.

    HOW 'RISK-TARGETED ciNPWT' IS REPORTED, AND WHY. The population distribution of
    predicted 90-day SSI risk does not exist in the evidence base, so a separately
    modelled risk-targeted third arm would require inventing one. Instead the plan
    reports risk-targeted use as a DERIVED result: per-stratum per-patient increments
    (from the base-case rows above) scaled by the number of patients treated. A programme
    that treats a proportion `f` of its cases at stratum risk `p` is mathematically
    identical to a third arm whose targeting rule selects exactly those patients -- with
    no fabricated distribution anywhere. Say that plainly; do not dress it up as a third
    arm.

    Every per-patient quantity is taken from the BASE CASE (24 cycles, 3%/yr discounting,
    2025 USD). Columns:

      inc_cost_total_usd            population incremental cost -- the actual budget line.
                                    NEGATIVE = the programme SAVES money.
      device_outlay_usd             gross ciNPWT acquisition spend (n_treated x unit cost).
      ssi_cost_averted_usd          SSI-attributable payments avoided; the offset that the
                                    device outlay is bought with. By construction
                                    inc_cost_total = device_outlay - dressing_offset
                                                     - ssi_cost_averted.
      net_budget_impact_100k_usd    population INMB = QALYs gained x $100,000
                                    - incremental cost. POSITIVE = good value.

    `c_decol_bundle` ($60/patient) is deliberately absent: universal decolonization is
    background in BOTH arms, so it cancels from the incremental and is not a ciNPWT
    budget line. A programme budget that needs the gross decolonization spend should add
    it as a separate, arm-independent item.
    """
    base = df[(df["scenario"] == "basecase")].set_index("stratum")
    dressing = I.cost_2025(I.cost.c_dressing_standard)

    rows = []
    for size in PROGRAM_SIZES:
        for sname, p in STRATA.items():
            r = base.loc[sname]
            device_unit = float(r["device_cost_used"])
            for frac in PROPORTIONS_TREATED:
                n = float(size) * float(frac)
                inc_cost_total = n * float(r["inc_cost"])
                device_outlay = n * device_unit
                # cost_A carries the arm-A dressing toll; cost_B carries the device toll.
                # Stripping both leaves the pure SSI-payment streams, whose difference is
                # the payments the programme avoids.
                ssi_averted = n * ((float(r["cost_A"]) - dressing)
                                   - (float(r["cost_B"]) - device_unit))
                qaly_gained = n * float(r["inc_qaly"])
                rows.append({
                    "program_size": int(size),
                    "stratum": sname,
                    "p_base": float(p),
                    "proportion_treated": float(frac),
                    "n_treated": n,
                    "horizon_cycles": int(r["horizon_cycles"]),
                    "wtp": WTP,
                    "inc_cost_per_patient": float(r["inc_cost"]),
                    "inc_qaly_per_patient": float(r["inc_qaly"]),
                    "inmb_per_patient_100k": float(r["inmb_100k"]),
                    "device_outlay_usd": device_outlay,
                    "ssi_cost_averted_usd": ssi_averted,
                    "inc_cost_total_usd": inc_cost_total,
                    "ssi_avoided": n * float(r["ssi_prevented_per_1000"]) / 1000.0,
                    "reop_avoided": n * float(r["reop_prevented_per_1000"]) / 1000.0,
                    "qaly_gained": qaly_gained,
                    "net_budget_impact_100k_usd": qaly_gained * WTP - inc_cost_total,
                    "note": (
                        "Derived from the base-case per-patient increment (24 cycles, "
                        f"{float(I.SETTINGS['disc_cost']):.0%}/yr discounting, "
                        f"{I.PRICE_YEAR} USD) scaled by n_treated. 'Risk-targeted ciNPWT' "
                        "is reported this way -- as a DERIVED result parameterized by the "
                        "proportion treated -- rather than as a separately modelled third "
                        "arm, because the population distribution of predicted 90-day SSI "
                        "risk is unavailable. The two are mathematically identical and "
                        "this avoids inventing a distribution. inc_cost_total_usd is the "
                        "budget line (negative = savings); net_budget_impact_100k_usd is "
                        "the population INMB at $100,000/QALY. The $60 decolonization "
                        "bundle is background in BOTH arms and cancels, so it is not a "
                        "ciNPWT budget line."
                    ),
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 8. I/O + formatting helpers                                                  #
# --------------------------------------------------------------------------- #
def write_results_csv(df: pd.DataFrame, filename: str) -> str:
    """Write a results CSV with `inputs.RESULTS_BANNER` as a '#'-prefixed line 1."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", newline="") as fh:
        fh.write("# " + I.RESULTS_BANNER + "\n")
        df.to_csv(fh, index=False)
    return path


def _money(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.0f}"


def _icer(x, dominance="") -> str:
    """Never print a bare NaN: under dominance the ratio is uninformative, so say so."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        if dominance == "ciNPWT dominant":
            return "n/a (dominant)"
        if dominance == "ciNPWT dominated":
            return "n/a (dominated)"
        if dominance == "not_run":
            return "not run"
        return "n/a"
    return f"${x:,.0f}/QALY"


def _pct(x, nd=3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{100.0 * x:.{nd}f}%"


# --------------------------------------------------------------------------- #
# 9. scenarios_summary.md                                                      #
# --------------------------------------------------------------------------- #
def build_summary_md(df, bi, five_year_dev, thresholds) -> str:
    L = []
    A = L.append

    A("# Scenario analyses and budget impact -- ciNPWT after instrumented posterior "
      "lumbar fusion")
    A("")
    A("> **" + I.RESULTS_BANNER + "**")
    A("")
    A("*Generated by `run_scenarios.py` from the read-only engine (`inputs.py` / "
      "`model.py`); no parameter value is owned by this script. Model-based result; not "
      "a substitute for clinical evidence.*")
    A("")
    A(f"Every scenario is run at all three reported risk strata -- low **p = "
      f"{STRATA['low']:.1%}** (Freire-Archer [17]), intermediate **p = "
      f"{STRATA['intermediate']:.1%}** (Edmiston [12]), high **p = {STRATA['high']:.1%}** "
      "(Mueller [6]) -- except where the scenario is itself a risk statement "
      "(decolonization), where the stratum value is treated as the *pre*-decolonization "
      "risk and the model is run at the residual. Increments are **`B_cinpwt` minus "
      f"`A_standard`**; INMB is at **${WTP:,.0f}/QALY** unless stated.")
    A("")

    # ---------------- decision stability: the headline ---------------------- #
    run_rows = df[df["dominance"] != "not_run"]
    flips = run_rows[run_rows["flips_vs_basecase"] == "Y"]
    neg = run_rows[run_rows["inmb_100k"] < 0]

    A("## Decision stability -- the question that matters")
    A("")
    A("Point estimates move in every scenario; what decides practice is whether the "
      "**sign of INMB** changes. In the base case ciNPWT is *dominant* (cheaper and "
      "better) at all three strata, so a sign flip means INMB turns **negative** -- "
      "ciNPWT stops being worth using at that risk level.")
    A("")
    if len(flips) == 0:
        A("**No scenario in the panel flips the sign of INMB at any risk stratum.** The "
          "decision is robust across the entire reported panel.")
    else:
        flip_scen = list(dict.fromkeys(flips["scenario"].tolist()))
        A(f"**{len(flips)} scenario x stratum cells flip the sign of INMB**, spread over "
          f"{len(flip_scen)} scenarios:")
        A("")
        A("| Scenario | Stratum | p | INMB @ $100k | Base-case INMB | Dominance |")
        A("|---|---|---:|---:|---:|---|")
        for _, r in flips.iterrows():
            b = df[(df["scenario"] == "basecase")
                   & (df["stratum"] == r["stratum"])].iloc[0]
            A(f"| `{r['scenario']}` | {r['stratum']} | {_pct(r['p'], 2)} "
              f"| **{_money(r['inmb_100k'])}** | {_money(b['inmb_100k'])} "
              f"| {r['dominance']} |")
        A("")
        by_stratum = flips.groupby("stratum").size().to_dict()
        A("Read by stratum: "
          + "; ".join(f"**{s}** {by_stratum.get(s, 0)} flip(s)" for s in STRATA)
          + ". The pattern is the substantive finding -- the decision is fragile only "
            "where the baseline risk is low, because the whole value of ciNPWT is the "
            "SSI cost it averts, and at low risk there is little to avert against a "
            "device toll that is charged to every patient.")
    A("")

    # A SECOND, WEAKER kind of decision change, deliberately kept separate: ciNPWT stops
    # being cost-SAVING while remaining cost-EFFECTIVE. Different conversation with a
    # payer, and it happens in scenarios where the INMB sign never moves.
    not_saving = run_rows[(run_rows["cost_saving"] == "N")
                          & (run_rows["inmb_100k"] > 0)]
    A("### A weaker decision change: cost-effective but no longer cost-saving")
    A("")
    A("In the base case ciNPWT is **dominant** -- it costs less *and* gains QALYs -- at "
      "every stratum, so no budget-holder has to weigh a price against a QALY at all. A "
      "scenario can take that away without turning INMB negative: ciNPWT then remains "
      "cost-**effective** at $100,000/QALY while ceasing to be cost-**saving**. That is a "
      "materially different conversation with a payer and is reported separately from "
      "the sign flips above (`cost_saving` column in `scenarios_results.csv`).")
    A("")
    if len(not_saving) == 0:
        A("_No scenario in the panel does this: wherever INMB stays positive, ciNPWT also "
          "stays cost-saving._")
    else:
        A("| Scenario | Stratum | Inc. cost | ICER | INMB @ $100k | Status |")
        A("|---|---|---:|---:|---:|---|")
        for _, r in not_saving.iterrows():
            A(f"| `{r['scenario']}` | {r['stratum']} | **+{_money(r['inc_cost'])}** "
              f"| {_icer(r['icer'], r['dominance'])} | {_money(r['inmb_100k'])} "
              f"| {r['dominance']} |")
        A("")
        A("The attribution-adjusted row is the one to note: at low risk, halving the "
          "SSI-attributable cost is enough to make ciNPWT a net *purchase* rather than a "
          "net *saving*, even though it still buys QALYs at well under $100,000 each.")
    A("")

    # ---------------- the full scenario table ------------------------------- #
    A("## Scenario table")
    A("")
    A(f"| Scenario | Stratum | p | Inc. cost | Inc. QALY | ICER | INMB @ ${WTP:,.0f} | "
      "CE quadrant | Sign flip? |")
    A("|---|---|---:|---:|---:|---|---:|---|---|")
    for _, r in df.iterrows():
        if r["dominance"] == "not_run":
            A(f"| `{r['scenario']}` | all | -- | -- | -- | not run | -- | -- "
              "| not run |")
            continue
        flip = {"Y": "**YES**", "N": "no",
                "reference": "-- (reference)"}[r["flips_vs_basecase"]]
        A(f"| `{r['scenario']}` | {r['stratum']} | {_pct(r['p'], 2)} "
          f"| {_money(r['inc_cost'])} | {r['inc_qaly']:+.5f} "
          f"| {_icer(r['icer'], r['dominance'])} | {_money(r['inmb_100k'])} "
          f"| {r['quadrant']} | {flip} |")
    A("")
    A("`ICER` is reported as `n/a (dominant)` wherever ciNPWT is cheaper **and** better: "
      "a cost-effectiveness *ratio* is uninformative under strict dominance and the "
      "engine returns `NaN` there by design. Read `dominance` / `CE quadrant`, never a "
      "bare `NaN`. `NW` = dominant.")
    A("")

    # ---------------- horizons and the toll table --------------------------- #
    A("## Horizons -- which SSI cost tranches are charged")
    A("")
    A("SSI cost enters as **three marginal Edmiston tranches** charged at the cycle of "
      "onset and at onset **+6** and **+12** cycles. Onset is confined to the first 90 "
      f"days, so entrants occupy `SUP_0`/`DEEP_0` at cycles 1-{model.ONSET_CYCLES} and "
      "the three tranches land at cycles **1-3**, **7-9** and **13-15**. A toll falling "
      "beyond the horizon is not charged. This is by design (BUILD-CONTRACT section "
      "6.2), not truncation error:")
    A("")
    A("| Horizon | Cycles | Tranche 1 (1-3) | Tranche 2 (7-9) | Tranche 3 (13-15) | "
      "Cost per SSI charged (2025 USD) |")
    A("|---|---:|---|---|---|---:|")
    for h in (3, 6, 12, 24, 60):
        cells = [{"full": "**charged**", "partial": "*partial*", "none": "not charged"}[
            d["charged"]] for d in model.tranches_charged(h)]
        amt = model.ssi_cost_per_case(horizon_cycles=h)["weighted"]
        A(f"| {HORIZON_LABELS[h]} | {h} | {cells[0]} | {cells[1]} | {cells[2]} "
          f"| ${amt:,.0f} |")
    A("")
    t2d, t2s = I.cost.c_ssi_deep_6_12mo, I.cost.c_ssi_sup_6_12mo
    A("**The 1-year row is the trap.** Tranche 2 lands at cycles 7-9, comfortably "
      "*inside* a 12-cycle horizon, and **is** charged; only tranche 3 is not. Dropping "
      f"it would silently remove ${float(t2d):,.0f} (deep) / ${float(t2s):,.0f} "
      f"(superficial) in {int(t2d.price_year)} USD from every 1-year row and would look "
      "like a plausible horizon effect rather than a bug. The selection here is driven "
      "off `model.tranches_charged()` / `model.ssi_cost_per_case(horizon_cycles=)`, "
      "which raises on a horizon that clips a tranche, rather than a hand-written "
      "filter.")
    A("")
    A("### The 5-year row is identical to the 2-year row, and that is correct")
    A("")
    A("In the base case the 5-year result reproduces the 2-year result **exactly** "
      f"(measured max deviation: incremental cost {five_year_dev['inc_cost']:.2e}, "
      f"incremental QALY {five_year_dev['inc_qaly']:.2e}, INMB "
      f"{five_year_dev['inmb_100k']:.2e} -- asserted in `run_scenarios.py`, not merely "
      "observed). Four structural facts force it:")
    A("")
    A("1. the SSI hazard is **zero after 90 days**, so no new event can occur after "
      "cycle 3;")
    A("2. all three cost tranches are charged by **cycle 15**;")
    A(f"3. both tunnels drain into `RECOV` by cycle {1 + I.DEEP_LEN + model.ONSET_CYCLES}"
      ", so no utility decrement survives past cycle 16; and")
    A("4. from there both arms hold **identical alive mass in states of identical "
      "utility** (`RECOV` = `WELL` in the base case), because mortality does not depend "
      "on state.")
    A("")
    A("So the 5-year row is **not** a redundant copy-paste, and it is **not** evidence "
      "that the horizon is irrelevant. It is evidence that under full recovery the model "
      "has nothing left to say after two years. The 5-year horizon becomes informative "
      "**only** through the residual-decrement scenario, where `RECOV` utility differs "
      "from `WELL` and the two arms therefore diverge for every remaining cycle:")
    A("")
    A("| Stratum | 2 y, full recovery | 5 y, full recovery | 2 y, residual 0.02 | "
      "5 y, residual 0.02 |")
    A("|---|---:|---:|---:|---:|")
    for sname in STRATA:
        def _g(k):
            return df[(df["scenario"] == k) & (df["stratum"] == sname)].iloc[0]
        r2, r5 = _g("horizon_2y_base"), _g("horizon_5y")
        rr2, rr5 = _g("residual_decrement"), _g("horizon_5y_residual_decrement")
        A(f"| {sname} | {_money(r2['inmb_100k'])} | {_money(r5['inmb_100k'])} "
          f"| {_money(rr2['inmb_100k'])} | **{_money(rr5['inmb_100k'])}** |")
    A("")
    A("The first two columns are equal by construction; the last two are not. That "
      "contrast **is** the horizon result and is the only honest way to report a 5-year "
      "row in this model.")
    A("")

    # ---------------- Medicare footnote ------------------------------------- #
    mcr = model.ssi_cost_per_case(sample=MEDICARE_SAMPLE)
    com = model.ssi_cost_per_case()
    sup_share = mcr["frac_sup"] * mcr["superficial"] / mcr["weighted"]
    # A2: MEDICARE_EXTRAPOLATED is now empty by construction -- no Medicare tranche is
    # extrapolated any more. Assert that rather than computing shares of an empty set,
    # which is what produced the published "0.0% / $0" artefact.
    assert not MEDICARE_EXTRAPOLATED, (
        f"MEDICARE_EXTRAPOLATED should be empty after the A2 correction, found "
        f"{MEDICARE_EXTRAPOLATED}. If a Medicare tranche is genuinely extrapolated again, "
        f"restore the disclosure block in build_summary_md() as well as this list."
    )

    A("## Medicare perspective -- the required footnote")
    A("")
    A(f"Weighted 24-month incremental cost per SSI: **${com['weighted']:,.0f}** "
      f"commercial vs **${mcr['weighted']:,.0f}** Medicare (2025 USD, deep fraction "
      f"{mcr['frac_deep']:.3f}).")
    A("")
    A("**A2 RETIRED THIS FOOTNOTE (2026-08-17).** Until the full-text verification gate "
      "this section stated that Edmiston reports the Medicare *superficial* payment at "
      "6 months only, and that the 12- and 24-month Medicare superficial tranches were "
      "`derived_assumption` extrapolations transferring the commercial growth ratio onto "
      "the reported 6-month value. **All three Medicare superficial payments are "
      "reported** -- $11,044, $17,967 and $24,096 -- in Edmiston's Abstract, Results, "
      "Table 3 and Discussion. The correct marginal tranches are $6,923 and $6,129, each "
      "roughly twice the extrapolated values this build used ($3,259 and $3,124). The "
      "extrapolation also failed in principle: Medicare payments grow FASTER than "
      "commercial between 6 and 12 months (1.627 vs 1.295), so a transferred commercial "
      "ratio understates them.")
    A("")
    A(f"**Every dollar in the Medicare row is now a reported Edmiston value**, and the "
      f"four-tranche identity closes exactly: 11,044 + 6,923 + 6,129 = 24,096. "
      f"Superficial SSI contributes {sup_share:.1%} of the weighted Medicare 24-month "
      f"cost per SSI at a superficial fraction of {mcr['frac_sup']:.3f}. No tranche in "
      f"this row is extrapolated, so no extrapolation footnote is required anywhere the "
      f"row appears.")
    A("")
    A("`medicare_6mo_horizon` (6 cycles) is retained, but its rationale has changed: it "
      "was the clean fallback that avoided the extrapolated tranches, and since there are "
      "no extrapolated tranches it is now simply a horizon sensitivity. At 6 cycles only "
      "tranche 1 (cycles 1-3) is charged and no tranche is clipped, so read it as a "
      "floor rather than a like-for-like substitute:")
    A("")
    A("| Stratum | Medicare 2 y | Medicare 6 mo | Commercial 2 y (base case) |")
    A("|---|---:|---:|---:|")
    for sname in STRATA:
        m2 = df[(df["scenario"] == "medicare") & (df["stratum"] == sname)].iloc[0]
        m6 = df[(df["scenario"] == "medicare_6mo_horizon")
                & (df["stratum"] == sname)].iloc[0]
        b = df[(df["scenario"] == "basecase") & (df["stratum"] == sname)].iloc[0]
        A(f"| {sname} | {_money(m2['inmb_100k'])} | {_money(m6['inmb_100k'])} "
          f"| {_money(b['inmb_100k'])} |")
    A("")

    # ---------------- decolonization ---------------------------------------- #
    A("## Decolonization -- the secondary analysis, in one line")
    A("")
    A("Universal *S. aureus* decolonization is background in **both** arms (SHEA/IDSA "
      "essential practice for implant procedures), so both its cost (~$60) and its "
      "effect **cancel exactly** from every base-case increment. It survives only as a "
      "secondary analysis, and the plan specifies exactly one transparent line for it:")
    A("")
    A("```")
    A("residual-risk multiplier = (1 - f_SA) + f_SA x OR_decol")
    A(f"                        = (1 - {F_SA:.2f}) + {F_SA:.2f} x {OR_DECOL:.2f}")
    A(f"                        = {1 - F_SA:.2f} + {F_SA * OR_DECOL:.4f}")
    A(f"                        = {M_DECOL:.4f}")
    A("```")
    A("")
    A(f"using an *S. aureus* fraction of spine-SSI organisms of **{F_SA:.0%}** "
      "(Freire-Archer 2023 [17]) and the spine-specific combined CHG + mupirocin odds "
      f"ratio **{OR_DECOL:.2f}** (Joshi 2026 [2]). Only the *S. aureus* share of the SSI "
      "pool is assumed to respond; the remaining "
      f"{1 - F_SA:.0%} is left untouched, which is why the multiplier is "
      f"{M_DECOL:.3f} and not {OR_DECOL:.2f}.")
    A("")
    p_star = thresholds["base_discounted"]
    p_star_undisc = thresholds["base_undiscounted"]
    p_star_atten = thresholds["attenuated"]
    A("**How far does decolonization shift patients below the ciNPWT threshold?** The "
      "ciNPWT break-even risk is a threshold on the risk the patient actually carries "
      "*into* the ciNPWT decision -- i.e. on the **residual** risk. Decolonization "
      f"multiplies that residual by {M_DECOL:.3f}, so expressed on the "
      "**pre-decolonization** risk scale the same break-even sits at "
      f"`p* / {M_DECOL:.3f}`:")
    A("")
    A("| Threshold, base-case reporting settings | Residual-risk scale | "
      "Pre-decolonization risk scale |")
    A("|---|---:|---:|")
    A(f"| ciNPWT break-even `p*` (2 y, 3%/yr, WTP ${WTP:,.0f}) | **{_pct(p_star)}** "
      f"| **{_pct(p_star / M_DECOL)}** |")
    A(f"| with attenuated effect RR = {RR_ATTEN:.2f} | {_pct(p_star_atten)} "
      f"| {_pct(p_star_atten / M_DECOL)} |")
    A("")
    A(f"So decolonization moves the required baseline risk **up by a factor "
      f"{1 / M_DECOL:.4f} (+{100 * (1 / M_DECOL - 1):.1f}%)**: patients whose "
      f"pre-decolonization 90-day risk lies between **{_pct(p_star)}** and "
      f"**{_pct(p_star / M_DECOL)}** cleared the ciNPWT threshold before "
      "decolonization and fall below it afterwards. That band is narrow in absolute "
      "terms because the threshold itself is low -- the honest reading is that "
      "decolonization removes only a thin sliver of patients from ciNPWT eligibility, "
      "not that it displaces ciNPWT.")
    A("")
    A("| Stratum | Pre-decolonization p | Residual p (x "
      f"{M_DECOL:.3f}) | INMB at pre-decol p | INMB at residual p | "
      "INMB at residual p, RR 0.85 |")
    A("|---|---:|---:|---:|---:|---:|")
    for sname in STRATA:
        b = df[(df["scenario"] == "basecase") & (df["stratum"] == sname)].iloc[0]
        d = df[(df["scenario"] == "decol_residual") & (df["stratum"] == sname)].iloc[0]
        da = df[(df["scenario"] == "decol_residual_attenuated")
                & (df["stratum"] == sname)].iloc[0]
        A(f"| {sname} | {_pct(b['p'], 2)} | {_pct(d['p'], 2)} | {_money(b['inmb_100k'])} "
          f"| {_money(d['inmb_100k'])} | {_money(da['inmb_100k'])} |")
    A("")
    A("**Limitation, stated rather than buried.** The residual SSI pool after "
      "decolonization is relatively **more gram-negative and polymicrobial**, because "
      "decolonization removes *S. aureus* selectively. ciNPWT's relative risk is a "
      "pooled all-surgery estimate and is assumed here to apply **equally** to that "
      "residual pool. No evidence in the base supports or refutes that assumption. The "
      f"`decol_residual_attenuated` row relaxes it directly (RR {RR_ATTEN:.2f} applied to "
      "the residual pool) and is the pessimistic corner of this analysis. A second "
      "limitation: the decolonization OR and the *S. aureus* fraction come from "
      "different populations, and `f_sa_ssi` carries no reported CI, so all the "
      "uncertainty in the multiplier comes from the odds ratio "
      f"({float(I.tp.m_decol_residual.low):.3f}-{float(I.tp.m_decol_residual.high):.3f}).")
    A("")
    A("### Break-even risk `p*` on both reporting conventions")
    A("")
    A("*Threshold provenance -- two reporting conventions, do not conflate them.* `p*` "
      "is solved **locally in `run_scenarios.py`** (`run_threshold.py` owns "
      "`threshold_results.csv` and is the authoritative module for every published "
      "threshold; this local solve exists only because the decolonization band above is "
      "defined in terms of `p*`). **Both bases are persisted here, each one named**, "
      "because BUILD-CONTRACT section 10 (\"BASIS LABELS -- added 2026-08-09\") requires "
      "every published threshold to name its basis and makes the **reporting base case** "
      "the manuscript headline:")
    A("")
    A(f"| Scenario | **Reporting base case ({BASE_HORIZON} cycles, "
      f"{float(I.SETTINGS['disc_cost']):.0%}/yr discounting, half-cycle correction) -- "
      f"MANUSCRIPT HEADLINE** | Undiscounted, no half-cycle "
      f"(BUILD-CONTRACT section-10 hand-calc cross-check only) |")
    A("|---|---:|---:|")
    A(f"| Base case -- matched attributable SSI cost (Pennington [14], "
      f"m_ssi_cost = {K_ATTRIB:.3f}) | **{_pct(p_star)}** ({_pct(p_star, 4)}) "
      f"| {_pct(p_star_undisc)} ({_pct(p_star_undisc, 4)}) |")
    A(f"| All-cause SSI cost, upper bound (Edmiston [12] at face value  "
      f"m_ssi_cost = 1.000) "
      f"| **{_pct(thresholds['attrib'])}** ({_pct(thresholds['attrib'], 4)}) "
      f"| {_pct(thresholds['attrib_undiscounted'])} "
      f"({_pct(thresholds['attrib_undiscounted'], 4)}) |")
    A("")
    A(f"Every scenario row in this file is run on the **reporting base case**, so the "
      f"reporting-basis column is the one that is like-for-like with everything else "
      f"here and with Table 3 / Table 6: the manuscript's base-case-to-upper-bound pair "
      f"is **{_pct(p_star)} -> {_pct(thresholds['attrib'])}**. The upper-bound row sits "
      f"BELOW the base case, which is the direction A2 established and is the whole "
      f"reason Edmiston's all-cause increment is treated as a bound: a larger averted "
      f"cost makes the device look like it pays off sooner. The undiscounted column "
      f"exists for one purpose -- it is the only basis on which the closed-form hand "
      f"calculation and the engine are commensurable. **Never pair a reporting-basis threshold with an "
      f"undiscounted one in the same sentence.** Discounting raises the threshold by "
      f"{100.0 * (p_star / p_star_undisc - 1):.2f}% at base and "
      f"{100.0 * (thresholds['attrib'] / thresholds['attrib_undiscounted'] - 1):.2f}% "
      f"under the all-cause bound, always in the same direction and for the same structural "
      f"reason: the device toll falls at cycle 0 where the discount factor is exactly 1, "
      f"while every benefit it buys -- the SSI cost tranches at months 1-15 and the "
      f"QALYs across all {BASE_HORIZON} cycles -- is discounted.")
    A("")

    # ---------------- budget impact ----------------------------------------- #
    A("## Budget impact")
    A("")
    A("**How 'risk-targeted ciNPWT' is reported, and why.** The population distribution "
      "of predicted 90-day SSI risk is **unavailable**, so a separately modelled "
      "risk-targeted third arm would require inventing one. Instead, per the plan, "
      "risk-targeted use is reported as a **derived** result: per-stratum per-patient "
      "increments scaled by the number of patients treated, parameterized by the "
      "**proportion treated**. A programme that treats a proportion *f* of its cases at "
      "stratum risk *p* is **mathematically identical** to a third arm whose targeting "
      "rule selects exactly those patients -- and this way nothing is fabricated. Say it "
      "plainly; it is a modelling choice with a reason, not a shortcut.")
    A("")
    A(f"Per-patient increments are the base case (24 cycles, "
      f"{float(I.SETTINGS['disc_cost']):.0%}/yr discounting, {I.PRICE_YEAR} USD). "
      "`inc_cost_total_usd` is the **budget line** -- negative means the programme saves "
      "money -- and `net_budget_impact_100k_usd` is the population INMB at "
      f"${WTP:,.0f}/QALY. The ~$60 decolonization bundle is background in **both** arms "
      "and cancels, so it is not a ciNPWT budget line.")
    A("")
    for size in PROGRAM_SIZES:
        A(f"### {size}-case programme")
        A("")
        A("| Proportion treated | Stratum | Patients treated | Device outlay | "
          "SSI payments averted | **Net budget impact (cost)** | SSIs avoided | "
          "Reoperations avoided | QALYs gained | Net benefit @ $100k |")
        A("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        sub = bi[bi["program_size"] == size]
        for frac in PROPORTIONS_TREATED:
            for sname in STRATA:
                r = sub[(sub["proportion_treated"] == frac)
                        & (sub["stratum"] == sname)].iloc[0]
                A(f"| {frac:.0%} | {sname} | {r['n_treated']:,.0f} "
                  f"| {_money(r['device_outlay_usd'])} "
                  f"| {_money(r['ssi_cost_averted_usd'])} "
                  f"| **{_money(r['inc_cost_total_usd'])}** "
                  f"| {r['ssi_avoided']:.1f} | {r['reop_avoided']:.1f} "
                  f"| {r['qaly_gained']:.2f} "
                  f"| {_money(r['net_budget_impact_100k_usd'])} |")
        A("")
    A("Every quantity is linear in `n_treated`, so any other programme size or "
      "proportion is a scaling of these rows -- which is precisely why a third arm would "
      "add nothing.")
    A("")

    # ---------------- what did not run --------------------------------------- #
    A("## Explicitly not run")
    A("")
    for key, desc, note in NOT_RUN:
        A(f"- **`{key}`** -- {desc}. {note}")
    A("")

    # ---------------- caveats ------------------------------------------------ #
    A("## What these scenarios do and do not carry")
    A("")
    A("- **Verified.** Phase A2 full-text verification completed 2026-08-17: 46 of 48 parameters are closed out against the 14 source full texts and 43 carry a table/page locator. Only 2 still carry `verify=True` -- the ciNPWT institutional net acquisition price and the superficial tunnel length -- because no available source resolves either.")
    A("- **The parameter store is never mutated.** Every scenario is a fresh `sample` "
      "dict (`P()`) plus a fresh run-kwargs dict (`S()`); a snapshot of all 48 "
      "`Param.value`s is compared before and after the run and a mismatch raises.")
    A("- **Cost is incremental-versus-no-SSI**, not total cost of care. There is no "
      "index-surgery and no well-state cost by design -- the index fusion, implant, ward "
      "stay and routine follow-up cancel exactly between arms.")
    A("- **Scenarios are one-at-a-time** except where a joint scenario is defined "
      "(`horizon_5y_residual_decrement`, `decol_residual_attenuated`). They are not a "
      "probabilistic statement; `run_psa.py` carries that. Note in particular that "
      "`m_ssi_cost` is `fixed` and is **not** sampled in the PSA, so attribution "
      "uncertainty lives entirely in the `allcause_edmiston` scenario and in the DSA.")
    A("- **No Medicare tranche is extrapolated any more** (A2 -- see above); the "
      "attribution factor remains a Medicare-basis ratio applied to commercial tranches, "
      "which is a transfer and is labelled as one.")
    A("- **`icer` is `NaN` wherever ciNPWT is dominant**, which is most of this panel. "
      "That is the engine's design, not a failure; report `dominance` / `quadrant`.")
    A("")
    A("---")
    A("")
    A("*Files written: `scenarios_results.csv`, `budget_impact.csv`, "
      "`scenarios_summary.md` -- all in `v2/results/`, each CSV carrying the results "
      "banner as line 1 (read with `pd.read_csv(path, skiprows=1)`).*")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "scenarios_summary.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    return path


# --------------------------------------------------------------------------- #
# 10. Main                                                                     #
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df, base_inmb = build_scenarios()
    five_year_dev = check_five_year_identity(df)

    #: Two reporting conventions, never to be conflated, and BOTH PERSISTED to
    #: `scenarios_summary.md` with their bases named (BUILD-CONTRACT section 10, "BASIS
    #: LABELS -- added 2026-08-09"). `*_discounted` / `attrib` use the base-case
    #: REPORTING settings (24 cycles, 3%/yr, half-cycle on) -- the setting every scenario
    #: row in this file is run under, therefore the right basis for the decolonization
    #: band AND the manuscript headline. `*_undiscounted` drops discounting and the
    #: half-cycle correction and is the BUILD-CONTRACT section-10 hand-calculation
    #: cross-check only. A2 moved both targets; they are solved live, not asserted here.
    #: `attrib` used to be printed and then dropped on the floor, which left the
    #: manuscript's own like-for-like pair (base -> attribution, both on the reporting
    #: basis) reachable from no persisted artifact at all. It is now in the summary.
    _undisc = dict(half_cycle=False, disc_cost=0.0, disc_qaly=0.0)
    thresholds = {
        "base_discounted": solve_threshold(),
        "base_undiscounted": solve_threshold(run_kw=S(**_undisc)),
        "attenuated": solve_threshold(run_kw=S(rr=RR_ATTEN)),
        # A2: solved at m_ssi_cost = 1.0, NOT at K_ATTRIB. K_ATTRIB (0.492) is now the
        # BASE CASE, so solving here at K_ATTRIB reproduced the base case exactly and this
        # file published a null "1.235% -> 1.235%" pair to the manuscript. The sensitivity
        # runs the other way now: the all-cause increment at face value.
        "attrib": solve_threshold(sample=P(m_ssi_cost=1.0)),
        "attrib_undiscounted": solve_threshold(sample=P(m_ssi_cost=1.0),
                                               run_kw=S(**_undisc)),
    }

    bi = build_budget_impact(df)

    # A2 housekeeping: `effect_superficial_only` leaves the TRACE untouched (it zeroes
    # prices and a decrement, not hazards), so its nnt / ssi_prevented_per_1000 /
    # reop_prevented_per_1000 columns come back identical to the base case. For a scenario
    # whose premise is that ciNPWT prevents none of the deep SSIs -- about three quarters
    # of them -- those counters are meaningless and a reader opening the CSV would take
    # them at face value. Blank them rather than publish a number that is wrong.
    _sup = df["scenario"] == "effect_superficial_only"
    if _sup.any():
        for _c in ("nnt", "ssi_prevented_per_1000", "reop_prevented_per_1000"):
            if _c in df.columns:
                df.loc[_sup, _c] = ""
    write_results_csv(df, "scenarios_results.csv")
    write_results_csv(bi, "budget_impact.csv")
    build_summary_md(df, bi, five_year_dev, thresholds)

    _assert_store_unmutated()

    # ---------------- stdout ------------------------------------------------- #
    print("=" * 78)
    print("SSI / ciNPWT risk-threshold Markov CUA -- SCENARIOS + BUDGET IMPACT")
    print(I.RESULTS_BANNER)
    print("=" * 78)
    print(f"Panel: {len(SCENARIOS)} scenarios x {len(STRATA)} strata "
          f"+ {len(NOT_RUN)} not_run rows = {len(df)} rows | WTP ${WTP:,.0f}/QALY | "
          f"{I.PRICE_YEAR} USD")
    print(f"Base-case INMB: " + "  ".join(f"{s} ${base_inmb[s]:,.0f}" for s in STRATA))
    print("Break-even risk p* -- TWO REPORTING CONVENTIONS, do not conflate "
          "(run_threshold.py is authoritative). BOTH are written to "
          "scenarios_summary.md:")
    print(f"  base case          reporting base case (2 y, 3%/yr, HCC) = "
          f"{100 * thresholds['base_discounted']:.4f}%   <- MANUSCRIPT HEADLINE")
    print(f"  base case          undiscounted, no HCC                  = "
          f"{100 * thresholds['base_undiscounted']:.4f}%   [BUILD-CONTRACT 10: 0.6515%]")
    print(f"  all-cause upper bound x1.000 reporting base case (2 y, 3%/yr, HCC) = "
          f"{100 * thresholds['attrib']:.4f}%   <- MANUSCRIPT HEADLINE")
    print(f"  attribution x{K_ATTRIB:.3f} undiscounted, no HCC                  = "
          f"{100 * thresholds['attrib_undiscounted']:.4f}%   [BUILD-CONTRACT 10: 1.372%]")
    print(f"  like-for-like manuscript pair, reporting base case: "
          f"{100 * thresholds['base_discounted']:.3f}% -> "
          f"{100 * thresholds['attrib']:.3f}%")
    print(f"Decolonization multiplier {M_DECOL:.4f} -> pre-decolonization threshold "
          f"{100 * thresholds['base_discounted'] / M_DECOL:.4f}% "
          f"(+{100 * (1 / M_DECOL - 1):.1f}%)")
    print(f"5 y vs 2 y base case: max |dev| inc_cost {five_year_dev['inc_cost']:.2e}, "
          f"inc_qaly {five_year_dev['inc_qaly']:.2e}, INMB {five_year_dev['inmb_100k']:.2e} "
          "(identical by construction -- asserted)")
    print()

    show = df[df["dominance"] != "not_run"].pivot_table(
        index="scenario", columns="stratum", values="inmb_100k", aggfunc="first")
    show = show.reindex([k for k, *_ in SCENARIOS])[list(STRATA)]
    print(f"INMB @ ${WTP:,.0f}/QALY by scenario x stratum (B minus A):")
    print(show.to_string(float_format=lambda v: f"{v:,.0f}"))
    print()

    flips = df[(df["flips_vs_basecase"] == "Y")]
    if len(flips):
        print(f"SIGN FLIPS vs base case ({len(flips)} scenario x stratum cells):")
        for _, r in flips.iterrows():
            print(f"  {r['scenario']:<32s} {r['stratum']:<14s} p={r['p']:.4f} "
                  f"INMB ${r['inmb_100k']:>10,.0f}   {r['dominance']}")
    else:
        print("SIGN FLIPS vs base case: NONE -- the decision is robust across the panel.")
    print()
    print("Not run (explicit rows): " + ", ".join(k for k, *_ in NOT_RUN))
    print()

    print("Budget impact (1,000-case programme, 100% treated):")
    for sname in STRATA:
        r = bi[(bi["program_size"] == 1000) & (bi["proportion_treated"] == 1.0)
               & (bi["stratum"] == sname)].iloc[0]
        print(f"  {sname:<14s} net cost {r['inc_cost_total_usd']:>12,.0f} | "
              f"SSIs avoided {r['ssi_avoided']:>6.1f} | reops avoided "
              f"{r['reop_avoided']:>5.1f} | QALYs {r['qaly_gained']:>6.2f} | "
              f"net benefit ${r['net_budget_impact_100k_usd']:>10,.0f}")
    print()

    print("Files written to:", RESULTS_DIR)
    expected = ["scenarios_results.csv", "budget_impact.csv", "scenarios_summary.md"]
    missing = []
    for fn in expected:
        path = os.path.join(RESULTS_DIR, fn)
        ok = os.path.exists(path)
        if not ok:
            missing.append(fn)
        print(f"  [{'OK' if ok else 'MISSING'}] {fn}")
    if missing:
        raise RuntimeError(f"run_scenarios.py failed to write: {missing}")


if __name__ == "__main__":
    main()
