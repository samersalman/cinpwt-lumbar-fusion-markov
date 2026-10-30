"""
model.py -- Markov engine for the SSI / ciNPWT risk-threshold cost-utility analysis.

At what predicted 90-day SSI risk does prophylactic closed-incision negative-pressure
wound therapy (ciNPWT) pay off after instrumented posterior lumbar fusion?

Arms        : A_standard = contemporary prevention + standard dressing (comparator)
              B_cinpwt   = contemporary prevention + ciNPWT (intervention)
Perspective : US commercial payer (payments basis), 2025 USD.
Cycle       : 1 month = 1/12 year. Base horizon 24 cycles (2 years).
Discounting : disc_t = 1 / (1 + r) ** (t * cycle_years), r annual, 3% base case.
Half-cycle  : trapezoidal (hc[0] = hc[-1] = 0.5) on DWELL streams ONLY.
              TOLLS ARE NEVER HALF-CYCLE WEIGHTED (point events, attributed whole).

EVERY number comes from `inputs.py` (BUILD-CONTRACT section 9: only inputs.py owns a
number). Every parameter is read through `val()` (dimensionless) or `_cost()` (dollars),
so PSA can override any input by name.

=============================================================================
GATE STATUS: VERIFIED -- see inputs.RESULTS_BANNER. Phase A2 full-text
verification completed 2026-08-17; 46 of 48 parameters are closed out against the
14 source full texts and 43 carry a table/page locator. The 2 that remain are the
ciNPWT institutional net acquisition price and the superficial tunnel length.
=============================================================================

-----------------------------------------------------------------------------
STRUCTURE (BUILD-CONTRACT section 2, as amended 2026-08-09)
-----------------------------------------------------------------------------
5 clinical states expanded into monthly tunnels. The slot vector is BUILT, not
hard-coded: `build_layout(n_sup, n_deep)` returns the index map, and
`build_layout(3, 12)` reproduces the frozen 18-slot base-case vector exactly
(asserted against `inputs.SLOT_NAMES` at import).

    slot 0                       WELL
    slots 1 .. n_sup             SUP_0 .. SUP_{n_sup-1}     superficial tunnel
    slots 1+n_sup .. +n_deep     DEEP_0 .. DEEP_{n_deep-1}  deep tunnel
    slot  1+n_sup+n_deep         RECOV
    slot  n_slots-1              DEAD (absorbing)

`run_arm(..., n_sup=None, n_deep=None)` defaults to the `inputs.py` widths. Only
`run_dsa.py` / `run_scenarios.py` may instantiate another width, and traces of
differing widths must be collapsed to the 5 `inputs.CLINICAL_STATES` (see
`collapse_trace`) before being compared or concatenated.

-----------------------------------------------------------------------------
THREE-LAYER SPLIT (mirct idiom): _cycle_rates -> _build_P -> run_arm
-----------------------------------------------------------------------------
`_cycle_rates` is the SINGLE SOURCE OF TRUTH for one cycle's survivor-conditional
rates. `_build_P` turns that bundle into a row-stochastic matrix; `run_arm` uses the
SAME bundle to compute the SSI-onset flows that drive the cost tolls. The matrix and
the toll accounting therefore cannot drift apart.

ORDER OF OPERATIONS inside every live row (the house convention, as in the sibling
engines): background death first, then the surviving mass `s = 1 - q` is distributed
among the clinical transitions. So the onset weights are conditional probabilities
AMONG SURVIVORS of the cycle, which is the natural reading of a competing-risks model.

-----------------------------------------------------------------------------
SSI ONSET -- the single easiest thing to get subtly wrong
-----------------------------------------------------------------------------
Onset occurs ONLY in the first 90 days and is ZERO from cycle 4 onward, so the
pivotal hazard is never extrapolated. `w_onset_*_m1..m3` are CONDITIONAL-PROBABILITY
inputs, NOT per-cycle hazards. The engine applies

    h_1 = p*w_1
    h_2 = p*w_2 / (1 - p*w_1)
    h_3 = p*w_3 / (1 - p*w_1 - p*w_2)

so the cumulative 90-day onset RISK equals `p` EXACTLY (asserted to 1e-12 over the
whole 291-point risk grid in the smoke test below). Using w_k directly as the cycle
hazard undershoots p by 3-4% at the top of the range.

The TOTAL hazard is set first (blended weights
`w_k = frac_deep*w_deep_k + frac_sup*w_sup_k`, which sum to 1 by construction) and is
split into deep/superficial with `frac_deep`/`frac_sup` AFTERWARDS, so the destination
split cannot perturb total incidence.

Arm B applies RR to the onset hazard INSIDE the engine (`p_ciNPWT = p_control x RR`),
so the two arms are correlated by construction and arm ordering holds in every PSA draw.

COMPETING MORTALITY, stated plainly: the cumulative onset RISK is exactly `p`, but the
cumulative expected COUNT `n_ssi` in a cohort that also faces perioperative (0.2%,
cycle 0) and background mortality is ~0.3% RELATIVE below `p` (2.1e-4 absolute at
p=0.066). That is correct competing-risks behaviour, not a bug -- `run_log.d/inputs.md`
says so explicitly -- and it is bounded and asserted in the smoke test.

-----------------------------------------------------------------------------
COSTS (BUILD-CONTRACT section 6 -- do not re-derive)
-----------------------------------------------------------------------------
* EVERY cost is read through `inputs.cost_2025(param, value)` via the local `_cost()`
  helper. NEVER call `inflate()` on a cost. Twelve params are 2019 USD and inflate;
  three are already 2025 USD and must not.
* SSI cost = three MARGINAL Edmiston tranches charged at onset+0, onset+6, onset+12
  cycles (`SETTINGS["toll_offset_cycles"]`). A toll beyond the horizon is not charged.
  Onset sits at cycles 1-3, so the tranches land at cycles 1-3, 7-9 and 13-15:

    | horizon | tranche 1 | tranche 2 (7-9) | tranche 3 (13-15) |
    | 3   (90d) | charged | NOT charged | NOT charged |
    | 12  (1y)  | charged | CHARGED     | NOT charged |
    | 24  (2y)  | charged | CHARGED     | CHARGED     |
    | 60  (5y)  | charged | CHARGED     | CHARGED     |

  The 1-year row is the trap: dropping tranche 2 would silently remove $15,109 (deep) /
  $6,137 (superficial) in 2019 USD from every 1-year scenario.
* Tolls are attached to the ONSET COHORT (the incidence flow), never to tunnel
  occupancy -- occupancy and onset cohort are not the same thing once mortality removes
  people, and the deep tunnel has emptied into RECOV before the onset+12 toll falls due.
* `m_ssi_cost` multiplies all six tranches together.
* ciNPWT device cost = one-time toll at cycle 0, ARM B ONLY.
  Standard-dressing differential = one-time toll at cycle 0, ARM A ONLY ($0 base case).
* NO index-surgery and NO well-state cost exist, by design. Per-arm "cost" is
  incremental-versus-no-SSI cost, not total cost of care.
* Decision 3 is absolute: `p_persistent_deep`, `p_reop_ssi`, `n_debridements` and
  `rr_readmit_sa` are COUNTERS ONLY. None of them multiplies a cost anywhere in this
  file. If one ever does, the model double-counts against Edmiston's all-cause increment.

See: v2/BUILD-CONTRACT.md, v2/run_log.d/inputs.md, v2/run_log.d/phase-a1.md,
v2/run_log.d/model.md (this module's own run-log fragment).
"""

from __future__ import annotations

import math
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd

import inputs as I
from inputs import tp, cost, util, mortality


# --------------------------------------------------------------------------- #
# 0. Frozen-signature defaults, cross-checked against inputs.py                #
#    (the literals below are BUILD-CONTRACT section 5 signature defaults; the  #
#     asserts guarantee they can never drift away from the parameter store)    #
# --------------------------------------------------------------------------- #
assert I.SETTINGS["horizon_cycles"] == 24, "run_arm's frozen default horizon must be 24"
assert I.SETTINGS["disc_cost"] == 0.03 and I.SETTINGS["disc_qaly"] == 0.03
assert I.SETTINGS["wtp_primary"] == 100_000.0
assert I.SETTINGS["half_cycle"] is True
assert I.SETTINGS["cycle_years"] == 1.0 / 12.0

#: The 90-day onset window, in cycles. NOT a magic number and NOT the tunnel width:
#: it is the number of onset-weight Params that inputs.py declares.
_W_ONSET_DEEP = (tp.w_onset_deep_m1, tp.w_onset_deep_m2, tp.w_onset_deep_m3)
_W_ONSET_SUP = (tp.w_onset_sup_m1, tp.w_onset_sup_m2, tp.w_onset_sup_m3)
ONSET_CYCLES = len(_W_ONSET_DEEP)
assert ONSET_CYCLES == len(_W_ONSET_SUP) == 3

#: The three marginal SSI cost tranches, in toll order (onset+0, +6, +12).
_C_SUP_TRANCHES = (cost.c_ssi_sup_0_6mo, cost.c_ssi_sup_6_12mo, cost.c_ssi_sup_12_24mo)
_C_DEEP_TRANCHES = (cost.c_ssi_deep_0_6mo, cost.c_ssi_deep_6_12mo, cost.c_ssi_deep_12_24mo)
assert len(I.SETTINGS["toll_offset_cycles"]) == len(_C_SUP_TRANCHES) == 3

#: Dominance / quadrant vocabularies (BUILD-CONTRACT section 5).
DOMINANCE_VOCAB = (
    "ciNPWT dominant",                      # cheaper & better  -> NW
    "ciNPWT dominated",                     # pricier & worse   -> SE
    "ciNPWT more costly & more effective",  #                   -> NE
    "ciNPWT less costly & less effective",  #                   -> SW
    "equal",                                #                   -> origin
)
QUADRANT_VOCAB = ("NE", "NW", "SE", "SW", "origin")

#: `quadrant` is DERIVED FROM `dominance` -- the two fields can therefore never disagree
#: (post-review fix 1, 2026-08-09). Re-deriving the quadrant from the raw signs with
#: strict inequalities, `("N" if inc_qaly > 0 else "S") + ("E" if inc_cost > 0 else "W")`,
#: is wrong on both degenerate zero cases:
#:     inc_cost < 0, inc_qaly == 0  -> 'ciNPWT dominant'  scored "SW", must be "NW"
#:     inc_cost == 0, inc_qaly < 0  -> 'ciNPWT dominated' scored "SW", must be "SE"
#: Table 2 and figS1 read BOTH fields off the same row, so a disagreement would print a
#: cheaper-and-better result in the more-costly-and-less-effective quadrant.
DOMINANCE_TO_QUADRANT = {
    "ciNPWT dominant": "NW",                      # cheaper (or equal) & better
    "ciNPWT dominated": "SE",                     # pricier (or equal) & worse
    "ciNPWT more costly & more effective": "NE",
    "ciNPWT less costly & less effective": "SW",
    "equal": "origin",
}
assert set(DOMINANCE_TO_QUADRANT) == set(DOMINANCE_VOCAB)
assert set(DOMINANCE_TO_QUADRANT.values()) == set(QUADRANT_VOCAB)


# --------------------------------------------------------------------------- #
# 1. Parameter accessors (the PSA override hooks)                              #
# --------------------------------------------------------------------------- #
def val(param, sample=None) -> float:
    """Return a DIMENSIONLESS parameter's value, overridden by `sample` when given.

    `param` is an `inputs.Param`. If `sample` is a dict containing `param.name`, that
    sampled value is returned; otherwise `param.value`. Used for EVERY read of a
    probability, utility, fraction or multiplier so PSA/DSA can override any input by
    name.

    DO NOT use this for a cost -- costs live in two price years and must go through
    `_cost()` / `inputs.cost_2025()`, which inflates iff the parameter's stored price
    year differs from 2025. `val(cost.c_ssi_deep_0_6mo)` would silently return a
    2019-USD number.
    """
    if sample is not None and param.name in sample:
        return float(sample[param.name])
    return float(param.value)


def _cost(param, sample=None, override=None) -> float:
    """Return a COST parameter in 2025 USD -- the only way a dollar enters this engine.

    Resolution order (highest first):
      1. `override`   -- an explicit float passed by a sweep (e.g. run_arm(device_cost=))
      2. `sample`     -- a PSA draw / DSA bound keyed by `param.name`
      3. `param.value`

    Whatever wins is expressed in the parameter's OWN stored price year and is handed to
    `inputs.cost_2025(param, value)`, which inflates iff `param.price_year != 2025`. So a
    gamma draw on `c_ssi_deep_0_6mo` is a 2019-USD draw and is inflated here exactly once,
    while `c_device_cinpwt` (already 2025 USD) is returned unchanged.
    """
    if override is not None:
        v = float(override)
    elif sample is not None and param.name in sample:
        v = float(sample[param.name])
    else:
        v = None
    return float(I.cost_2025(param, v))


#: Every name a `sample` dict may legitimately carry: the `Param.name` of every
#: parameter in the store. A key outside this set is either a typo or a name this engine
#: never reads -- both change nothing, so `run_arm` WARNS instead of accepting silently
#: (post-review fix 5, 2026-08-09; the exact trap a PSA/DSA driver walks into).
KNOWN_PARAM_NAMES = frozenset(p.name for p in I.all_params())

#: `sample` CANNOT set the tunnel widths -- they are a LAYOUT, chosen by the
#: `n_sup=` / `n_deep=` kwargs. `sample={"n_tunnel_deep": 24}` used to be accepted and
#: silently ignored (QALY delta exactly 0.0, trace still 18 slots wide).
#: name -> (layout key it would have to agree with, the kwarg that actually sets it)
_LAYOUT_PARAM_TO_KWARG = {
    "n_tunnel_sup": ("sup_len", "n_sup"),
    "n_tunnel_deep": ("deep_len", "n_deep"),
}


def _check_p_base(p_base) -> float:
    """Validate the study's x-axis and return it as a float.

    `p_base` is a 90-day cumulative RISK, so it must be a probability. Outside [0, 1] the
    conditional-onset arithmetic is meaningless: a negative `p_base` used to run to
    completion and return `n_ssi = 0.0` with no complaint, while `arm`, `horizon_cycles`
    and the tunnel widths all validated loudly (post-review fix 6, 2026-08-09).
    """
    p = float(p_base)
    if not (0.0 <= p <= 1.0):
        raise ValueError(
            "p_base must be a probability in [0.0, 1.0] -- it is the 90-day cumulative "
            f"all-SSI RISK in the comparator arm, not a rate or a count; got {p_base!r}"
        )
    return p


def _check_sample(sample, layout) -> None:
    """Guard a `sample` dict against the two ways it fails silently (post-review fix 5).

    1. RAISES if `sample` carries `n_tunnel_sup` / `n_tunnel_deep` that DISAGREE with the
       resolved layout. Those two Params do not drive the engine layout -- the
       `n_sup=` / `n_deep=` kwargs do -- so a sweep that sampled them would produce
       silent zero-swing rows (exactly the failure the BUILD-CONTRACT section 2 amendment
       was written to prevent). A value that AGREES with the resolved layout is harmless
       and is allowed through.
    2. WARNS (never raises) on any key that is not a known `inputs.all_params()` name. A
       typo in a PSA/DSA driver is worse than a warning: it would sweep nothing while
       reporting a swing of zero. Raising here is not an option -- an over-broad but
       correctly-spelled sample is legitimate (the engine ignores `n_debridements` and
       `rr_readmit_sa` by design).
    """
    if not sample:
        return

    for pname, (layout_key, kwarg) in _LAYOUT_PARAM_TO_KWARG.items():
        if pname in sample:
            want = int(round(float(sample[pname])))
            have = int(layout[layout_key])
            if want != have:
                raise ValueError(
                    f"sample[{pname!r}] = {sample[pname]!r} disagrees with the resolved "
                    f"layout ({layout_key} = {have} cycles) and would be SILENTLY "
                    f"IGNORED: tunnel widths are a layout, not a sampled Param. Pass "
                    f"{kwarg}={want} to run_arm/run_pair instead (and collapse traces "
                    f"with collapse_trace before comparing widths), or drop {pname!r} "
                    f"from the sample."
                )

    for key in sorted(k for k in sample if k not in KNOWN_PARAM_NAMES):
        warnings.warn(
            f"sample key {key!r} is not a known inputs.all_params() parameter name and "
            f"is IGNORED by the engine -- a swing of exactly zero is not evidence of no "
            f"effect. Check the spelling against [p.name for p in inputs.all_params()]; "
            f"tunnel widths use the n_sup= / n_deep= kwargs, not sample.",
            UserWarning,
            stacklevel=3,
        )


def _norm_row(row: np.ndarray) -> np.ndarray:
    """Defensively renormalize one probability row to sum to 1.0.

    Clamps tiny negatives (float noise, or an extreme PSA draw) to 0, then divides by the
    row sum. A degenerate all-zero row sends its mass to the last slot (DEAD) rather than
    creating or destroying cohort mass.
    """
    row = np.where(row < 0.0, 0.0, row)
    total = row.sum()
    if total <= 0.0:
        out = np.zeros_like(row)
        out[-1] = 1.0
        return out
    return row / total


# --------------------------------------------------------------------------- #
# 2. Layout builder (BUILD-CONTRACT section 2, amendment of 2026-08-09)        #
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=64)
def _layout_core(n_sup: int, n_deep: int):
    """Cached immutable core of a slot layout. `build_layout` wraps this in a fresh dict
    so a caller can never mutate another caller's layout."""
    n_sup = int(n_sup)
    n_deep = int(n_deep)
    if n_sup < 1 or n_deep < 1:
        raise ValueError(f"tunnel widths must be >= 1 cycle; got n_sup={n_sup}, n_deep={n_deep}")
    well = 0
    sup_start = 1
    deep_start = sup_start + n_sup
    recov = deep_start + n_deep
    dead = recov + 1
    n_slots = 1 + n_sup + n_deep + 2
    assert dead == n_slots - 1
    names = tuple(
        ["WELL"]
        + [f"SUP_{i}" for i in range(n_sup)]
        + [f"DEEP_{i}" for i in range(n_deep)]
        + ["RECOV", "DEAD"]
    )
    s2c = np.array([0] + [1] * n_sup + [2] * n_deep + [3, 4], dtype=np.int64)
    s2c.flags.writeable = False
    return (n_slots, well, sup_start, n_sup, deep_start, n_deep, recov, dead, names, s2c)


def build_layout(n_sup: int = 3, n_deep: int = 12) -> dict:
    """Build the engine slot vector for tunnels of `n_sup` and `n_deep` cycles.

    BUILD-CONTRACT section 2 was amended on 2026-08-09 so the slot vector is BUILT rather
    than hard-coded: `n_tunnel_sup` (DSA range 2-6) and `n_tunnel_deep` (6-24) are a
    genuine evidence conflict (Urquhart [21] reports PROM recovery by 1 year, McFarland
    [19] a persisting decrement) and deserve to be a real scenario axis rather than two
    silent zero-swing tornado rows.

    Returns a FRESH dict (safe to keep, never shared between callers) with keys:
        n_slots, well, sup_start, sup_len, deep_start, deep_len, recov, dead,
        slot_names (list[str]), slot_to_clinical (read-only np.ndarray[int], len n_slots)

    Invariants, all asserted at import for the base case:
        n_slots        = 1 + n_sup + n_deep + 2
        build_layout(3, 12) reproduces the frozen 18-slot vector EXACTLY --
        same indices, same names, same collapse as inputs.SLOT_NAMES / N_SLOTS /
        WELL / SUP_START / SUP_LEN / DEEP_START / DEEP_LEN / RECOV / DEAD.

    Traces of differing widths must NEVER be compared or concatenated; collapse them to
    the 5 `inputs.CLINICAL_STATES` first (`collapse_trace`).
    """
    (n_slots, well, sup_start, sup_len, deep_start, deep_len,
     recov, dead, names, s2c) = _layout_core(int(n_sup), int(n_deep))
    return {
        "n_slots": n_slots,
        "well": well,
        "sup_start": sup_start,
        "sup_len": sup_len,
        "deep_start": deep_start,
        "deep_len": deep_len,
        "recov": recov,
        "dead": dead,
        "slot_names": list(names),
        "slot_to_clinical": s2c,
    }


def _resolve_layout(n_sup=None, n_deep=None) -> dict:
    """Layout for the given widths, defaulting to the `inputs.py` base-case widths."""
    ns = int(I.SUP_LEN) if n_sup is None else int(n_sup)
    nd = int(I.DEEP_LEN) if n_deep is None else int(n_deep)
    return build_layout(ns, nd)


def collapse_trace(trace: np.ndarray, n_sup=None, n_deep=None) -> np.ndarray:
    """Collapse a raw (T+1, n_slots) engine trace to (T+1, 5) `inputs.CLINICAL_STATES`.

    THE ONLY SAFE WAY to compare or concatenate traces from different tunnel widths
    (BUILD-CONTRACT section 2 amendment). Column order is
    ["Well", "Superficial SSI", "Deep SSI", "Recovered", "Dead"].
    """
    trace = np.asarray(trace, dtype=float)
    L = _resolve_layout(n_sup, n_deep)
    if trace.ndim != 2 or trace.shape[1] != L["n_slots"]:
        raise ValueError(
            f"trace has {trace.shape[-1]} columns but layout has {L['n_slots']} slots; "
            "pass the n_sup/n_deep the trace was produced with"
        )
    out = np.zeros((trace.shape[0], len(I.CLINICAL_STATES)), dtype=float)
    s2c = L["slot_to_clinical"]
    for j in range(L["n_slots"]):
        out[:, s2c[j]] += trace[:, j]
    return out


# --------------------------------------------------------------------------- #
# 3. Cohort resolution (age / sex mix / cycle length)                          #
# --------------------------------------------------------------------------- #
def _cohort(sample=None, settings=None):
    """Resolve (age0, sex_mix, cycle_years).

    The `Param` objects are the source of truth (run_log.d/inputs.md). Resolution order:
    a PSA/DSA draw in `sample` wins; otherwise an explicit `settings` override; otherwise
    the Param. `sex_mix` is passed opaquely to `inputs.background_mortality`, which
    accepts None, a float fraction male, or a {'M':..,'F':..} dict.
    """
    S = I.SETTINGS if settings is None else settings

    if sample is not None and mortality.age_start.name in sample:
        age0 = float(sample[mortality.age_start.name])
    elif settings is not None and settings.get("age0") is not None:
        age0 = float(settings["age0"])
    else:
        age0 = float(mortality.age_start.value)

    if sample is not None and mortality.frac_male.name in sample:
        sex_mix = float(sample[mortality.frac_male.name])
    elif settings is not None and settings.get("sex_mix") is not None:
        sex_mix = settings["sex_mix"]
    else:
        sex_mix = float(mortality.frac_male.value)

    cycle_years = float(S.get("cycle_years", I.SETTINGS["cycle_years"]))
    return age0, sex_mix, cycle_years


# --------------------------------------------------------------------------- #
# 4. Layer 1 -- per-cycle rate bundle (FROZEN SIGNATURE)                       #
# --------------------------------------------------------------------------- #
def _cycle_rates(arm, cycle, *, p_base, rr, sample=None, settings=None) -> dict:
    """Every survivor-conditional rate governing the transition OUT OF `cycle`.

    THE SINGLE SOURCE OF TRUTH shared by `_build_P` (which turns it into a matrix) and by
    `run_arm`'s toll accounting (which turns the same onset hazards into the incidence
    flows that the cost tranches are attached to).

    Parameters
    ----------
    arm : 'A_standard' | 'B_cinpwt'
    cycle : int
        The cycle being left. Onset acts on the transitions out of cycles 0, 1, 2, so
        entrants occupy SUP_0/DEEP_0 at cycles 1, 2, 3 (BUILD-CONTRACT section 2 tunnel
        semantics). The onset hazard is ZERO for cycle >= 3.
    p_base : float
        Baseline (arm A) 90-day cumulative all-SSI risk. THE STUDY'S X-AXIS.
    rr : float | None
        ciNPWT relative risk. None = read `tp.rr_cinpwt` through `val()` so PSA can still
        override it; a float overrides. IGNORED for arm A, whose rr_used is always 1.0.
    sample : dict | None   PSA/DSA overrides by parameter name.
    settings : dict | None overrides for cycle_years / age0 / sex_mix.

    Returns
    -------
    dict with keys
        arm, cycle, age, cycle_years,
        q          total probability of dying during this cycle (background, plus
                   perioperative mortality in CYCLE 0 ONLY, combined as
                   1-(1-q_bg)(1-p_periop) -- both arms, identical)
        surv       1 - q
        rr_used    the RR actually applied (1.0 for arm A)
        p_arm      p_base * rr_used, the arm's 90-day cumulative onset risk
        h_total    conditional probability of SSI onset THIS cycle among survivors
        h_sup, h_deep   h_total split by frac_sup / frac_deep
        frac_sup, frac_deep  (renormalized defensively; a Dirichlet draw may not sum to 1)
        w_total    the blended onset weight used this cycle (0 outside the 90-day window)
        periop     the perioperative mortality applied this cycle (0 except cycle 0)

    ORDER OF OPERATIONS: background death is taken FIRST; `h_total` is then applied to the
    surviving mass. The onset weights are therefore conditional probabilities among
    survivors, and the cumulative 90-day onset RISK is exactly `p_arm`.
    """
    if arm not in I.ARMS:
        raise ValueError(f"arm must be one of {I.ARMS}, got {arm!r}")
    cycle = int(cycle)
    if cycle < 0:
        raise ValueError(f"cycle must be >= 0, got {cycle}")
    p_base = _check_p_base(p_base)   # post-review fix 6 -- validated in the SINGLE SOURCE
                                     # OF TRUTH, so transition_matrix() is covered too

    age0, sex_mix, cycle_years = _cohort(sample, settings)
    age = age0 + cycle * cycle_years

    # ---- mortality: background always; perioperative in cycle 0 only, both arms ----
    q_bg = I.background_mortality(age, sex_mix, t=cycle_years)
    periop = val(mortality.p_periop_death, sample) if cycle == 0 else 0.0
    q = 1.0 - (1.0 - q_bg) * (1.0 - periop)
    q = min(max(q, 0.0), 1.0)
    surv = 1.0 - q

    # ---- relative effect: correlated BY CONSTRUCTION (p_ciNPWT = p_control x RR) ----
    if arm == "B_cinpwt":
        rr_used = val(tp.rr_cinpwt, sample) if rr is None else float(rr)
    else:
        rr_used = 1.0
    p_arm = float(p_base) * rr_used

    # ---- destination split (applied AFTER the total hazard is set) ----
    f_deep = val(tp.frac_deep, sample)
    f_sup = val(tp.frac_sup, sample)
    tot = f_deep + f_sup
    if tot > 0.0:
        f_deep, f_sup = f_deep / tot, f_sup / tot
    else:                                     # degenerate draw: send everything deep
        f_deep, f_sup = 1.0, 0.0

    # ---- onset: CONDITIONAL-probability weights, zero from cycle 4 onward ----
    k = cycle  # 0-based index into the onset-weight triple
    if k < ONSET_CYCLES and p_arm > 0.0:
        # Blend the deep and superficial weight triples into ONE total-onset triple, so
        # the total hazard is set first and the destination split cannot perturb total
        # incidence. sum(w) == 1 because both triples sum to 1 and f_deep + f_sup == 1.
        w = [f_deep * val(d, sample) + f_sup * val(s, sample)
             for d, s in zip(_W_ONSET_DEEP, _W_ONSET_SUP)]
        w_total = w[k]
        cum_prev = p_arm * sum(w[:k])          # marginal onset risk already used up
        denom = 1.0 - cum_prev
        h_total = (p_arm * w_total / denom) if denom > 0.0 else 1.0
        h_total = min(max(h_total, 0.0), 1.0)
    else:
        w_total = 0.0
        h_total = 0.0

    return {
        "arm": arm,
        "cycle": cycle,
        "age": age,
        "cycle_years": cycle_years,
        "q": q,
        "surv": surv,
        "periop": periop,
        "rr_used": rr_used,
        "p_arm": p_arm,
        "h_total": h_total,
        "h_sup": h_total * f_sup,
        "h_deep": h_total * f_deep,
        "frac_sup": f_sup,
        "frac_deep": f_deep,
        "w_total": w_total,
    }


# --------------------------------------------------------------------------- #
# 5. Layer 2 -- transition matrix (FROZEN SIGNATURE)                           #
# --------------------------------------------------------------------------- #
def _build_P(arm, cycle, *, p_base, rr, sample=None, settings=None,
             n_sup=None, n_deep=None) -> np.ndarray:
    """Row-stochastic (n_slots, n_slots) transition matrix for the transition OUT OF
    `cycle`. Built entirely from one `_cycle_rates` bundle.

    Row construction (background death first, survivors distributed):
        WELL      -> DEEP_0 : s * h_deep      -> SUP_0 : s * h_sup
                  -> WELL   : s * (1 - h)     -> DEAD  : q
        SUP_i     -> SUP_{i+1} : s            (last SUP slot -> RECOV : s)   -> DEAD : q
        DEEP_i    -> DEEP_{i+1}: s            (last DEEP slot -> RECOV : s)  -> DEAD : q
        RECOV     -> RECOV : s                                                -> DEAD : q
        DEAD      -> DEAD  : 1                (absorbing)

    There is NO superficial -> deep progression (no rate exists in the evidence base;
    omitted by design and disclosed as a conservative simplification).

    Row sums are asserted to 1 within 1e-9 (BUILD-CONTRACT section 5).
    `n_sup` / `n_deep` are additive optional kwargs (default = the `inputs.py` widths);
    the frozen positional/keyword signature is unchanged.
    """
    L = _resolve_layout(n_sup, n_deep)
    n = L["n_slots"]
    well, recov, dead = L["well"], L["recov"], L["dead"]
    sup0, nsup = L["sup_start"], L["sup_len"]
    dep0, ndeep = L["deep_start"], L["deep_len"]

    r = _cycle_rates(arm, cycle, p_base=p_base, rr=rr, sample=sample, settings=settings)
    q, s = r["q"], r["surv"]

    P = np.zeros((n, n), dtype=float)

    # ---- WELL: death, then SSI onset among survivors, then stay well ----
    P[well, dead] = q
    P[well, dep0] = s * r["h_deep"]
    P[well, sup0] = s * r["h_sup"]
    P[well, well] = s * (1.0 - r["h_total"])
    P[well] = _norm_row(P[well])              # the only three-way split; normalize it

    # ---- tunnels: advance one slot per cycle; the last slot exits to RECOV ----
    for i in range(nsup):
        src = sup0 + i
        dst = (sup0 + i + 1) if i < nsup - 1 else recov
        P[src, dst] = s
        P[src, dead] = q
    for i in range(ndeep):
        src = dep0 + i
        dst = (dep0 + i + 1) if i < ndeep - 1 else recov
        P[src, dst] = s
        P[src, dead] = q

    # ---- RECOV: background mortality only ----
    P[recov, recov] = s
    P[recov, dead] = q

    # ---- DEAD: absorbing ----
    P[dead, dead] = 1.0

    # ---- numerical safety net: no negatives, every row sums to 1 ----
    # (the whole-matrix form of _norm_row; a no-op on a valid base case, and the guard
    #  that keeps an extreme PSA draw from producing a non-stochastic row)
    np.clip(P, 0.0, None, out=P)
    rs = P.sum(axis=1, keepdims=True)
    rs[rs == 0.0] = 1.0
    P /= rs

    assert np.all(np.abs(P.sum(axis=1) - 1.0) < 1e-9), \
        f"transition matrix rows must sum to 1 (arm={arm}, cycle={cycle})"
    return P


def transition_matrix(arm, cycle, *, p_base, rr, sample=None, settings=None,
                      n_sup=None, n_deep=None) -> np.ndarray:
    """Public alias for `_build_P`. Returns the row-stochastic (n_slots, n_slots) matrix
    governing the transition out of `cycle` for `arm` at baseline risk `p_base`."""
    return _build_P(arm, cycle, p_base=p_base, rr=rr, sample=sample,
                    settings=settings, n_sup=n_sup, n_deep=n_deep)


# --------------------------------------------------------------------------- #
# 6. Layer 3 -- cohort run for one arm (FROZEN SIGNATURE)                      #
# --------------------------------------------------------------------------- #
def run_arm(arm, *, p_base, horizon_cycles=24, device_cost=None, rr=None,
            half_cycle=True, disc_cost=0.03, disc_qaly=0.03,
            sample=None, settings=None, n_sup=None, n_deep=None) -> dict:
    """Trace one arm's cohort for `horizon_cycles` monthly cycles and accumulate costs,
    QALYs and event counts.

    Parameters
    ----------
    arm : 'A_standard' | 'B_cinpwt'
        A = contemporary prevention + standard dressing (comparator);
        B = contemporary prevention + ciNPWT (intervention).
    p_base : float (keyword-only, REQUIRED)
        Baseline 90-day cumulative all-SSI risk in the COMPARATOR arm. This is the
        study's x-axis, not a base case: arm B's risk is derived inside the engine as
        `p_base * RR`, so the arms are correlated by construction and arm ordering holds
        in every PSA draw. MUST lie in [0.0, 1.0] -- it is a cumulative risk, and
        anything else raises `ValueError` rather than running to a meaningless answer.
    horizon_cycles : int, default 24 (= 2 years; scenarios 3 / 12 / 60)
    device_cost : float | None
        ciNPWT unit cost in 2025 USD. None = read `cost.c_device_cinpwt` through
        `inputs.cost_2025()` so a PSA draw keyed 'c_device_cinpwt' can still override it;
        a float overrides both. THIS IS HOW THE PRICE SWEEP WORKS. Charged as a one-time
        cycle-0 toll in ARM B ONLY.
    rr : float | None
        ciNPWT relative risk. None = read `tp.rr_cinpwt` through `val()` (PSA can still
        override); a float overrides. IGNORED for arm A (its rr_used is always 1.0).
    half_cycle : bool, default True
        Trapezoidal correction hc[0] = hc[-1] = 0.5 on DWELL streams only.
        TOLLS ARE NEVER HALF-CYCLE WEIGHTED.
    disc_cost, disc_qaly : float, default 0.03
        ANNUAL discount rates; applied at monthly resolution as
        1 / (1 + r) ** (t * cycle_years).
    sample : dict | None
        PSA/DSA overrides keyed by `Param.name`. Cost draws are expressed in the
        parameter's OWN price year and are inflated exactly once inside `_cost()`.
        Two guards (post-review fix 5): `n_tunnel_sup` / `n_tunnel_deep` that DISAGREE
        with the resolved layout RAISE (they set nothing -- use `n_sup=` / `n_deep=`),
        and any key that is not an `inputs.all_params()` name emits a `UserWarning`
        because it would otherwise sweep nothing while reporting a zero swing.
    settings : dict | None
        Overrides for `toll_offset_cycles`, `cycle_years`, `age0`, `sex_mix`.
        Defaults to `inputs.SETTINGS`.
    n_sup, n_deep : int | None
        Tunnel widths (BUILD-CONTRACT section 2 amendment). None = the `inputs.py`
        base-case widths (3 and 12). Only run_dsa.py / run_scenarios.py should vary
        them, and traces of differing widths must be collapsed with `collapse_trace`
        before being compared.

    Returns
    -------
    dict with EXACTLY these keys (BUILD-CONTRACT section 5):
        arm, p_base, rr_used, device_cost_used, horizon_cycles,
        trace                 np.ndarray (T+1, n_slots); row 0 = the cycle-0 start state
        cost, qaly            DISCOUNTED totals (2025 USD / QALYs)
        cost_undisc, qaly_undisc
        life_years            undiscounted, half-cycle weighted, in YEARS
        n_ssi, n_ssi_superficial, n_ssi_deep
                              cumulative expected onset counts per 1 patient
        n_reop, n_persistent  COUNTERS ONLY -- these carry NO incremental cost
        by_cycle              pd.DataFrame: cycle, month, alive, cost, qaly, cost_disc,
                              qaly_disc, new_ssi, new_sup, new_deep, dead
        cost_stream, qaly_stream
                              np.ndarray (T+1,), DISCOUNTED and post-half-cycle; they sum
                              to `cost` and `qaly`
        share_cost_after_6mo, share_qaly_after_6mo
                              fraction of the DISCOUNTED total accruing strictly after
                              cycle 6 (the plan's data-coverage disclosure)

    Notes
    -----
    * `device_cost_used` is the resolved ciNPWT unit cost for the RUN (the sweep
      coordinate) and is reported identically for both arms; it is CHARGED only in arm B.
      Arm A instead pays the standard-dressing differential ($0 in the base case).
    * Per-arm `cost` is INCREMENTAL-VERSUS-NO-SSI cost. There is no index-surgery and no
      well-state cost, by design -- the index fusion, implant, ward stay and routine
      follow-up cancel exactly between arms.
    * `cost` is exactly linear in `device_cost` for arm B: the device toll lands in cycle
      0, where the discount factor is 1 and no half-cycle weight applies. So
      `cost_B(C) == cost_B(C0) + (C - C0)` to machine precision, and NOTHING else in the
      result depends on C. A price sweep can therefore run one pair per `p` and shift the
      cost (asserted in the smoke test).
    """
    if arm not in I.ARMS:
        raise ValueError(f"arm must be one of {I.ARMS}, got {arm!r}")
    T = int(horizon_cycles)
    if T < 1:
        raise ValueError(f"horizon_cycles must be >= 1, got {horizon_cycles}")
    p_base = _check_p_base(p_base)            # post-review fix 6

    S = I.SETTINGS if settings is None else settings
    L = _resolve_layout(n_sup, n_deep)
    _check_sample(sample, L)                  # post-review fix 5
    n_slots = L["n_slots"]
    well, recov, dead = L["well"], L["recov"], L["dead"]
    sup0, nsup = L["sup_start"], L["sup_len"]
    dep0, ndeep = L["deep_start"], L["deep_len"]

    age0, sex_mix, cycle_years = _cohort(sample, settings)
    offsets = tuple(int(o) for o in S.get("toll_offset_cycles",
                                          I.SETTINGS["toll_offset_cycles"]))

    # ---------------- unit costs (2025 USD), read ONLY through _cost() -------------- #
    m_ssi = val(cost.m_ssi_cost, sample)          # dimensionless attribution multiplier
    c_sup = [m_ssi * _cost(p, sample) for p in _C_SUP_TRANCHES]
    c_deep = [m_ssi * _cost(p, sample) for p in _C_DEEP_TRANCHES]
    device_cost_used = _cost(cost.c_device_cinpwt, sample, device_cost)
    dressing = _cost(cost.c_dressing_standard, sample)

    # ---------------- utilities: decrements are POSITIVE magnitudes ----------------- #
    u_well = val(util.u_well, sample)
    u = np.full(n_slots, u_well, dtype=float)
    u[sup0:sup0 + nsup] = u_well - val(util.du_sup, sample)
    u[dep0:dep0 + ndeep] = u_well - val(util.du_deep, sample)
    u[recov] = u_well - val(util.du_recovered_residual, sample)
    u[dead] = 0.0

    # ---------------- trace the cohort ---------------------------------------------- #
    trace = np.zeros((T + 1, n_slots), dtype=float)
    trace[0, well] = 1.0

    new_sup = np.zeros(T + 1, dtype=float)   # entrants OCCUPYING SUP_0  at cycle t
    new_deep = np.zeros(T + 1, dtype=float)  # entrants OCCUPYING DEEP_0 at cycle t

    # The matrix depends on `cycle` only through (a) the onset hazard, which is nonzero
    # for cycles 0..2 only, and (b) the integer cohort age. Cache by that key: a 24-cycle
    # run builds 5 matrices instead of 24, which is what keeps the 291 x 101 threshold
    # sweep fast.
    p_cache: dict = {}
    for t in range(T):
        if t < ONSET_CYCLES:
            key = ("onset", t)
        else:
            key = ("post", int(math.floor(age0 + t * cycle_years)))
        P = p_cache.get(key)
        if P is None:
            P = _build_P(arm, t, p_base=p_base, rr=rr, sample=sample, settings=settings,
                         n_sup=n_sup, n_deep=n_deep)
            p_cache[key] = P

        row = trace[t]
        if t < ONSET_CYCLES:
            # SAME rate bundle the matrix was built from -> the onset flow and the matrix
            # cannot drift apart. Entrants occupy SUP_0/DEEP_0 at cycle t+1.
            r = _cycle_rates(arm, t, p_base=p_base, rr=rr, sample=sample, settings=settings)
            m_well = row[well] * r["surv"]
            new_sup[t + 1] = m_well * r["h_sup"]
            new_deep[t + 1] = m_well * r["h_deep"]

        trace[t + 1] = row @ P

    # ---------------- structural invariants (BUILD-CONTRACT section 5) -------------- #
    row_sums = trace.sum(axis=1)
    assert np.all(np.abs(row_sums - 1.0) < 1e-8), \
        f"trace rows must sum to 1 (max dev {np.max(np.abs(row_sums - 1.0)):.2e})"
    assert np.all(trace >= -1e-12), "negative occupancy in the trace"
    dead_col = trace[:, dead]
    assert np.all(np.diff(dead_col) >= -1e-15), "trace[:, DEAD] must be non-decreasing"

    # ---------------- cost tolls (NEVER half-cycle weighted) ------------------------ #
    # Three MARGINAL Edmiston tranches attached to the ONSET COHORT, charged at
    # onset+0 / +6 / +12 cycles. A toll falling beyond the horizon is not charged --
    # which is exactly the per-horizon table in BUILD-CONTRACT section 6.2.
    toll_cost = np.zeros(T + 1, dtype=float)
    for k, off in enumerate(offsets):
        if off > T:
            continue                              # tranche falls beyond the horizon
        m = T + 1 - off
        toll_cost[off:] += new_sup[:m] * c_sup[k] + new_deep[:m] * c_deep[k]

    # One-time cycle-0 tolls: device in arm B, standard-dressing differential in arm A.
    if arm == "B_cinpwt":
        toll_cost[0] += device_cost_used
    else:
        toll_cost[0] += dressing

    # ---------------- dwell streams, half-cycle, discounting ------------------------ #
    dwell_qaly = (trace @ u) * cycle_years        # QALYs accrued while occupying a state
    # There is NO dwell cost in this model, by design (see the module docstring).

    hc = np.ones(T + 1, dtype=float)
    if half_cycle:
        hc[0] = 0.5
        hc[-1] = 0.5

    years = np.arange(T + 1, dtype=float) * cycle_years
    df_cost = (1.0 + disc_cost) ** (-years)
    df_qaly = (1.0 + disc_qaly) ** (-years)

    cost_stream = toll_cost * df_cost             # tolls: no half-cycle weight
    qaly_stream = dwell_qaly * hc * df_qaly       # dwell: trapezoidal weight

    cost_total = float(cost_stream.sum())
    qaly_total = float(qaly_stream.sum())
    cost_undisc = float(toll_cost.sum())
    qaly_undisc = float((dwell_qaly * hc).sum())

    alive = 1.0 - dead_col
    life_years = float((alive * hc).sum() * cycle_years)

    # ---------------- event counts -------------------------------------------------- #
    n_sup_ssi = float(new_sup.sum())
    n_deep_ssi = float(new_deep.sum())
    n_ssi = n_sup_ssi + n_deep_ssi
    # COUNTERS ONLY (plan Decision 3). Neither multiplies a cost anywhere in this engine:
    # Edmiston's payment is an ALL-CAUSE incremental spend that already contains
    # reoperation, readmission, I&D and antibiotics.
    n_reop = n_ssi * val(tp.p_reop_ssi, sample)              # reoperation among ALL SSI
    n_persistent = n_deep_ssi * val(tp.p_persistent_deep, sample)   # deep SSI only

    # ---------------- accrual disclosure (share strictly after cycle 6) ------------- #
    share_cost = (float(cost_stream[7:].sum() / cost_total)
                  if abs(cost_total) > 0.0 else 0.0)
    share_qaly = (float(qaly_stream[7:].sum() / qaly_total)
                  if abs(qaly_total) > 0.0 else 0.0)

    # ---------------- per-cycle table ------------------------------------------------ #
    cyc = np.arange(T + 1, dtype=np.int64)
    by_cycle = pd.DataFrame({
        "cycle": cyc,
        "month": cyc,                              # 1 cycle == 1 month
        "alive": alive,
        "cost": toll_cost,                         # undiscounted, sums to cost_undisc
        "qaly": dwell_qaly * hc,                   # undiscounted, sums to qaly_undisc
        "cost_disc": cost_stream,
        "qaly_disc": qaly_stream,
        "new_ssi": new_sup + new_deep,
        "new_sup": new_sup,
        "new_deep": new_deep,
        "dead": dead_col,
    })

    return {
        "arm": arm,
        "p_base": float(p_base),
        "rr_used": float(1.0 if arm == "A_standard"
                         else (val(tp.rr_cinpwt, sample) if rr is None else float(rr))),
        "device_cost_used": float(device_cost_used),
        "horizon_cycles": T,
        "trace": trace,
        "cost": cost_total,
        "qaly": qaly_total,
        "cost_undisc": cost_undisc,
        "qaly_undisc": qaly_undisc,
        "life_years": life_years,
        "n_ssi": n_ssi,
        "n_ssi_superficial": n_sup_ssi,
        "n_ssi_deep": n_deep_ssi,
        "n_reop": float(n_reop),
        "n_persistent": float(n_persistent),
        "by_cycle": by_cycle,
        "cost_stream": cost_stream,
        "qaly_stream": qaly_stream,
        "share_cost_after_6mo": share_cost,
        "share_qaly_after_6mo": share_qaly,
    }


def run_pair(*, p_base, **kw):
    """Run both arms at the same `p_base` and settings; return `(res_a, res_b)`.

    Accepts every `run_arm` keyword. Arm B's onset risk is derived INSIDE the engine as
    `p_base * RR`, so the pair is correlated by construction: whatever `sample` is
    passed, arm B's risk is always RR times arm A's and the arm ordering can never invert.

    Use with `incremental(res_b, res_a)` -- note the B-first argument order.
    """
    res_a = run_arm("A_standard", p_base=p_base, **kw)
    res_b = run_arm("B_cinpwt", p_base=p_base, **kw)
    return res_a, res_b


# --------------------------------------------------------------------------- #
# 7. Incremental analysis -- B (ciNPWT) MINUS A (standard)                     #
# --------------------------------------------------------------------------- #
def incremental(res_b: dict, res_a: dict, wtp: float = 100_000.0) -> dict:
    """Incremental cost-effectiveness of ciNPWT versus standard dressing, B MINUS A.

    Uses the DISCOUNTED `cost` / `qaly` from two `run_arm` results. NOTE THE ARGUMENT
    ORDER: the intervention comes first.

    Returns
    -------
    dict with keys
        inc_cost   res_b['cost'] - res_a['cost']
        inc_qaly   res_b['qaly'] - res_a['qaly']
        icer       inc_cost / inc_qaly; np.nan under STRICT DOMINANCE (either direction)
                   and when the QALY increment is numerically zero
        inmb       inc_qaly * wtp - inc_cost   (incremental net monetary benefit)
        dominance  one of
                     'ciNPWT dominant'                     (cheaper & better)
                     'ciNPWT dominated'                    (pricier & worse)
                     'ciNPWT more costly & more effective'
                     'ciNPWT less costly & less effective'
                     'equal'
        quadrant   'NE' | 'NW' | 'SE' | 'SW' | 'origin'   (Bouras "CE Quadrant" column,
                   cost on the x-axis and effect on the y-axis, so
                   NW = cheaper & better = dominant, SE = pricier & worse = dominated)

    `quadrant` is a pure function of `dominance` (`DOMINANCE_TO_QUADRANT`), so the two
    reported fields can never contradict each other -- including on the degenerate cases
    where exactly one increment is zero.
    """
    inc_cost = float(res_b["cost"] - res_a["cost"])
    inc_qaly = float(res_b["qaly"] - res_a["qaly"])
    inmb = float(inc_qaly * float(wtp) - inc_cost)

    eps = 1e-12
    cost_more = inc_cost > eps
    cost_less = inc_cost < -eps
    qaly_more = inc_qaly > eps
    qaly_less = inc_qaly < -eps

    if not (cost_more or cost_less or qaly_more or qaly_less):
        dominance = "equal"
    elif qaly_more and not cost_more:
        dominance = "ciNPWT dominant"                       # cheaper (or equal) & better
    elif qaly_less and not cost_less:
        dominance = "ciNPWT dominated"                      # pricier (or equal) & worse
    elif cost_more and qaly_more:
        dominance = "ciNPWT more costly & more effective"
    elif cost_less and qaly_less:
        dominance = "ciNPWT less costly & less effective"
    elif cost_less:                                         # cheaper, no QALY difference
        dominance = "ciNPWT dominant"
    else:                                                   # pricier, no QALY difference
        dominance = "ciNPWT dominated"

    if dominance in ("ciNPWT dominant", "ciNPWT dominated", "equal") or abs(inc_qaly) <= eps:
        icer = float("nan")
    else:
        icer = inc_cost / inc_qaly

    # `quadrant` is DERIVED FROM `dominance`, never re-derived from the raw signs, so the
    # two fields cannot disagree (post-review fix 1). See DOMINANCE_TO_QUADRANT for the
    # two degenerate zero cases the sign-based derivation used to get wrong.
    quadrant = DOMINANCE_TO_QUADRANT[dominance]

    assert dominance in DOMINANCE_VOCAB and quadrant in QUADRANT_VOCAB

    return {
        "inc_cost": inc_cost,
        "inc_qaly": inc_qaly,
        "icer": float(icer),
        "inmb": inmb,
        "dominance": dominance,
        "quadrant": quadrant,
    }


# --------------------------------------------------------------------------- #
# 8. Small public helpers (additive; the frozen API above is unchanged)        #
# --------------------------------------------------------------------------- #
def ssi_cost_per_case(sample=None, tranches=None, horizon_cycles=None,
                      settings=None) -> dict:
    """Per-case SSI cost in 2025 USD, by type and weighted -- the number the threshold
    arithmetic turns on.

    `tranches` optionally restricts to a subset of tranche indices (e.g. `(0,)` for a
    90-day horizon, `(0, 1)` for a 1-year horizon). `horizon_cycles` makes that selection
    from `tranches_charged()` instead of by hand -- and REFUSES a horizon that clips a
    tranche. Pass one or the other, never both.

    THE 'FULL'-ONLY CONVENTION IS VALID ONLY FOR HORIZONS WHERE NO TRANCHE IS CLIPPED
    (post-review fix 4, 2026-08-09). This function has no `partial` concept: it either
    charges a whole tranche or none of it. `run_arm`, by contrast, charges a clipped
    tranche to whichever onset cohorts reach it inside the horizon (`m = T + 1 - off`).
    The two therefore disagree on any horizon `T` with `1 + off <= T < ONSET_CYCLES + off`
    -- at T=7 and T=13 by roughly $8-10k per SSI. Rather than let a caller silently
    under-report, a clipped horizon RAISES. The four BUILD-CONTRACT horizons
    (3 / 12 / 24 / 60) are all clean, so nothing in this build is affected; the guard
    exists because `run_threshold.py` and `run_scenarios.py` were told to drive their
    tranche selection off `tranches_charged()`.

    Returns {'superficial', 'deep', 'weighted', 'frac_deep', 'frac_sup'}. Provided for
    `validation.py` and `make_tables.py`; `run_arm` computes its own tolls independently.
    """
    m = val(cost.m_ssi_cost, sample)
    f_deep = val(tp.frac_deep, sample)
    f_sup = val(tp.frac_sup, sample)
    tot = f_deep + f_sup
    f_deep, f_sup = (f_deep / tot, f_sup / tot) if tot > 0 else (1.0, 0.0)

    def _by_type(idx):
        return (float(sum(m * _cost(_C_SUP_TRANCHES[i], sample) for i in idx)),
                float(sum(m * _cost(_C_DEEP_TRANCHES[i], sample) for i in idx)))

    if horizon_cycles is not None:
        if tranches is not None:
            raise ValueError(
                "pass either `tranches` or `horizon_cycles` to ssi_cost_per_case, not "
                "both -- they are two ways of making the same selection"
            )
        sched = tranches_charged(horizon_cycles, settings=settings)
        clipped = [d for d in sched if d["charged"] == "partial"]
        idx = tuple(d["index"] for d in sched if d["charged"] == "full")
        if clipped:
            s_only, d_only = _by_type(idx)
            full_only = f_deep * d_only + f_sup * s_only
            spans = "; ".join(
                f"tranche {d['index'] + 1} (onset+{d['offset']}) spans cycles "
                f"{d['first_cycle']}-{d['last_cycle']}" for d in clipped
            )
            raise ValueError(
                f"horizon_cycles={int(horizon_cycles)} CLIPS a cost tranche mid-window "
                f"({spans}), so a per-case cost is not well defined here: this function "
                f"charges whole tranches only and would report "
                f"${full_only:,.0f}/SSI, while run_arm(horizon_cycles="
                f"{int(horizon_cycles)}) charges the clipped tranche to the onset "
                f"cohorts that reach it and so bills MORE. Use run_arm / run_pair for a "
                f"clipped horizon, or choose a horizon where tranches_charged() reports "
                f"no 'partial' (the contract horizons 3/12/24/60 are all clean)."
            )
    elif tranches is not None:
        idx = tuple(tranches)
    else:
        idx = tuple(range(len(_C_SUP_TRANCHES)))

    c_sup, c_deep = _by_type(idx)
    return {
        "superficial": float(c_sup),
        "deep": float(c_deep),
        "weighted": float(f_deep * c_deep + f_sup * c_sup),
        "frac_deep": float(f_deep),
        "frac_sup": float(f_sup),
    }


def tranches_charged(horizon_cycles: int, settings=None) -> list[dict]:
    """Which SSI cost tranches are charged at a given horizon (BUILD-CONTRACT 6.2).

    Onset sits at cycles 1..ONSET_CYCLES, so tranche k lands at cycles
    (1 + offset_k) .. (ONSET_CYCLES + offset_k). Returns one dict per tranche with
    `index`, `offset`, `first_cycle`, `last_cycle` and `charged` in
    {'full', 'partial', 'none'} -- rendered verbatim into `basecase_summary.md` so a
    reader cannot mistake the 90-day and 1-year numbers for errors.

    'PARTIAL' IS REAL AND MUST NOT BE DROPPED (post-review fix 4, 2026-08-09). A horizon
    with `first_cycle <= T < last_cycle` clips a tranche mid-window: `run_arm` charges it
    to the onset cohorts that reach it (`m = T + 1 - off`), so the common shortcut of
    keeping only `charged == "full"` UNDER-REPORTS the cost per SSI -- by roughly $8-10k
    at T=7 and T=13. That shortcut is valid ONLY for horizons where no tranche is
    clipped, which is true of all four contract horizons (3 / 12 / 24 / 60) and is the
    reason it is safe in this build. `ssi_cost_per_case(horizon_cycles=T)` enforces it
    (it raises on a clipped horizon) -- prefer it over filtering this list by hand.
    """
    S = I.SETTINGS if settings is None else settings
    offsets = tuple(int(o) for o in S.get("toll_offset_cycles",
                                          I.SETTINGS["toll_offset_cycles"]))
    T = int(horizon_cycles)
    out = []
    for k, off in enumerate(offsets):
        first, last = 1 + off, ONSET_CYCLES + off
        if last <= T:
            state = "full"
        elif first <= T:
            state = "partial"
        else:
            state = "none"
        out.append({"index": k, "offset": off, "first_cycle": first,
                    "last_cycle": last, "charged": state})
    return out


# --------------------------------------------------------------------------- #
# 9. Import-time structural checks                                             #
# --------------------------------------------------------------------------- #
def _check_layout_matches_inputs() -> None:
    """build_layout(3, 12) MUST reproduce the frozen 18-slot vector exactly."""
    L = build_layout(int(I.SUP_LEN), int(I.DEEP_LEN))
    assert L["n_slots"] == I.N_SLOTS == 18, (L["n_slots"], I.N_SLOTS)
    assert L["slot_names"] == list(I.SLOT_NAMES), "slot names diverge from inputs.py"
    assert (L["well"], L["sup_start"], L["sup_len"]) == (I.WELL, I.SUP_START, I.SUP_LEN)
    assert (L["deep_start"], L["deep_len"]) == (I.DEEP_START, I.DEEP_LEN)
    assert (L["recov"], L["dead"]) == (I.RECOV, I.DEAD)
    assert [int(x) for x in L["slot_to_clinical"]] == \
        [I.slot_to_clinical(s) for s in range(I.N_SLOTS)]
    # the base-case widths must equal the tunnel-length Params
    assert int(tp.n_tunnel_sup.value) == I.SUP_LEN
    assert int(tp.n_tunnel_deep.value) == I.DEEP_LEN
    # a different layout must still be internally consistent
    for ns, nd in ((2, 24), (6, 6), (4, 18)):
        M = build_layout(ns, nd)
        assert M["n_slots"] == 1 + ns + nd + 2
        assert M["slot_names"][0] == "WELL" and M["slot_names"][-1] == "DEAD"
        assert M["slot_names"][M["recov"]] == "RECOV"
        assert list(M["slot_to_clinical"]) == [0] + [1] * ns + [2] * nd + [3, 4]


_check_layout_matches_inputs()


# --------------------------------------------------------------------------- #
# 10. Smoke test (no files written)                                            #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    S = I.SETTINGS
    STRATA = S["strata"]
    P_INT = STRATA["intermediate"]

    print("=" * 78)
    print("model.py SMOKE TEST -- SSI/ciNPWT risk-threshold Markov CUA")
    print(I.RESULTS_BANNER)
    print("=" * 78)

    # ---- 1. layout builder ---------------------------------------------------------
    L = build_layout(3, 12)
    assert L["n_slots"] == I.N_SLOTS and L["slot_names"] == list(I.SLOT_NAMES)
    L2 = build_layout(2, 24)
    assert L2["n_slots"] == 29 and L2["slot_names"][L2["recov"]] == "RECOV"
    print(f"[OK] build_layout(3,12) reproduces the frozen {I.N_SLOTS}-slot vector; "
          f"build_layout(2,24) -> {L2['n_slots']} slots.")

    # ---- 2. transition matrices are row-stochastic ---------------------------------
    for arm in I.ARMS:
        for p in (0.005, P_INT, 0.15, 0.5):
            for t in range(0, 30):
                P = transition_matrix(arm, t, p_base=p, rr=None)
                assert P.shape == (I.N_SLOTS, I.N_SLOTS)
                assert np.all(np.abs(P.sum(axis=1) - 1.0) < 1e-9)
                assert (P >= -1e-15).all()
    assert transition_matrix("A_standard", 3, p_base=0.15, rr=None)[I.WELL, I.DEEP_START] == 0.0, \
        "onset hazard must be ZERO from cycle 3 onward (entrants at cycles 1-3 only)"
    print("[OK] all transition_matrix rows sum to 1 (both arms, cycles 0-29, p in "
          "{0.005, 0.066, 0.15, 0.5}); onset hazard is zero from cycle 4's transition on.")

    # ---- 3. THE trap: cumulative onset RISK reproduces p EXACTLY --------------------
    worst = 0.0
    for p in S["risk_grid"]:
        for arm in I.ARMS:
            surv_no_ssi = 1.0
            for t in range(ONSET_CYCLES):
                r = _cycle_rates(arm, t, p_base=float(p), rr=None)
                surv_no_ssi *= (1.0 - r["h_total"])
            cum = 1.0 - surv_no_ssi
            target = float(p) * (1.0 if arm == "A_standard" else float(tp.rr_cinpwt.value))
            worst = max(worst, abs(cum - target))
    assert worst < 1e-12, f"cumulative onset risk deviates from p by {worst:.3e}"
    print(f"[OK] cumulative 90-day onset RISK reproduces p exactly over the whole "
          f"{len(S['risk_grid'])}-point risk grid, both arms (max dev {worst:.2e} < 1e-12).")

    # ---- 4. cohort onset COUNT: attenuated only by competing mortality --------------
    dev_rel = []
    for p in (0.005, 0.015, P_INT, 0.114, 0.15):
        r = run_arm("A_standard", p_base=p, horizon_cycles=24)
        assert r["n_ssi"] <= p + 1e-15, "cohort count cannot exceed the onset risk"
        dev_rel.append((p - r["n_ssi"]) / p)
        assert abs(r["n_ssi"] - (r["n_ssi_superficial"] + r["n_ssi_deep"])) < 1e-15
    assert max(dev_rel) < 0.01, f"incidence attenuation {max(dev_rel):.4f} exceeds 1%"
    print(f"[OK] cohort onset COUNT sits {min(dev_rel)*100:.3f}-{max(dev_rel)*100:.3f}% "
          f"below p -- competing perioperative ({float(mortality.p_periop_death) * 100:.2f}%"
          f" cycle 0) + background mortality. "
          f"Documented behaviour, not a bug (run_log.d/inputs.md).")

    # ---- 5. trace integrity + absorption -------------------------------------------
    for arm in I.ARMS:
        for h in (3, 12, 24, 60):
            r = run_arm(arm, p_base=P_INT, horizon_cycles=h)
            tr = r["trace"]
            assert tr.shape == (h + 1, I.N_SLOTS)
            assert np.allclose(tr.sum(axis=1), 1.0, atol=1e-8)
            assert (tr >= -1e-12).all()
            assert np.all(np.diff(tr[:, I.DEAD]) >= -1e-15)
            # tunnels must have drained by the 24-cycle horizon
            if h >= 24:
                assert tr[-1, I.SUP_START:I.DEEP_START + I.DEEP_LEN].sum() < 1e-12
    print("[OK] trace rows sum to 1, occupancy non-negative, DEAD non-decreasing "
          "(both arms, horizons 3/12/24/60); tunnels drained by cycle 24.")

    # ---- 6. toll scheduling per horizon (BUILD-CONTRACT 6.2) -----------------------
    costs_by_h = {h: run_arm("A_standard", p_base=P_INT, horizon_cycles=h,
                             half_cycle=False, disc_cost=0.0, disc_qaly=0.0)["cost"]
                  for h in (3, 12, 24, 60)}
    per_case = ssi_cost_per_case()
    n_ssi_24 = run_arm("A_standard", p_base=P_INT, horizon_cycles=24,
                       half_cycle=False, disc_cost=0.0, disc_qaly=0.0)["n_ssi"]
    # horizon_cycles= (not a hand-rolled 'full'-only filter) so a clipped horizon would
    # raise instead of quietly under-reporting -- post-review fix 4.
    exp = {h: n_ssi_24 * ssi_cost_per_case(horizon_cycles=h)["weighted"]
           for h in (3, 12, 24, 60)}
    for h in (3, 12, 24, 60):
        assert abs(costs_by_h[h] - exp[h]) < 1e-6, (h, costs_by_h[h], exp[h])
    assert costs_by_h[3] < costs_by_h[12] < costs_by_h[24] == costs_by_h[60]
    print(f"[OK] toll scheduling matches the section-6.2 table exactly: undiscounted "
          f"arm-A cost at p={P_INT} is "
          f"{'/'.join(f'${costs_by_h[h]:,.0f}' for h in (3, 12, 24, 60))} "
          f"at horizons 3/12/24/60 (tranche 2 IS charged at 12 cycles).")

    # ---- 7. weighted 24-month cost per SSI (BUILD-CONTRACT 10) ---------------------
    # The three-tranche sums come from the Params, NOT from literals (post-review fix 3):
    # hard-coding 93,741 / 32,821 made this assertion compare two literals and it would
    # have gone on passing after an A2 revision of Edmiston. 78,206 (2019 USD) and
    # 91,024 (2025 USD) stay literal ON PURPOSE -- they are BUILD-CONTRACT section-10
    # cross-check TARGETS, not parameter values, and deriving them from inputs.py would
    # make the check tautological.
    #
    # A2 UPDATE 2026-08-17 -- recomputed, not relaxed. The A2 audit moved the SSI split
    # from 0.742/0.258 (rounded rates 4.9/6.6) to 0.745/0.255 (counts 10,296/3,517 over
    # 13,813), which is the basis Edmiston's own prose uses ("75% were deep-incisional").
    # The six commercial tranches are unchanged, so the whole move is the split:
    #     old  0.742 x 93,741 + 0.258 x 32,821 = 69,555.8 +  8,467.8 = $78,023.6
    #     new  0.745 x 93,741 + 0.255 x 32,821 = 69,837.0 +  8,369.4 = $78,206.4
    #     delta +$182.8 (+0.23%)
    # Inflating on the same medical-care CPI factor (580.102 / 498.413 = 1.163898):
    #     78,206.4 x 1.163898 = $91,024.3   (was 78,023.6 x 1.163898 = $90,811.6)
    deep_2019 = sum(float(p) for p in _C_DEEP_TRANCHES)
    sup_2019 = sum(float(p) for p in _C_SUP_TRANCHES)
    w2019 = float(tp.frac_deep) * deep_2019 + float(tp.frac_sup) * sup_2019
    w2025 = per_case["weighted"]
    # TWO BASES, AND THEY MUST NOT BE CONFLATED.
    #   w2019 is read straight off the Params, so it is Edmiston's ALL-CAUSE incremental
    #   payment per SSI on a 2019-USD basis: $78,206.
    #   w2025 comes through ssi_cost_per_case(), which applies BOTH the medical-care CPI
    #   factor AND m_ssi_cost -- the attribution multiplier that A2 moved from 1.000
    #   (Edmiston's all-cause increment taken at face value) to 0.492 (Pennington's
    #   matched increment, same-basis). So w2025 is the model's SSI-ATTRIBUTABLE cost:
    #       78,206.4 x 1.163898 = $91,024.3  all-cause, 2025 USD
    #       91,024.3 x 0.492    = $44,784.0  attributable, 2025 USD  <- what the engine charges
    #   Both are asserted, because a reader who sees only one of them will misread the
    #   model: the headline threshold is driven by $44,784, not by $91,024.
    m_attrib = float(cost.m_ssi_cost)
    w2025_allcause = w2025 / m_attrib
    assert abs(w2019 - 78206.0) <= 50.0, w2019
    assert abs(w2025_allcause - 91024.0) <= 100.0, w2025_allcause
    assert abs(w2025 - 44784.0) <= 100.0, w2025
    print(f"[OK] weighted 24-month cost per SSI = ${w2019:,.0f} (2019 USD all-cause, "
          f"target $78,206 +/-50) -> ${w2025_allcause:,.0f} (2025 USD all-cause, target "
          f"$91,024 +/-100) -> ${w2025:,.0f} (2025 USD ATTRIBUTABLE at "
          f"m_ssi_cost={m_attrib:.3f}, target $44,784 +/-100).")

    # ---- 8. counters carry no cost -------------------------------------------------
    base = run_arm("A_standard", p_base=P_INT, horizon_cycles=24)
    bumped = run_arm("A_standard", p_base=P_INT, horizon_cycles=24,
                     sample={"p_reop_ssi": 0.9, "p_persistent_deep": 0.9,
                             "n_debridements": 5.0, "rr_readmit_sa": 9.0})
    assert abs(bumped["cost"] - base["cost"]) < 1e-12, \
        "a counter parameter changed a cost -- Decision 3 violated (double counting)"
    assert abs(bumped["qaly"] - base["qaly"]) < 1e-12
    assert bumped["n_reop"] > base["n_reop"]
    print("[OK] Decision 3 holds: p_reop_ssi / p_persistent_deep / n_debridements / "
          "rr_readmit_sa move the COUNTERS and nothing else.")
    assert abs(base["n_reop"] / base["n_ssi"] - float(tp.p_reop_ssi)) < 1e-12
    print(f"[OK] reoperation among SSI patients = {base['n_reop']/base['n_ssi']:.3f} "
          f"(Karamian [20] 0.286, tol 0.05).")

    # ---- 9. device cost is a pure additive cycle-0 toll ----------------------------
    b0 = run_arm("B_cinpwt", p_base=P_INT, horizon_cycles=24)
    b1 = run_arm("B_cinpwt", p_base=P_INT, horizon_cycles=24, device_cost=600.0)
    assert abs((b1["cost"] - b0["cost"]) - (600.0 - b0["device_cost_used"])) < 1e-9
    assert np.allclose(b1["trace"], b0["trace"]) and abs(b1["qaly"] - b0["qaly"]) < 1e-15
    a0 = run_arm("A_standard", p_base=P_INT, horizon_cycles=24)
    a1 = run_arm("A_standard", p_base=P_INT, horizon_cycles=24, device_cost=600.0)
    assert abs(a1["cost"] - a0["cost"]) < 1e-15, "arm A must never pay the device cost"
    print("[OK] device cost is an exact additive cycle-0 toll in arm B only; the trace, "
          "QALYs and SSI tolls do not depend on it (a price sweep can shift, not re-run).")

    # ---- 10. val() / sample override hook ------------------------------------------
    assert abs(val(tp.rr_cinpwt, {"rr_cinpwt": 0.4}) - 0.4) < 1e-15
    Pb = transition_matrix("B_cinpwt", 0, p_base=P_INT, rr=None)
    Pb2 = transition_matrix("B_cinpwt", 0, p_base=P_INT, rr=None, sample={"rr_cinpwt": 0.4})
    assert not np.allclose(Pb, Pb2), "sample override had no effect on the matrix"
    r_ov = run_arm("A_standard", p_base=P_INT, horizon_cycles=24,
                   sample={"c_ssi_deep_0_6mo": 2.0 * float(cost.c_ssi_deep_0_6mo)})
    assert r_ov["cost"] > base["cost"], "cost PSA override had no effect"
    assert abs(_cost(cost.c_device_cinpwt) - 250.0) < 1e-12, \
        "a 2025-USD parameter was inflated -- price-year trap"
    print("[OK] val()/_cost() PSA override hooks work; cost_2025(c_device_cinpwt) = "
          f"{_cost(cost.c_device_cinpwt):.1f} (NOT inflated).")

    # ---- 10b. post-review guard rails (2026-08-09) ---------------------------------
    # (1) fix 1 -- `quadrant` is derived FROM `dominance`, so they cannot disagree. The
    #     two degenerate zero cases the old sign-based derivation got wrong:
    deg_a = incremental({"cost": 100.0, "qaly": 1.0}, {"cost": 150.0, "qaly": 1.0})
    assert deg_a["inc_cost"] < 0.0 and deg_a["inc_qaly"] == 0.0
    assert (deg_a["dominance"], deg_a["quadrant"]) == ("ciNPWT dominant", "NW"), deg_a
    deg_b = incremental({"cost": 100.0, "qaly": 1.0}, {"cost": 100.0, "qaly": 1.5})
    assert deg_b["inc_cost"] == 0.0 and deg_b["inc_qaly"] < 0.0
    assert (deg_b["dominance"], deg_b["quadrant"]) == ("ciNPWT dominated", "SE"), deg_b
    # ... and the four non-degenerate corners still map as before
    for (cb, qb), want in (((150.0, 1.5), ("ciNPWT more costly & more effective", "NE")),
                           ((50.0, 0.5), ("ciNPWT less costly & less effective", "SW")),
                           ((50.0, 1.5), ("ciNPWT dominant", "NW")),
                           ((150.0, 0.5), ("ciNPWT dominated", "SE")),
                           ((100.0, 1.0), ("equal", "origin"))):
        got = incremental({"cost": cb, "qaly": qb}, {"cost": 100.0, "qaly": 1.0})
        assert (got["dominance"], got["quadrant"]) == want, (cb, qb, got)
        assert got["quadrant"] == DOMINANCE_TO_QUADRANT[got["dominance"]]
    print("[OK] incremental(): `quadrant` is a pure function of `dominance` -- the two "
          "degenerate zero cases now give (dominant, NW) and (dominated, SE).")

    # (2) fix 5 -- inert `sample` keys cannot pass silently
    try:
        run_arm("A_standard", p_base=P_INT, sample={"n_tunnel_deep": I.DEEP_LEN + 12})
    except ValueError as exc:
        assert "n_deep" in str(exc), exc
    else:
        raise AssertionError("sample['n_tunnel_deep'] disagreeing with the layout must "
                             "raise, not be silently ignored")
    try:
        run_arm("A_standard", p_base=P_INT, sample={"n_tunnel_sup": I.SUP_LEN + 3})
    except ValueError as exc:
        assert "n_sup" in str(exc), exc
    else:
        raise AssertionError("sample['n_tunnel_sup'] disagreeing with the layout must raise")
    # a value that AGREES with the resolved layout is harmless and must NOT raise
    run_arm("A_standard", p_base=P_INT,
            sample={"n_tunnel_sup": I.SUP_LEN, "n_tunnel_deep": I.DEEP_LEN})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_arm("A_standard", p_base=P_INT, sample={"NOT_A_PARAM": 99.0})
    assert any("NOT_A_PARAM" in str(c.message) for c in caught), \
        "an unknown sample key must warn (a typo in a PSA/DSA driver sweeps nothing)"
    with warnings.catch_warnings(record=True) as caught_ok:
        warnings.simplefilter("always")
        run_arm("A_standard", p_base=P_INT, sample={"rr_cinpwt": 0.5, "u_well": 0.7})
    assert not [c for c in caught_ok if "sample key" in str(c.message)], \
        "a correctly-spelled sample must not warn"
    print("[OK] sample guards: n_tunnel_sup/n_tunnel_deep disagreeing with the layout "
          "RAISE (use n_sup=/n_deep=); an unknown key WARNS instead of sweeping nothing.")

    # (3) fix 6 -- an out-of-range p_base is a hard error, not n_ssi = 0.0
    for bad in (-0.05, 1.5, float("nan")):
        try:
            run_arm("A_standard", p_base=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"p_base={bad} must raise ValueError")
        try:
            transition_matrix("A_standard", 0, p_base=bad, rr=None)
        except ValueError:
            pass
        else:
            raise AssertionError(f"transition_matrix with p_base={bad} must raise")
    for good in (0.0, 1.0, P_INT):
        run_arm("A_standard", p_base=good, horizon_cycles=3)
    print("[OK] p_base is validated as a probability: -0.05 / 1.5 / NaN raise "
          "ValueError; 0.0 and 1.0 are accepted.")

    # (4) fix 4 -- a horizon that clips a tranche can no longer be answered with the
    #     'full'-only convention. The four contract horizons are clean; 7 and 13 are not.
    for h in (3, 12, 24, 60):
        assert not [d for d in tranches_charged(h) if d["charged"] == "partial"], h
        by_h = ssi_cost_per_case(horizon_cycles=h)["weighted"]
        by_idx = ssi_cost_per_case(
            tranches=[d["index"] for d in tranches_charged(h) if d["charged"] == "full"]
        )["weighted"]
        assert abs(by_h - by_idx) < 1e-9, (h, by_h, by_idx)
    for h in (7, 13):
        assert [d for d in tranches_charged(h) if d["charged"] == "partial"], h
        try:
            ssi_cost_per_case(horizon_cycles=h)
        except ValueError as exc:
            assert "CLIPS" in str(exc), exc
        else:
            raise AssertionError(f"a clipped horizon (T={h}) must not silently return a "
                                 "full-only cost per SSI")
    print("[OK] ssi_cost_per_case(horizon_cycles=) raises on a clipped horizon (T=7, "
          "T=13) and matches the 'full'-only selection at the clean contract horizons "
          "3/12/24/60.")

    # ---- 11. NNT reproduction (BUILD-CONTRACT 10) ----------------------------------
    print("\n" + "-" * 78)
    print("SANITY TARGETS")
    print("-" * 78)
    # A2 UPDATE 2026-08-17. The second target was pinned at p = 0.135, a control rate this
    # build BACK-SOLVED from Zwanenburg's printed NNT of 19 at RR 0.61 (1/(19 x 0.39) =
    # 0.13495). Zwanenburg actually REPORTS its pooled control rate -- 14.3% (315/2205),
    # in the text, in Table 2, in Table 5 and in Fig. 2 -- and the string "13.5" appears
    # nowhere in the paper. The target is therefore re-derived at the reported rate:
    #     NNT = 1 / (p x (1 - RR)) = 1 / (0.143 x 0.39) = 17.93
    # Note this model cannot reproduce Zwanenburg's printed 19 at its own reported control
    # rate, and that is not an error in either: Zwanenburg computes NNT as
    # 1/(control risk - intervention risk) on CRUDE arm risks and rounds UP (18.38 -> 19),
    # a method that reproduces all four of its printed NNTs, whereas this model applies
    # the pooled RR multiplicatively. The gap is crude-vs-pooled RR and is disclosed
    # rather than engineered away by keeping a control rate the source contradicts.
    for p, want in ((0.066, 38.9), (0.143, 17.9)):
        ra = run_arm("A_standard", p_base=p, horizon_cycles=24, half_cycle=False,
                     disc_cost=0.0, disc_qaly=0.0)
        rb = run_arm("B_cinpwt", p_base=p, horizon_cycles=24, half_cycle=False,
                     disc_cost=0.0, disc_qaly=0.0)
        nnt = 1.0 / (ra["n_ssi"] - rb["n_ssi"])
        assert abs(nnt - want) < 0.15, (p, nnt, want)
        print(f"  NNT at p={p:.3f}, RR={float(tp.rr_cinpwt):.2f} : {nnt:6.2f}  "
              f"(target {want}, tol 0.15)   [PASS]")

    # ---- 12. threshold p* ----------------------------------------------------------
    from scipy.optimize import brentq

    def _inmb(p, **kw):
        ra = run_arm("A_standard", p_base=p, horizon_cycles=24, half_cycle=False,
                     disc_cost=0.0, disc_qaly=0.0, **kw)
        rb = run_arm("B_cinpwt", p_base=p, horizon_cycles=24, half_cycle=False,
                     disc_cost=0.0, disc_qaly=0.0, **kw)
        return incremental(rb, ra, wtp=S["wtp_primary"])["inmb"]

    p_star = brentq(_inmb, 1e-6, 0.5, xtol=1e-12)
    dq = (float(tp.frac_deep) * float(util.du_deep) * (I.DEEP_LEN / 12.0)
          + float(tp.frac_sup) * float(util.du_sup) * (I.SUP_LEN / 12.0))
    hand_2025 = float(cost.c_device_cinpwt.value) / (
        (1.0 - float(tp.rr_cinpwt)) * (w2025 + S["wtp_primary"] * dq))
    hand_mixed = float(cost.c_device_cinpwt.value) / (
        (1.0 - float(tp.rr_cinpwt)) * (w2019 + S["wtp_primary"] * dq))
    print(f"  QALY loss per SSI                       : {dq:.5f}  (plan 0.0794)")
    print(f"  Engine threshold p* (undisc, no HCC)    : {p_star*100:.4f}%")
    print(f"  Hand-calc, ALL 2025 USD                 : {hand_2025*100:.4f}%   <- consistent")
    print(f"  Hand-calc on the CONTRACT's mixed basis  : {hand_mixed*100:.4f}%   "
          f"<- SSI cost in 2019 USD, device in 2025 USD")
    assert abs(p_star - hand_2025) / hand_2025 < 0.01, (p_star, hand_2025)
    print("  [PASS] the engine reproduces the PRICE-YEAR-CONSISTENT hand calculation to "
          "<1% relative.")
    print("  [FLAG] the contract's original 0.746% mixes price years; see "
          "run_log.d/model.md 'Objection 1'.")

    # ---- 13. engine summary --------------------------------------------------------
    print("\n" + "-" * 78)
    print("ENGINE SUMMARY (canonical results come from run_basecase.py)")
    print("-" * 78)
    print(f"{'stratum':<14}{'p':>7}{'cost A':>11}{'cost B':>11}{'inc cost':>11}"
          f"{'inc QALY':>11}{'INMB@100k':>12}  dominance")
    for name, p in STRATA.items():
        ra, rb = run_pair(p_base=p, horizon_cycles=24)
        inc = incremental(rb, ra, wtp=S["wtp_primary"])
        print(f"{name:<14}{p:>7.3f}{ra['cost']:>11,.0f}{rb['cost']:>11,.0f}"
              f"{inc['inc_cost']:>11,.0f}{inc['inc_qaly']:>11.5f}"
              f"{inc['inmb']:>12,.0f}  {inc['dominance']} ({inc['quadrant']})")

    print("\nSMOKE TEST PASSED")
