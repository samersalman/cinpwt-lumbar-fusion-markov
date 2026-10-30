"""
inputs.py -- Single source of truth for every number in this build.

Literature-Based Markov Cost-Utility Analysis (CUA):
At what predicted 90-day SSI risk does prophylactic closed-incision negative-pressure
wound therapy (ciNPWT) pay off after instrumented posterior lumbar fusion, under
contemporary prevention that already includes universal S. aureus decolonization?

Arms        : A_standard = contemporary prevention + standard dressing (comparator)
              B_cinpwt   = contemporary prevention + ciNPWT (intervention)
Perspective : US commercial payer, payments basis, one consistent basis. 2025 USD.
Cycle       : 1 month = 1/12 year. Trapezoidal half-cycle correction on DWELL streams
              only -- tolls are point events and are attributed whole to their cycle.
Horizon     : 24 cycles (2 years) base case; 3 / 12 / 60-cycle scenarios.
Discounting : 3%/yr on costs and QALYs (0% and 5% scenarios), applied at monthly
              resolution as disc_t = 1 / (1 + r) ** (t * cycle_years).
WTP         : $100,000/QALY primary; $50,000 and $150,000 also reported;
              CEAC grid np.arange(0, 200_001, 2_500).

=============================================================================
GATE STATUS: VERIFIED -- Phase A2 full-text verification completed 2026-08-17.
=============================================================================
All 14 requested full texts (5 Tier-1, 9 Tier-2) were opened and every number was
checked against the source. 46 of the 48 parameters are now closed out; 43 of those
carry a `source_locator` naming the table/figure and PDF page the value was read from,
and the other 3 are scenario constants or Tier-3 parameters no source was requested for.
Only 2 parameters retain `verify=True` -- the ciNPWT institutional net acquisition price
and the superficial tunnel length -- because no available source can resolve either. Every correction was re-checked by an independent agent instructed to
REFUTE it; four proposed corrections were defeated and the original values stand
(see `v2/AUDIT-2026-08-17.md` section C). The completed worksheet is
`v2/verification-notes.md`. Every results CSV carries `RESULTS_BANNER` as line 1 and
every .md summary opens with the same statement.

RESIDUAL ASSUMPTIONS, stated plainly: the ciNPWT institutional net acquisition price
(no source reports one -- Mueller 2026 [9] gives $495/$830 LIST, carried as an axis);
the standard-dressing cost ($0); the superficial tunnel length (3 cycles); the
attenuated-RR scenario constant; and the decolonization bundle cost (Kline 2018 [1],
no full text).

-----------------------------------------------------------------------------
THE THREE ANALYSIS AXES (this is what the paper reports -- see SETTINGS)
-----------------------------------------------------------------------------
  baseline 90-day all-SSI risk `p`   0.005 -> 0.15   291 pts  SETTINGS["risk_grid"]
  ciNPWT unit cost `C`               $100  -> $900   161 pts  SETTINGS["price_grid"]
  ciNPWT relative risk `RR`          0.40  -> 1.00    61 pts  SETTINGS["rr_grid"]

`tp.p_base_ssi` is an AXIS, not a base case. Its `value` field exists only so that
`run_arm()` has a default sweep position; no headline result is reported at a single
value of it.

-----------------------------------------------------------------------------
PRICE-YEAR SPLIT RULE -- read this before touching any cost
-----------------------------------------------------------------------------
Costs live in TWO price years and mixing them silently corrupts every result.

  * The TWELVE Edmiston-derived cost tranches (six commercial + six Medicare) are
    stored in **2019 USD** and carry `price_year=2019`. They MUST be inflated at read
    time (BLS medical-care CPI, 2019 -> 2025 factor 1.163898 read live from
    `data/cpi_medical.csv`; never hard-coded).
  * `c_device_cinpwt`, `c_dressing_standard` and `c_decol_bundle` are ALREADY in
    **2025 USD** and carry `price_year=2025`. They MUST NOT be inflated.
  * `m_ssi_cost` and `k_attrib_pennington` are dimensionless multipliers and carry
    `price_year=None`.

Do NOT rely on a naming convention or a comment to tell these apart. Read every cost
through `cost_2025(param)` (or `cost_2025(param, value=<PSA draw>)`), which inflates
if and only if `param.price_year != PRICE_YEAR`. Import-time assertions below check
that exactly 12 costs are 2019 USD and exactly 3 are 2025 USD.

-----------------------------------------------------------------------------
FOR model.py -- three structural rules that are easy to get wrong
-----------------------------------------------------------------------------
1. ONSET WEIGHTS ARE CONDITIONAL-PROBABILITY INPUTS, NOT PER-CYCLE HAZARDS.
   `w_onset_*_m1/m2/m3` distribute the whole 90-day risk `p` across the first three
   cycles. The engine must apply them CONDITIONALLY so the 90-day cumulative equals
   `p` exactly:

       h_1 = p * w_1
       h_2 = p * w_2 / (1 - p * w_1)
       h_3 = p * w_3 / (1 - p * w_1 - p * w_2)

   Then new SSI in cycle k = p * w_k and the three sum to p. Using `w_k` directly as
   the cycle hazard undershoots `p` by roughly 3-4% at the top of the risk range.
   The deep/superficial split (`frac_deep` / `frac_sup`) is applied AFTER the total
   hazard is set, so the destination split cannot perturb total incidence. The hazard
   is ZERO from cycle 4 onward -- the pivotal hazard needs no extrapolation.

2. SSI COST IS A MASTER HAZARD, CHARGED AS THREE MARGINAL TOLLS.
   Edmiston's incremental payment is all-cause spend over the window and already
   contains readmission, I&D, revision and antibiotics. Never add Blumberg's $16,242,
   Rosenthal's RR 2.15 or Freire-Archer's 1.4 debridements on top. The six stored cost
   values are MARGINAL tranches, already differenced -- do not difference them again
   and do not charge the cumulative figures. Tolls fall at onset, onset+6 and onset+12
   cycles (`SETTINGS["toll_offset_cycles"]`); a toll beyond the horizon is not charged.
   `p_persistent_deep`, `p_reop_ssi`, `n_debridements` and `rr_readmit_sa` are COUNTERS
   / CROSS-CHECKS ONLY and carry NO incremental cost (plan Decision 3).

3. background_mortality(age, sex_mix, t) TAKES `t` IN YEARS.
   Monthly cycles pass `t=1/12`. The conversion is `1 - (1 - qx) ** t`, never `qx * t`
   -- the linear form is ~0.5% relative off per cycle at age 60 and compounds over all
   24 cycles.

There is NO index-surgery cost and NO well-state cost, by design: the model is purely
incremental, so the index fusion, implant, ward stay and routine follow-up cancel
exactly between arms. Per-arm "cost" is incremental-versus-no-SSI cost, not total cost
of care. Do not invent one.

-----------------------------------------------------------------------------
PSA DISTRIBUTION CONVENTION (consumed by run_psa.py)
-----------------------------------------------------------------------------
    'beta'      probabilities and utilities      fit from (value, low, high) as 95% CI
    'gamma'     costs                            fit from (value, low, high) as 95% CI
    'lognormal' relative effects                 median-preserving, sampled on log scale
    'normal'    utility DECREMENTS               fit from (value, low, high) as 95% CI
    'dirichlet' destination splits               grouped by `group`
    'fixed'     not sampled in PSA
Method of moments treats [low, high] as an approximate 95% CI, SE = (high - low)/3.92,
preserving the mean. The ciNPWT relative effect is correlated BY CONSTRUCTION inside the
engine (p_ciNPWT = p_control x RR), so arm ordering holds in every draw. The `ssi_split`
Dirichlet pair uses the DOWN-WEIGHTED concentration in
`SETTINGS["dirichlet_kappa_ssi_split"]`, not the raw Edmiston counts.

Utility decrements (`du_sup`, `du_deep`, `du_recovered_residual`) are stored as POSITIVE
MAGNITUDES and are SUBTRACTED from `u_well` inside the engine. Storing them negative
inverts the DSA and PSA sign conventions.

This module performs no model computation (that lives in model.py). Downstream scripts
depend on the names and signatures here -- keep them STABLE.

See: v2/BUILD-CONTRACT.md (frozen interface), v2/proposed-inputs.md (derivations and
citations), v2/run_log.d/phase-a1.md (parameter specification), v2/run_log.d/inputs.md
(this module's own run-log fragment).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# 0. Paths and price year                                                      #
# --------------------------------------------------------------------------- #
# Scripts are run with cwd = v2/analysis/, but never rely on that: resolve every
# path from this file's own location (BUILD-CONTRACT section 1).
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V1_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
DATA_DIR = os.path.join(THIS_DIR, "data")
RESULTS_DIR = os.path.join(V1_DIR, "results")
FIGURES_DIR = os.path.join(V1_DIR, "figures")
TABLES_DIR = os.path.join(V1_DIR, "tables")

CPI_CSV = os.path.join(DATA_DIR, "cpi_medical.csv")
LIFE_TABLE_CSV = os.path.join(DATA_DIR, "life_table_2023.csv")

#: All model costs are reported in this price year (BUILD-CONTRACT section 3).
PRICE_YEAR = 2025

#: Price year of the Edmiston 2023 [12] payment figures as published.
EDMISTON_PRICE_YEAR = 2019


# --------------------------------------------------------------------------- #
# 1. Arms                                                                      #
# --------------------------------------------------------------------------- #
ARMS = ["A_standard", "B_cinpwt"]                     # A = comparator, B = intervention
ARM_LABELS = {
    "A_standard": "Contemporary prevention + standard dressing",
    "B_cinpwt": "Contemporary prevention + ciNPWT",
}
#: Both arms receive universal S. aureus decolonization, so it cancels from the
#: base-case incremental and survives only as a secondary analysis.


# --------------------------------------------------------------------------- #
# 2. The 18-slot engine state vector (FROZEN, BUILD-CONTRACT section 2)        #
# --------------------------------------------------------------------------- #
# slot 0      WELL              Well (post-fusion, no SSI)
# slots 1-3   SUP_0  .. SUP_2   Superficial SSI, months 1-3 of a 3-month tunnel
# slots 4-15  DEEP_0 .. DEEP_11 Deep SSI, months 1-12 of a 12-month tunnel
# slot 16     RECOV             Recovered post-SSI (kept distinct from WELL so the
#                               residual-decrement scenario has somewhere to live)
# slot 17     DEAD              Death (absorbing)
#
# Tunnel semantics: a cohort entering the superficial tunnel at cycle t occupies
# SUP_0 at t+1, advances one slot per cycle, and leaves the last slot to RECOV net of
# mortality. Onset is confined to the first 90 days, so entrants occupy SUP_0/DEEP_0
# at cycles 1-3 -- which is why the cost tolls land at cycles 1-3, 7-9 and 13-15.
WELL = 0
SUP_START = 1
SUP_LEN = 3
DEEP_START = 4
DEEP_LEN = 12
RECOV = 16
DEAD = 17
N_SLOTS = 18

SLOT_NAMES: list[str] = (
    ["WELL"]
    + [f"SUP_{i}" for i in range(SUP_LEN)]
    + [f"DEEP_{i}" for i in range(DEEP_LEN)]
    + ["RECOV", "DEAD"]
)

#: The 5 reported clinical states the 18 slots collapse to.
CLINICAL_STATES = ["Well", "Superficial SSI", "Deep SSI", "Recovered", "Dead"]


def slot_to_clinical(slot: int) -> int:
    """Map an engine slot index (0..17) to a CLINICAL_STATES index (0..4)."""
    s = int(slot)
    if s == WELL:
        return 0
    if SUP_START <= s < SUP_START + SUP_LEN:
        return 1
    if DEEP_START <= s < DEEP_START + DEEP_LEN:
        return 2
    if s == RECOV:
        return 3
    if s == DEAD:
        return 4
    raise ValueError(f"slot {slot} is outside the {N_SLOTS}-slot engine vector")


# --------------------------------------------------------------------------- #
# 3. Parameter container (FROZEN, BUILD-CONTRACT section 7)                    #
# --------------------------------------------------------------------------- #
@dataclass
class Param:
    """One model parameter: value, plausible range, PSA distribution, source, and
    the two audit flags this build turns on (`verify`, `evidence_status`).

    `price_year` is an ADDITION to the section-7 field list, permitted because
    section 7 says "Also exported". It exists so the 2019-USD / 2025-USD split
    (see the module docstring) is structural rather than a naming convention:
    read every cost through `cost_2025()`, which inflates if and only if
    `price_year != PRICE_YEAR`. Non-cost parameters carry `price_year=None`.
    """

    name: str
    value: float
    low: float
    high: float
    dist: str                    # 'beta'|'gamma'|'lognormal'|'normal'|'dirichlet'|'fixed'
    source: str                  # e.g. "Edmiston 2023 [12]"
    verify: bool = False         # True => still gates on a Tier-1/Tier-2 PDF
    group: str = ""              # dirichlet grouping key
    note: str = ""
    evidence_status: str = ""    # 'reported'|'derived'|'derived_assumption'
    #                              |'candidate_unverified'|'not_reported'
    price_year: int | None = None  # cost parameters only; None = dimensionless
    source_locator: str = ""     # Phase A2: table/figure + PDF page the value was read
    #                              from, e.g. "Table 3, PDF p. 7". Non-empty for every
    #                              parameter closed out against a full text.

    def __float__(self) -> float:
        return float(self.value)


#: Vocabularies the import-time audit enforces.
DISTRIBUTIONS = frozenset(
    {"beta", "gamma", "lognormal", "normal", "dirichlet", "fixed"}
)
EVIDENCE_STATUSES = frozenset(
    {"reported", "derived", "derived_assumption", "candidate_unverified", "not_reported"}
)


# --------------------------------------------------------------------------- #
# 4. Rate <-> probability helpers                                              #
# --------------------------------------------------------------------------- #
def rate_from_prob(p: float, t: float) -> float:
    """Constant hazard implied by a cumulative probability `p` observed over `t`
    time units: r = -ln(1 - p) / t.  Time units are YEARS throughout this build."""
    p = float(p)
    if p >= 1.0:
        return math.inf
    if p <= 0.0:
        return 0.0
    return -math.log(1.0 - p) / float(t)


def prob_from_rate(r: float, t: float = 1.0) -> float:
    """Cumulative probability over `t` time units under a constant hazard `r`:
    p = 1 - exp(-r * t)."""
    return 1.0 - math.exp(-float(r) * float(t))


def convert_prob(p: float, t_from: float, t_to: float = 1.0) -> float:
    """Re-express a probability observed over `t_from` as the equivalent over `t_to`
    under a constant hazard:  1 - (1 - p) ** (t_to / t_from).

    Examples used by this build:
        convert_prob(qx, 1.0, 1/12)   annual life-table qx -> monthly
        convert_prob(p,  0.25, 1/12)  90-day risk -> uniform monthly (the STRUCTURAL
                                      SCENARIO only; the base case uses the
                                      median-matched conditional onset weights).
    """
    return prob_from_rate(rate_from_prob(p, t_from), t_to)


# --------------------------------------------------------------------------- #
# 5. CPI medical-care inflation (CSV-backed; no index value is ever hard-coded) #
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def load_cpi() -> dict[int, float]:
    """{year -> annual-average index} from `data/cpi_medical.csv`.

    BLS CUUR0000SAM (CPI-U, U.S. city average, Medical care, NSA), period M13 =
    annual average, 2010-2025, with a MULTI-LINE `#` provenance header -- hence
    `comment="#"`, NOT `skiprows=1` (that convention governs results CSVs only).
    The index column is literally named `index`, so it must be read as `df["index"]`;
    `df.index` would silently return the row index.
    """
    df = pd.read_csv(CPI_CSV, comment="#")
    if "year" not in df.columns or "index" not in df.columns:
        raise ValueError(f"{CPI_CSV} must have columns 'year,index'; got {list(df.columns)}")
    return {int(y): float(v) for y, v in zip(df["year"], df["index"])}


#: Back-compat alias for the sibling-project loader name.
_load_cpi = load_cpi


def inflate(value: float, from_year: int, to_year: int = PRICE_YEAR) -> float:
    """Inflate a nominal cost from `from_year` to `to_year` using the BLS medical-care
    CPI: value * index[to_year] / index[from_year]. Raises on an out-of-range year
    rather than clamping or extrapolating.

    The pivotal conversion for this build is 2019 -> 2025 = 580.102 / 498.413 =
    1.163898, which turns the plan's weighted 24-month cost per SSI of $78,024
    (2019 USD) into $90,812 (2025 USD) -- BUILD-CONTRACT section 10 as corrected
    2026-08-09. Never hard-code that factor; read it from the CSV.
    """
    cpi = load_cpi()
    if int(from_year) not in cpi:
        raise ValueError(
            f"medical-care CPI has no index for from_year={from_year}; "
            f"available {min(cpi)}-{max(cpi)}"
        )
    if int(to_year) not in cpi:
        raise ValueError(
            f"medical-care CPI has no index for to_year={to_year}; "
            f"available {min(cpi)}-{max(cpi)}"
        )
    return float(value) * cpi[int(to_year)] / cpi[int(from_year)]


@lru_cache(maxsize=8)
def inflation_factor(from_year: int, to_year: int = PRICE_YEAR) -> float:
    """index[to_year] / index[from_year]. Convenience wrapper over `inflate`."""
    return inflate(1.0, from_year, to_year)


def cost_2025(param: Param, value: float | None = None) -> float:
    """Return a cost parameter in PRICE_YEAR (2025) USD, inflating IF AND ONLY IF the
    parameter is stored in a different price year.

    THE ONE WAY TO READ A COST. Every consumer -- model.py, run_*.py, make_tables.py --
    must go through this. `value` optionally overrides `param.value` with a PSA draw or
    a DSA bound expressed in the parameter's OWN stored price year (so a gamma draw on
    `c_ssi_deep_0_6mo` is a draw in 2019 USD and is inflated here, exactly once).

    Raises for a parameter with `price_year=None` (a dimensionless multiplier such as
    `m_ssi_cost` or `k_attrib_pennington`), because inflating one would be a bug.
    """
    if not isinstance(param, Param):
        raise TypeError("cost_2025 expects a Param; pass the Param, not its value")
    if param.price_year is None:
        raise ValueError(
            f"{param.name} is dimensionless (price_year=None) and must not be inflated; "
            "read it as float(param)"
        )
    v = float(param.value) if value is None else float(value)
    if int(param.price_year) == PRICE_YEAR:
        return v
    return inflate(v, int(param.price_year), PRICE_YEAR)


# --------------------------------------------------------------------------- #
# 6. Life table / background mortality (CSV-backed)                            #
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def load_life_table() -> pd.DataFrame:
    """US Life Tables 2023 (Arias, NVSR 74(6), Tables 2-3) annual qx by single year of
    age and sex, as a DataFrame indexed by integer age with columns
    ['qx_male', 'qx_female'].

    Age range 40-100 inclusive. `age` is the STARTING integer age of the half-open
    interval [x, x+1). Age 100 is the NVSR "100 and older" terminal row and carries
    qx = 1.0 by construction -- a closing convention, not an observation; nothing may
    divide by (1 - qx) there. Multi-line `#` provenance header, hence `comment="#"`.
    """
    df = pd.read_csv(LIFE_TABLE_CSV, comment="#")
    missing = {"age", "qx_male", "qx_female"} - set(df.columns)
    if missing:
        raise ValueError(f"{LIFE_TABLE_CSV} is missing columns {sorted(missing)}")
    df = df.set_index("age").sort_index()
    return df[["qx_male", "qx_female"]]


def _resolve_frac_male(sex_mix=None) -> float:
    """Normalize `sex_mix` to a fraction male in [0, 1].

    Accepts None (use `mortality.frac_male`, the source of truth), a bare float
    (fraction male), or a dict keyed 'M'/'F' or 'male'/'female'.
    """
    if sex_mix is None:
        return float(mortality.frac_male.value)
    if isinstance(sex_mix, Param):
        return float(sex_mix.value)
    if isinstance(sex_mix, (int, float)):
        fm = float(sex_mix)
    elif isinstance(sex_mix, dict):
        keyed = {str(k).strip().lower()[:1]: float(v) for k, v in sex_mix.items()}
        if "m" not in keyed or "f" not in keyed:
            raise ValueError(f"sex_mix dict must be keyed M/F (or male/female); got {sex_mix}")
        total = keyed["m"] + keyed["f"]
        if total <= 0:
            raise ValueError(f"sex_mix must have positive total; got {sex_mix}")
        fm = keyed["m"] / total
    else:
        raise TypeError(f"sex_mix must be None, a float or a dict; got {type(sex_mix)!r}")
    if not 0.0 <= fm <= 1.0:
        raise ValueError(f"fraction male must be in [0, 1]; got {fm}")
    return fm


@lru_cache(maxsize=512)
def _qx_blended(age: int, frac_male: float) -> float:
    """Sex-mix-blended ANNUAL qx at integer `age`. Raises outside the table range --
    a clamp would hide a mis-specified cohort age."""
    lt = load_life_table()
    a_min, a_max = int(lt.index.min()), int(lt.index.max())
    if not a_min <= age <= a_max:
        raise ValueError(
            f"age {age} is outside the life table range {a_min}-{a_max}; extend "
            f"{os.path.basename(LIFE_TABLE_CSV)} from the NCHS spreadsheets rather "
            "than interpolating"
        )
    row = lt.loc[age]
    return float(frac_male * row["qx_male"] + (1.0 - frac_male) * row["qx_female"])


def background_mortality(age, sex_mix=None, t: float = 1.0) -> float:
    """Probability of dying over `t` **YEARS** at cohort `age`, from the 2023 US life
    table blended at `sex_mix`.

        p = 1 - (1 - qx) ** t          <- MONTHLY CYCLES PASS t = 1/12

    NOT `qx * t` and NOT `qx / 12`: at age 60 the linear form gives 0.00095475 against
    the correct 0.00095980, a 0.5% relative error that compounds across all 24 cycles.
    `age` is floored to the integer starting age of the life-table interval.
    """
    a = int(math.floor(float(age)))
    qx = _qx_blended(a, _resolve_frac_male(sex_mix))
    if qx >= 1.0:
        return 1.0
    return 1.0 - (1.0 - qx) ** float(t)


# --------------------------------------------------------------------------- #
# 7. tp -- transitions and effects (24 parameters)                             #
# --------------------------------------------------------------------------- #
class tp:
    """Transition probabilities, relative effects, onset weights, tunnel lengths and
    the clinical counters. Derivations: proposed-inputs.md sections 1-3, 6."""

    # ---- The x-axis -------------------------------------------------------- #
    p_base_ssi = Param(
        "p_base_ssi", 0.066, 0.005, 0.15, "beta",
        "Edmiston 2023 [12]", verify=False, evidence_status="reported",
        source_locator="Results PDF p. 2 / Table 1b PDF p. 4 (journal pp. 89 and 91)",
        note="AXIS, NOT A BASE CASE. The baseline 90-day all-SSI risk is the "
             "independent variable, swept over SETTINGS['risk_grid'] (0.005-0.15, 291 "
             "points). value=0.066 is only the default sweep position so run_arm() has "
             "a default; no headline result in this paper is reported at a single value "
             "of it. Edmiston's 6.6% = 1.7% superficial + 4.9% deep in spinal-fusion "
             "claims (n=210,019 13,813 SSIs) ascertained over 3-180 d and plotted as "
             "a labelled anchor on a 90-day axis without being rescaled to it. A2 NOTE: "
             "Edmiston's cohort is ALL spinal fusion with NO anatomic restriction and is "
             "49.1% pure-anterior; its posterior-only family reports all-SSI 9.05% and "
             "deep 7.01% (n=87868). A posterior-lumbar-specific model could justify the "
             "higher anchor but the cost tranches are published on the pooled cohort so "
             "pooling both keeps rate and cost on ONE basis. Disclosed not silently "
             "resolved. run_psa.py MUST "
             "NOT sample this parameter: P(cost-effective | p) conditions on p, and "
             "integrating over it would destroy acceptability_by_risk.csv and Fig 4.")

    # ---- Relative effect (pivotal) ----------------------------------------- #
    rr_cinpwt = Param(
        "rr_cinpwt", 0.61, 0.49, 0.76, "lognormal",
        "Zwanenburg 2020 [11]", verify=False, evidence_status="reported",
        source_locator="Fig. 2 / Table 2, PDF pp. 4 and 6 (Ann Surg 2020;272(1):81-91)",
        note="PIVOTAL. **A2-CONFIRMED EXACTLY**: RR 0.61 (0.49-0.76) SSI-SPECIFIC (not "
             "a wound-complication composite) RCT-ONLY (observational pooled separately "
             "at RR 0.41) Mantel-Haenszel RANDOM effects 28 RCTs n = 4398 I2 = 27% "
             "Tau2 = 0.08 Z = 4.33 P < 0.0001. GRADE certainty HIGH with zero domains "
             "downgraded -- this build had recorded GRADE as NOT REPORTED. Pooled "
             "control rate is 14.3% (315/2205) as REPORTED not the ~13.5% this build "
             "back-solved from NNT 19; '13.5' appears nowhere in the paper.\n"
             "**THREE A2 FINDINGS THAT MUST REACH THE DISCUSSION.** (1) DEEP SSI: "
             "Zwanenburg found NO significant benefit for deep SSI in the RCTs and its "
             "authors warn that 'iNPWT efficacy may potentially be limited to "
             "superficial SSIs only and such SSIs are of less clinical relevance than "
             "deep SSIs.' This model draws ~89% of its cost saving from averted DEEP "
             "SSI so this is a first-order threat to its logic and is now carried as a "
             "named scenario (deep RR = 1.0 effect on superficial only) rather than "
             "left unstated. (2) NEAREST SPECIALTY SUBGROUP IS NULL: orthopedic/trauma "
             "RR 0.68 (0.43-1.08); there is no spine subgroup and there are ZERO spine "
             "RCTs. Clean/CDC class I surgery -- which is what lumbar fusion is -- gives "
             "RR 0.50 (0.37-0.68). The two candidate refinements bracket the base case "
             "in opposite directions which is why the pooled estimate is retained. "
             "(3) CONFLICT OF INTEREST: the senior author holds institutional grants "
             "from and sits on the advisory board of Acelity/KCI (Prevena) and speaks "
             "for Acelity/KCI and Smith & Nephew (PICO); the first author speaks for two "
             "wound-care distributors -- in a footnote block that then states 'The "
             "authors report no conflicts of interest.'\n"
             "PSA: median-preserving lognormal on the log scale sigma = "
             "(ln 0.76 - ln 0.49)/3.92 = 0.11197 median exp(ln 0.61) = 0.61. "
             "Correlated BY CONSTRUCTION in the engine (p_ciNPWT = p_control x RR) so "
             "arm ordering holds in every draw and the DSA carries one correlated 'gap' "
             "row rather than two arm rows. Chosen over the spine-specific Chen estimate "
             "because a weaker assumed effect RAISES the threshold so the paper's claim "
             "errs in the harder direction.")
    rr_cinpwt_spine = Param(
        "rr_cinpwt_spine", 0.434, 0.222, 0.824, "lognormal",
        "Chen 2022 [8] (OR converted to RR)", verify=False, evidence_status="derived",
        source_locator="Table 2 row 1 / Fig. 2, PDF p. 4 (journal p. 1549)",
        note="Favorable (spine-specific) scenario only. **A2 CORRECTIONS.** (1) Chen "
             "reports an ODDS ratio 0.399 (0.198-0.802); this build used it directly as "
             "a risk ratio. Converted properly at Chen's own pooled control rate "
             "p_c = 43/322 = 0.13354: RR = OR/(1 - p_c + p_c x OR) = 0.399/0.91974 = "
             "0.4338, bounds 0.2218 and 0.8238. At this model's operating range the "
             "substitution was nearly immaterial (+0.9% relative at p=0.015, +7.4% at "
             "p=0.114) but always in the optimistic direction, so it is corrected. "
             "(2) k = 4 studies and n = 525, NOT the 5 and 994 this build recorded -- "
             "994 is Chen's review-wide total across all five outcomes and includes "
             "Naylor 2020, which is absent from the SSI forest plot. (3) The pooling is "
             "FIXED-EFFECT Mantel-Haenszel with I2 = 0.0% (het. p = 0.566), which this "
             "build recorded as NOT REPORTED; the published CI therefore carries no "
             "between-study variance. (4) Zero RCTs: 1 prospective observational + 3 "
             "retrospective cohorts.\n"
             "**NOT INDEPENDENT OF rr_cinpwt_lambrechts** -- see that parameter.")
    rr_cinpwt_lambrechts = Param(
        "rr_cinpwt_lambrechts", 0.431, 0.238, 0.781, "lognormal",
        "Lambrechts 2022 [7]", verify=False, evidence_status="reported",
        source_locator="Fig. 4A, PDF p. 8 (journal p. 265)",
        note="Favorable-scenario cross-check. **A2 CORRECTION 0.397 -> 0.431.** This "
             "build derived 4.49 / 11.32 = 0.3966 as a ratio of two SEPARATELY POOLED "
             "single-arm proportions computed over non-identical study sets (7 vs 6 "
             "studies). That is not a valid relative risk and can carry no CI, which is "
             "why it was held fixed. Lambrechts DOES report a direct comparative "
             "Mantel-Haenszel estimate in Fig. 4A -- OR 2.32 (1.28-4.21) for standard "
             "dressing vs ciNPWT -- which inverts to 0.431 (0.238-0.781) and supplies a "
             "genuine CI, so the parameter becomes lognormal.\n"
             "**CRITICAL DISCLOSURE: THIS IS NOT AN INDEPENDENT SECOND SPINE ESTIMATE.** "
             "Lambrechts states verbatim that Chen 'included 5 of the 8 studies used in "
             "our meta-analysis', and A2 confirmed Chen's 4 SSI studies are a STRICT "
             "SUBSET of Lambrechts' 6 with IDENTICAL 2x2 tables (Chen's forest-plot ORs "
             "are exact reciprocals of Lambrechts'). The apparent agreement of 0.399 and "
             "0.397 was the same data counted twice, not convergent validity. Mueller "
             "2021 [6] -- also carried separately as rr_cinpwt_mueller -- is the "
             "largest-weight study inside BOTH, so those 217 patients appear a third "
             "time. Both parameters are retained so the scenario table stays comparable "
             "with v1, but Methods and Table 4 must state that the three favourable "
             "spine estimates rest on one pooled study set of at most 6 non-randomised "
             "studies, and no RCT contributes to any of them.")
    rr_cinpwt_mueller = Param(
        "rr_cinpwt_mueller", 0.281, 0.083, 0.948, "lognormal",
        "Mueller 2021 [6]", verify=False, evidence_status="derived",
        source_locator="Table 2 'Instrumentation: Yes' row, PDF p. 5 (journal p. E449)",
        note="Most favorable bound in the evidence base. **VALUE CONFIRMED, COUNTS "
             "CORRECTED.** This build recorded the instrumented subgroup as ~18/156 and "
             "~4/118. The true counts are 14/123 (standard) and 3/94 (ciNPT): 156 and "
             "118 are the WHOLE ARM sizes, not the subgroup denominators. The original "
             "counts are not merely unsupported but impossible -- the entire standard "
             "arm contains only 17 SSIs. Recomputing from the true cells: "
             "(3/94)/(14/123) = 0.2804, so 0.281 stands; a 95% CI of 0.083-0.948 is now "
             "derivable, so the parameter becomes lognormal instead of fixed.\n"
             "**DO NOT USE MUELLER'S PRINTED EFFECT COLUMN.** Its Table 2 prints "
             "'RR 0.662 (0.513-0.853)' for this row. A2 established that the whole "
             "column is a transposed-2x2 artifact -- it reproduces in 30 of 33 rows as "
             "P(standard arm | no SSI) / P(standard arm | SSI), and prints finite RRs "
             "with CIs excluding 1 even for rows where the ciNPT arm had ZERO events. "
             "Only the cell counts and the chi-squared p-values are usable. The "
             "manuscript must footnote that 0.281 is recomputed from cell counts, or a "
             "reviewer opening Table 2 will read a discrepancy.")
    rr_cinpwt_attenuated = Param(
        "rr_cinpwt_attenuated", 0.85, 0.76, 1.00, "fixed",
        "scenario constant", verify=False, evidence_status="derived_assumption",
        note="Attenuated-effect scenario constant, not an estimate: tests the "
             "assumption that ciNPWT's RR applies equally to the residual (relatively "
             "more gram-negative and polymicrobial) SSI left after decolonization. "
             "No requested PDF can resolve a defined constant, hence verify=False.")

    # ---- Destination split (Dirichlet pair) -------------------------------- #
    frac_deep = Param(
        "frac_deep", 0.745, 0.30, 0.80, "dirichlet",
        "Edmiston 2023 [12]", verify=False, group="ssi_split", evidence_status="derived",
        source_locator="Table 1b, PDF p. 4 (Infect Control Hosp Epidemiol 2023;44(1):91)",
        note="**A2 CORRECTION 0.742 -> 0.745.** Derived from COUNTS 10,296 / 13,813 = "
             "0.74538, not from the rounded rates 4.9 / 6.6 = 0.74242. The tiebreak is "
             "Edmiston's own prose: 'of these, 75% were deep-incisional SSIs' -- the "
             "count basis rounds to 75%, the rate basis to 74%. Materially trivial "
             "($115 per SSI case, 0.23%) but the count basis is the paper's own. PSA: sample as a PAIR with frac_sup using "
             "the DOWN-WEIGHTED concentration SETTINGS['dirichlet_kappa_ssi_split'] "
             "= 10.767, i.e. alpha ~ (7.99, 2.78) -- NOT the raw Edmiston counts "
             "(~10,291 deep / ~3,570 superficial), which would make PSA treat the split "
             "as known. The real uncertainty is claims-coding validity, not sampling. "
             "The wide 0.30-0.80 DSA range is retained and is now doing more work: "
             "Urquhart 2021 [21] -- prospective and matched to exactly this model's "
             "population -- reports 31/79 = 0.392 deep. A2 examined switching to 0.392 "
             "and REJECTED it. The two sources agree on the DEEP RATE within 18% "
             "(Urquhart 5.79% vs Edmiston 4.90%) and differ 5.4-fold only on "
             "SUPERFICIAL, a bucket Urquhart treated with two weeks of oral antibiotics "
             "and which claims cannot see. frac_deep weights Edmiston's CLAIMS-VISIBLE "
             "cost tranches, so swapping the split alone -- while still pricing "
             "superficial SSI at Edmiston's $32,821 -- prices claims-invisible cases as "
             "though they generated claims and makes the model LESS internally coherent, "
             "not more. 0.392 already sits inside this range at the 18th percentile, so "
             "the published tornado already reports a more adverse case. This is a "
             "disclosed limitation, not an unexamined one.")
    frac_sup = Param(
        "frac_sup", 0.255, 0.20, 0.70, "dirichlet",
        "Edmiston 2023 [12]", verify=False, group="ssi_split", evidence_status="derived",
        source_locator="Table 1b, PDF p. 4 (Infect Control Hosp Epidemiol 2023;44(1):91)",
        note="**A2 CORRECTION 0.258 -> 0.255.** Derived from counts 3,517 / 13,813 = "
             "0.25462; frac_deep + frac_sup = 1.000 exactly. Sample as a pair with "
             "frac_deep, group='ssi_split'. A2 also corrected the all-SSI COUNT from "
             "13,861 to 13,813: 13,861 appears nowhere in the paper, and 13,813 is the "
             "only value satisfying both closure identities (10,296 + 3,517 = 13,813 "
             "and 13,813 + 196,206 = 210,019).")

    # ---- 90-day onset weights (CONDITIONAL inputs -- see module docstring) -- #
    # lambda = ln2/median; F(t) = 1 - exp(-lambda t); w_k = [F(30k) - F(30(k-1))]/F(90).
    # Deep, median 28 d: lambda = 0.024755/d; F(30)=0.524152 F(60)=0.773569 F(90)=0.892253.
    w_onset_deep_m1 = Param(
        "w_onset_deep_m1", 0.587, 0.333, 0.700, "fixed",
        "Edmiston 2023 [12]", verify=False, evidence_status="derived",
        source_locator="Fig. 2 (median + IQR), PDF p. 5 (Infect Control Hosp Epidemiol 2023;44(1):92)",
        note="CONDITIONAL-PROBABILITY INPUT, NOT A CYCLE HAZARD. The engine must apply "
             "h_1 = p*w_1, h_2 = p*w_2/(1 - p*w_1), h_3 = p*w_3/(1 - p*w_1 - p*w_2) so "
             "the 90-day cumulative equals p exactly. Derived from Edmiston's median "
             "28 d deep onset: 0.524152/0.892253. Triple sums to 1.000.\n"
             "**A2.** The median 28 d is CONFIRMED (Fig. 2) and Fig. 2 additionally "
             "publishes the full quartiles -- deep P25/P50/P75/P90 = 13/28/70/134 d -- "
             "which this build never used. The weights themselves are NOT reported by "
             "Edmiston: they are DERIVED from the published median under an "
             "exponential-onset assumption, which is a modelling choice and is labelled "
             "as one. An A2 verifier asserted the superficial triple was internally "
             "inconsistent with its stated 44 d median; that claim was checked during "
             "adjudication and REJECTED -- both triples reproduce the stated medians "
             "exactly (see AUDIT section C). Fig. 2's P90 of 134 d (deep) and 149 d "
             "(superficial) means roughly 20% of deep and 24% of superficial onsets fall "
             "AFTER day 90; that is a statement about total burden, not about these "
             "weights, which are CONDITIONAL on onset inside the 90-day window that "
             "defines this model's risk axis.")
    w_onset_deep_m2 = Param(
        "w_onset_deep_m2", 0.280, 0.200, 0.334, "fixed",
        "Edmiston 2023 [12]", verify=False, evidence_status="derived",
        source_locator="Fig. 2 (median + IQR), PDF p. 5 (Infect Control Hosp Epidemiol 2023;44(1):92)",
        note="(0.773569 - 0.524152)/0.892253. Conditional input -- see w_onset_deep_m1.")
    w_onset_deep_m3 = Param(
        "w_onset_deep_m3", 0.133, 0.100, 0.334, "fixed",
        "Edmiston 2023 [12]", verify=False, evidence_status="derived",
        source_locator="Fig. 2 (median + IQR), PDF p. 5 (Infect Control Hosp Epidemiol 2023;44(1):92)",
        note="(0.892253 - 0.773569)/0.892253. Conditional input -- see w_onset_deep_m1. "
             "Hazard is ZERO from cycle 4 onward, so the pivotal hazard needs no "
             "extrapolation over the 2-year horizon.")
    # Superficial, median 44 d: lambda = 0.015753/d; F(30)=0.376621 F(60)=0.611398 F(90)=0.757754.
    w_onset_sup_m1 = Param(
        "w_onset_sup_m1", 0.497, 0.333, 0.600, "fixed",
        "Edmiston 2023 [12]", verify=False, evidence_status="derived",
        source_locator="Fig. 2 (median + IQR), PDF p. 5 (Infect Control Hosp Epidemiol 2023;44(1):92)",
        note="CONDITIONAL-PROBABILITY INPUT -- see w_onset_deep_m1. Derived from the "
             "median 44 d superficial onset: 0.376621/0.757754. Triple sums to 1.000. "
             "Alternative reported as a structural scenario: uniform (1/3, 1/3, 1/3); at "
             "3%/yr the choice is immaterial. **A2:** median 44 d CONFIRMED (Fig. 2, "
             "IQR 26-87, P90 149). See w_onset_deep_m1 for the derivation caveat and for "
             "the rejected inconsistency claim.")
    w_onset_sup_m2 = Param(
        "w_onset_sup_m2", 0.310, 0.250, 0.334, "fixed",
        "Edmiston 2023 [12]", verify=False, evidence_status="derived",
        source_locator="Fig. 2 (median + IQR), PDF p. 5 (Infect Control Hosp Epidemiol 2023;44(1):92)",
        note="(0.611398 - 0.376621)/0.757754. Conditional input -- see w_onset_sup_m1.")
    w_onset_sup_m3 = Param(
        "w_onset_sup_m3", 0.193, 0.150, 0.334, "fixed",
        "Edmiston 2023 [12]", verify=False, evidence_status="derived",
        source_locator="Fig. 2 (median + IQR), PDF p. 5 (Infect Control Hosp Epidemiol 2023;44(1):92)",
        note="(0.757754 - 0.611398)/0.757754. Conditional input -- see w_onset_sup_m1.")

    # ---- Tunnel lengths (must match the frozen slot vector) ---------------- #
    n_tunnel_sup = Param(
        "n_tunnel_sup", 3, 2, 6, "fixed",
        "assumption (no source reports a superficial-SSI duration)",
        verify=True, evidence_status="derived_assumption",
        note="Cycles (months) in the superficial tunnel. MUST equal SUP_LEN = 3 in the "
             "frozen 18-slot vector; asserted at import. It is exposed as a parameter "
             "for the scenario analysis, but changing it changes N_SLOTS, so a scenario "
             "must rebuild the slot vector rather than pass a different number to the "
             "base-case engine.")
    n_tunnel_deep = Param(
        "n_tunnel_deep", 12, 6, 24, "fixed",
        "Urquhart 2021 [21]", verify=False, evidence_status="derived",
        source_locator="Table III, PDF p. 6 (J Bone Joint Surg Am 2021;103:2110)",
        note="Cycles (months) in the deep tunnel. MUST equal DEEP_LEN = 12 in the frozen "
             "18-slot vector; asserted at import. See n_tunnel_sup for the scenario "
             "caveat. **A2: value and 6-24 range both UPHELD after an attempt to narrow "
             "them was REFUTED.** Urquhart's SSI group is significantly worse at 1.5 3 "
             "and 6 months (ODI) and at 6 months (PCS -4.0 -6.8 to -1.2) and null at 12 "
             "and 24 months on all five PROMs plus satisfaction and MCID -- so recovery "
             "is bounded in (6 12] with no measurement in between and 12 is its upper "
             "bound. A proposal to cut the DSA high from 24 to 12 was rejected: the "
             "12-month PCS CI is -4.7 to +1.0 (P = 0.195) which is a failure to reject "
             "rather than a demonstration of equivalence and admits a residual around "
             "0.09 and McFarland 2023 [19]'s deep-specific analysis shows ZERO time "
             "attenuation (Q(6) = 2.09 P = 0.91). LABELLING CAVEAT: Urquhart's 79 SSIs "
             "are POOLED (48 superficial 31 complicated) so a deep tunnel is being set "
             "from a 61%-superficial group -- which makes 12 a LOWER bound and "
             "reinforces rather than undermines the 6-24 range.")

    # ---- Counters and cross-checks: NO INCREMENTAL COST (Decision 3) ------- #
    p_persistent_deep = Param(
        "p_persistent_deep", 0.25, 0.12, 0.353, "beta",
        "Freire-Archer 2023 [17]", verify=False, evidence_status="reported",
        source_locator="Results + Fig. 3, PDF pp. 7 and 12 (Spine 2023;48(16):E275)",
        note="COUNTER ONLY -- CARRIES NO INCREMENTAL COST (plan Decision 3). Persistent "
             "infection after initial I&D. **A2: 24.9% (95% CI 16.8-35.3%) is the "
             "TRIM-AND-FILL BIAS-ADJUSTED value and is the paper's UPPER bound -- its "
             "own wording is 'at least 12% up to 25%' and the OBSERVED pooled estimate "
             "is 12% (6.3-20% k=29). The base case keeps 25% but the DSA low bound is "
             "widened 0.168 -> 0.12: a sensitivity range that excludes the number in the "
             "source's own Conclusions is indefensible.** Edmiston's "
             "payment is ALL-CAUSE incremental spend and already contains it; if this "
             "ever multiplies a cost the model double-counts. It exists to produce "
             "'persistent infections prevented per 1,000' and a validation row.")
    p_reop_ssi = Param(
        "p_reop_ssi", 0.286, 0.163, 0.451, "beta",
        "Karamian 2022 [20]", verify=False, evidence_status="reported",
        source_locator="Table 2 'Revision rate' row, PDF p. 4 (journal p. 1058)",
        note="COUNTER ONLY -- CARRIES NO INCREMENTAL COST (Decision 3). Value 10/35 = "
             "0.2857 A2-CONFIRMED (independently corroborated by reproducing the printed "
             "p = 0.048 as a Yates-corrected chi-squared on 10/35 vs 13/105). Range is a "
             "Wilson 95% score interval: centre 0.30691 half-width 0.14365 -> "
             "(0.163 0.451). Produces 'reoperations prevented per 1000'.\n"
             "**THREE A2 LABEL CORRECTIONS.** (1) NOT 'at 1 year' -- Karamian states no "
             "ascertainment window for revision at all; mean follow-up is 2.69 +/- 1.56 "
             "y in the SSI arm. Treating 0.286 as a 1-year risk inflates the annual "
             "hazard ~2.7x (the 1-year-equivalent is ~0.118). (2) NOT 'reoperation' "
             "generally -- the variable is 'Revision rate' defined as 'instrumentation "
             "removal or exchange involving the index levels' i.e. HARDWARE REVISION "
             "not I&D. Karamian contradicts itself here also stating twice that 'all "
             "patients in the SSI group retained instrumentation'; unresolvable from the "
             "PDF and recorded as a source-integrity flag. (3) ATTACHES TO THE DEEP "
             "TUNNEL not to all SSI. Karamian's CDC definition is nominally pooled but "
             "its cohort is 94.3% deep/organ-space by composition (31 deep + 1 "
             "meningitis + 1 disk space of 35) because entry required intraoperative "
             "evidence of infection -- non-operative superficial SSI cannot enter. This "
             "model's SSI state is only 74.5% deep.\n"
             "DISCLOSED LIMITATION: 0.286 is an ALL-CAUSE revision rate; the matched "
             "uninfected arm has 13/105 = 12.4% on the same variable so the "
             "SSI-ATTRIBUTABLE excess is 0.162 (RR 2.31 95% CI 1.11-4.79). Preventing an "
             "SSI prevents only the excess so 'reoperations prevented per 1000' as "
             "reported is an upper bound. Changing the counter's semantics is a "
             "model-structure change outside A2 scope and is stated rather than made.\n"
             "**BLIND SPOT-CHECK FINDING (2026-08-17).** Because Karamian's 'Revision "
             "rate' counts only instrumentation removal or exchange, it EXCLUDES the "
             "debridements every infected patient underwent (mean 1.26 per patient; "
             "readmission 100%). On any definition of reoperation that includes I&D, the "
             "reoperation rate after SSI in that cohort was effectively 100%, not 28.6%. "
             "So 0.286 is a narrow hardware-revision rate, and the counter it feeds "
             "should be read as 'hardware revisions prevented', never as 'reoperations "
             "prevented'. The comparator is a 3:1 propensity-matched control group.")
    n_debridements = Param(
        "n_debridements", 2.11, 1.70, 2.52, "gamma",
        "Freire-Archer 2023 [17]", verify=False, evidence_status="reported",
        source_locator="Fig. 4, PDF p. 13 (Spine 2023;48(16):E281)",
        note="CROSS-CHECK ONLY -- NEVER COSTED (Decision 3); already inside Edmiston's "
             "all-cause increment. **A2 CORRECTION 1.4 (0.8-1.9) -> 2.11 (1.70-2.52) "
             "with three label errors fixed.** (1) '0.8-1.9' was never a range of study "
             "means -- it is a 95% CI labelled as such in the same Results sentence on a "
             "trim-and-fill adjusted value. (2) The count is TOTAL debridements "
             "INCLUDING the index I&D per the Methods ('including initial incision and "
             "debridement') not 'additional' ones; the abstract says 'additional' and "
             "contradicts the paper's own Methods and this build copied the abstract. "
             "(3) It covers ALL deep SSI (N = 617 matches Table 1's deep-SSI counts) not "
             "the persistent subset. Decisive: a lower bound of 0.8 is BELOW the logical "
             "floor of 1.0 for a count that includes the index procedure which is "
             "dispositive that 1.4 was not physically meaningful. Observed pooled value "
             "2.11 (1.70-2.52) I2 = 95% k = 21. If a downstream consumer needs "
             "'additional beyond index' that quantity is 1.11 (0.70-1.52). Karamian 2022 "
             "[20] independently reports 1.26 +/- 0.61 in a single centre using routine "
             "vancomycin powder.")
    rr_readmit_sa = Param(
        "rr_readmit_sa", 2.15, 2.06, 2.25, "lognormal",
        "Rosenthal 2019 [13]", verify=False, evidence_status="reported",
        note="CROSS-CHECK ONLY -- NEVER ADDED TO ANY COST (Decision 3): readmission is "
             "already inside Edmiston's all-cause increment. Rosenthal's hospital costs "
             "are on a 2015 USD basis and are excluded from this build entirely. Not "
             "requested at A2, hence verify=False.")

    # ---- Decolonization (secondary analysis; cancels from the base case) ---- #
    or_decol = Param(
        "or_decol", 0.56, 0.33, 0.98, "lognormal",
        "Joshi 2026 [2]", verify=False, evidence_status="reported",
        source_locator="Table 4, PDF p. 5 (J Neurosurg Spine 2026;45:130)",
        note="**A2-CONFIRMED, and the (1-f) + f x OR construction SURVIVES.** Joshi's "
             "Methods define the multivariable outcome as 'any S. aureus superficial "
             "SSI BSI DII or organ/space infection' -- it is organism-restricted to "
             "S. aureus, which is exactly what the construction requires, so the "
             "multiplier does NOT collapse to the bare OR. OR 0.56 (0.33-0.98) p = 0.041 "
             "multivariable logistic reference = no decolonization. RESIDUAL DEFECT, "
             "disclosed: the endpoint is a COMPOSITE in which 18 of 68 events (26.5%) "
             "are bloodstream infections rather than SSIs, so substituting it for an "
             "SSI-specific OR assumes effect homogeneity across infection site -- "
             "untested and unrecoverable from this paper. Joshi also reports a better-"
             "powered any-decolonization OR of 0.60 (0.42-0.84) p = 0.003; 0.56 is "
             "retained because this model represents the combined CHG + mupirocin "
             "regimen specifically. Secondary decolonization analysis only. "
             "Spine-specific combined CHG + "
             "mupirocin. Universal decolonization is background in BOTH arms per "
             "SHEA/IDSA, so it CANCELS from the ciNPWT incremental; the secondary "
             "analysis reports only how far it shifts patients below the threshold.")
    rr_decol_broad = Param(
        "rr_decol_broad", 0.41, 0.30, 0.55, "lognormal",
        "Seidelman 2023 [28]", verify=False, evidence_status="reported",
        note="Secondary-analysis alternative to or_decol; not spine-specific. From a "
             "source not requested at A2, hence verify=False.")
    f_sa_ssi = Param(
        "f_sa_ssi", 0.46, 0.391, 0.521, "fixed",
        "Freire-Archer 2023 [17]", verify=False, evidence_status="reported",
        source_locator="Results + Fig. 5, PDF pp. 8 and 14 (Spine 2023;48(16):E276)",
        note="S. aureus fraction of spine SSI. **A2: the claim that no CI is reported is "
             "FALSE** -- the paper prints 46% (95% CI 39.1-52.1%) I2 = 65.3% k = 39 now "
             "carried in low/high. dist stays 'fixed' so PSA behaviour is unchanged and "
             "all m_decol_residual uncertainty still flows from or_decol's CI. ESTIMAND "
             "CAVEAT: Fig. 5's x-axis is 'Number of Deep SSI Samples with Culture "
             "Growth' so 46% is the share of culture-positive ISOLATES not of "
             "INFECTIONS which is what the (1-f) + f x OR construction strictly needs. "
             "The two biases run opposite ways -- polymicrobial dilution pushes 0.46 "
             "down culture-negative exclusion pushes it up -- and the paper states most "
             "infections were monomicrobial so the net distortion is modest. "
             "Single-sourced: Joshi 2026 [2] counted only S. aureus and cannot "
             "corroborate it.")
    m_decol_residual = Param(
        "m_decol_residual", 0.798, 0.692, 0.991, "fixed",
        "Freire-Archer 2023 [17] + Joshi 2026 [2]", verify=False, evidence_status="derived",
        source_locator="derived: Freire-Archer Fig. 5 PDF p. 14 + Joshi Table 4 PDF p. 5",
        note="Residual-risk multiplier after universal decolonization = "
             "(1 - f_sa_ssi) + f_sa_ssi * or_decol = 0.54 + 0.46 x 0.56 = 0.7976. "
             "Low/high propagate the OR's CI only: 0.54 + 0.46 x 0.33 = 0.6918 and "
             "0.54 + 0.46 x 0.98 = 0.9908. Secondary analysis only.")


# --------------------------------------------------------------------------- #
# 8. cost -- SSI tolls, device, comparator, Medicare scenario (17 parameters)  #
#    PRICE YEARS ARE MIXED ON PURPOSE. Read every one through cost_2025().     #
# --------------------------------------------------------------------------- #
# Edmiston reports CUMULATIVE incremental payments at 6/12/24 months; the eight values
# below are the MARGINAL tranches, already differenced, so no dollar is counted twice:
#   superficial   20,800 | 26,937 | 32,821 ->  20,800 + 6,137 + 5,884  = 32,821
#   deep          59,766 | 74,875 | 93,741 ->  59,766 + 15,109 + 18,866 = 93,741
#   Medicare deep 48,662 | 53,757 | 73,803 ->  48,662 + 5,095 + 20,046 = 73,803
#   Medicare sup  11,044 | 17,967 | 24,096 ->  11,044 + 6,923 + 6,129  = 24,096
# All FOUR identities are asserted at import.
#
# A2 CORRECTION (2026-08-17). Two claims this build made about Edmiston were false:
#   1. "Edmiston reports NO SD, CI or IQR for any payment." Table 3 is titled
#      "Mean SSI Cost (95% CI)" and publishes a 95% CI for ALL TWELVE payments. The
#      assumed +/-50% span (CV 0.2551) is retired.
#   2. "The Medicare 12- and 24-month superficial payments are NOT REPORTED." They are
#      reported in four places (Abstract, Results, Table 3, Discussion) as $17,967 and
#      $24,096. The extrapolated tranches 3,259 / 3,124 were each ~2x too low.
#
# HOW THE RANGES ARE NOW BUILT. The published CIs are on the CUMULATIVE payments; the
# model's parameters are the MARGINAL tranches. For the four 0-6 mo tranches cumulative
# == marginal, so the published CI is carried verbatim. For the eight later tranches the
# marginal SE is recovered under an independent-increments assumption,
#     SE(marginal) = sqrt(SE(cum_t)^2 - SE(cum_t-1)^2),
# which is defined for every pair because SE grows with the window. Bounds are then
# value -/+ 1.96 x SE, preserving the SE = (high - low)/3.92 idiom BUILD-CONTRACT
# section 9 binds the PSA to. This is an APPROXIMATION and is declared as one in
# Methods: the two nested cumulative estimates are positively correlated, and the
# covariance is not recoverable from the published table. The conservative alternative
# (SE1 + SE2, the perfectly-anticorrelated bound) is reported as a check.
#
# Note these CIs are Wald intervals on least-squares means from a log-link gamma GLM on
# n = 210,019 -- they are SAMPLING precision on a mean, not the dispersion of cost across
# patients and not attribution uncertainty. Attribution uncertainty lives in m_ssi_cost,
# which is where it belongs; the retired +/-50% span was double-counting it.
class cost:
    """Incremental SSI payments (US commercial payer, Edmiston 2023 [12]), the device
    and comparator tolls, and the Medicare perspective scenario.
    Derivations: proposed-inputs.md section 4."""

    # ---- Commercial SSI tolls -- 2019 USD, MUST be inflated ---------------- #
    c_ssi_sup_0_6mo = Param(
        "c_ssi_sup_0_6mo", 20800, 18394, 23287, "gamma",
        "Edmiston 2023 [12] commercial", verify=False, evidence_status="reported",
        price_year=2019, source_locator="Table 3, PDF p. 7 (Infect Control Hosp Epidemiol 2023;44(1):94)",
        note="2019 USD -- inflate via cost_2025(). Superficial SSI 0-6 mo incremental "
             "payment, charged WHOLE as a toll at the cycle of SSI onset. A2-VERIFIED: "
             "range is Edmiston's PUBLISHED 95% CI ($18,394-$23,287), not the retired "
             "+/-50% assumption. At 0-6 mo cumulative == marginal so the CI transfers "
             "directly. Implied CV 0.060. The meaningful cost swing is m_ssi_cost.")
    c_ssi_sup_6_12mo = Param(
        "c_ssi_sup_6_12mo", 6137, 4948, 7326, "gamma",
        "Edmiston 2023 [12] commercial (derived)", verify=False, evidence_status="derived",
        price_year=2019, source_locator="Table 3, PDF p. 7 (Infect Control Hosp Epidemiol 2023;44(1):94)",
        note="2019 USD -- inflate via cost_2025(). MARGINAL tranche 26,937 - 20,800; "
             "already differenced, do not difference again. Charged at onset + 6 cycles "
             "(i.e. cycles 7-9), which is INSIDE the 12-cycle 1-year horizon. "
             "A2-VERIFIED: range from the published cumulative CIs via "
             "SE = sqrt(1387.8^2 - 1248.2^2) = 606.5.")
    c_ssi_sup_12_24mo = Param(
        "c_ssi_sup_12_24mo", 5884, 3770, 7998, "gamma",
        "Edmiston 2023 [12] commercial (derived)", verify=False, evidence_status="derived",
        price_year=2019, source_locator="Table 3, PDF p. 7 (Infect Control Hosp Epidemiol 2023;44(1):94)",
        note="2019 USD -- inflate via cost_2025(). MARGINAL tranche 32,821 - 26,937. "
             "Charged whole at onset + 12 cycles (cycles 13-15); it represents spending "
             "through months 25-27 post-index, so accrual reaches slightly past the "
             "nominal 2-year horizon. We disclose it rather than truncate it. "
             "A2-VERIFIED: range via SE = sqrt(1757.7^2 - 1387.8^2) = 1078.6.")
    c_ssi_deep_0_6mo = Param(
        "c_ssi_deep_0_6mo", 59766, 57550, 62030, "gamma",
        "Edmiston 2023 [12] commercial", verify=False, evidence_status="reported",
        price_year=2019, source_locator="Table 3, PDF p. 7 (Infect Control Hosp Epidemiol 2023;44(1):94)",
        note="2019 USD -- inflate via cost_2025(). Deep SSI 0-6 mo incremental payment, "
             "charged whole at the cycle of SSI onset. This is the single largest "
             "number in the model. A2-VERIFIED: range is the PUBLISHED 95% CI "
             "($57,550-$62,030), implied CV 0.019 -- the retired +/-50% assumption was "
             "13.4x too wide here, the worst case in the table.")
    c_ssi_deep_6_12mo = Param(
        "c_ssi_deep_6_12mo", 15109, 13612, 16606, "gamma",
        "Edmiston 2023 [12] commercial (derived)", verify=False, evidence_status="derived",
        price_year=2019, source_locator="Table 3, PDF p. 7 (Infect Control Hosp Epidemiol 2023;44(1):94)",
        note="2019 USD -- inflate via cost_2025(). MARGINAL tranche 74,875 - 59,766, "
             "charged at onset + 6 cycles. Dropping it from the 1-year scenario would "
             "silently remove $15,109 (2019 USD) per deep SSI and look like a plausible "
             "horizon effect rather than a bug. A2-VERIFIED: range via "
             "SE = sqrt(1374.5^2 - 1142.9^2) = 763.6.")
    c_ssi_deep_12_24mo = Param(
        "c_ssi_deep_12_24mo", 18866, 16269, 21463, "gamma",
        "Edmiston 2023 [12] commercial (derived)", verify=False, evidence_status="derived",
        price_year=2019, source_locator="Table 3, PDF p. 7 (Infect Control Hosp Epidemiol 2023;44(1):94)",
        note="2019 USD -- inflate via cost_2025(). MARGINAL tranche 93,741 - 74,875, "
             "charged at onset + 12 cycles. A2-VERIFIED: range via "
             "SE = sqrt(1909.2^2 - 1374.5^2) = 1325.1.")

    # ---- Attribution: the DSA driver and the scenario constant ------------- #
    m_ssi_cost = Param(
        "m_ssi_cost", 0.492, 0.306, 1.199, "fixed",
        "derived envelope: Pennington 2019 [14] to Edmiston 2023 [12]",
        verify=False, evidence_status="derived", price_year=None,
        source_locator="Pennington Table 3 PDF p. 3 / Edmiston Table 3 PDF p. 7",
        note="DIMENSIONLESS. Scales ALL SIX cost tranches TOGETHER so the tornado "
             "carries one interpretable SSI-attributable-cost bar -- which "
             "BUILD-CONTRACT section 10 requires to be the TOP bar. Swing THIS in "
             "run_dsa.py, never the six tranches independently, or the pivotal "
             "parameter fragments into six small bars and hides.\n"
             "**A2 DECISION -- THE BASE CASE IS NOW MATCHED-ANCHORED (was 1.000).** "
             "A2 verification established that Edmiston's payments are regression-"
             "adjusted ALL-CAUSE spend from a GLM with log link and gamma error, "
             "least-squares means, covariate adjustment only, with NO MATCHING anywhere "
             "in the paper. That estimand is an upper bound on attributable cost by "
             "construction: the residual gap is confounding by indication, since "
             "patients who develop an SSI are sicker and spend more regardless. "
             "Pennington's is a matched contrast (gender age +/-5 y BMI +/-5 date of "
             "surgery operating surgeon; 1:1 n=18/arm) in exactly this model's "
             "population. Anchoring the headline on the upper bound would rest the "
             "paper's central claim on its weakest assumption -- and Edmiston's five "
             "authors all hold J&J/Ethicon relationships a directional conflict on the "
             "pivotal benefit-side parameter. So the base case takes the matched "
             "anchor 0.492 and Edmiston's all-cause increment (1.000) becomes an "
             "upper-bound sensitivity. Envelope relative to the weighted 24-month cost "
             "per SSI of $78,206 (2019 USD): low 23,963/78,206 = 0.306 (Pennington's "
             "matched increment applied as the whole all-SSI 24-month cost -- the "
             "extreme-conservative end) high 93,741/78,206 = 1.199 (the deep-SSI value "
             "applied to every SSI). 'DSA driver' is a role and lives here not in "
             "dist.")
    k_attrib_pennington = Param(
        "k_attrib_pennington", 0.492, 0.435, 0.549, "fixed",
        "Pennington 2019 [14] / Edmiston 2023 [12]", verify=False,
        evidence_status="derived", price_year=None,
        source_locator="Pennington Table 3, PDF p. 3 (journal p. 107)",
        note="DIMENSIONLESS. **A2 CORRECTION 0.427 -> 0.492 (+15.2%) and it now sets "
             "the BASE CASE not a scenario.** Two errors were compounded in 0.427. "
             "(1) PRICE YEAR: Pennington states 'All costs in 2013 dollars' in the "
             "footnotes of both Table 3 and Table 4; the build treated them as 2019 USD. "
             "(2) COST CONCEPT: $37,009/$16,227 are TOTAL = direct + indirect (lost "
             "wages) -- a SOCIETAL basis. The payer-basis figures are the direct row "
             "$35,432 vs $14,992 increment $20,440 which is what this model's stated "
             "US commercial payer perspective requires and what Pennington's own Results "
             "call 'the direct cost of a deep wound infection'. Same-basis derivation: "
             "20,440 x (498.413/425.134) = $23,963 in 2019 USD; 23,963 / 48,662 = "
             "0.49244. Range propagates the SD reading of Pennington's dispersion "
             "(the paper never states whether +/- is an SD or an SEM). Both numerator "
             "and denominator are now DEEP-SSI 6-month 2019-USD; only the payer axis "
             "still transports (Medicare-priced to commercial) which a ratio tolerates "
             "far better than a level and which is labelled here as a transfer.")

    # ---- Intervention and comparator -- ALREADY 2025 USD, DO NOT INFLATE --- #
    c_device_cinpwt = Param(
        "c_device_cinpwt", 250, 100, 900, "gamma",
        "assumed net institutional price; list prices from Mueller 2026 [9]",
        verify=True, evidence_status="derived_assumption", price_year=2025,
        source_locator="list anchors: Mueller 2026 Materials, PDF p. 2 (journal p. 182)",
        note="ALREADY 2025 USD -- cost_2025() must return 250.0 unchanged. AXIS: swept "
             "over SETTINGS['price_grid'] ($100-$900 2025 USD, 161 points) as the second "
             "analysis axis rather than a fixed input, which is how this build converts "
             "the single largest evidence gap into a reported result. One-time toll at "
             "cycle 0 ARM B ONLY.\n"
             "**A2 CORRECTION TO THE SOURCE STRING.** This build asserted that no "
             "source reports a ciNPWT acquisition cost and that 'Mueller 2026 [9] does "
             "not state a device cost'. That is false. Mueller 2026 devotes a numbered "
             "Materials heading to it: '$495 per application' for the 3M Prevena "
             "Peel-and-Place 20 cm system and '$830 per application' for the Prevena "
             "Plus Customizable system both 2023 LIST prices restated across Tables "
             "1-3 and Figures 1-2. The $250 base case is retained and RELABELLED as an "
             "assumed NET institutional price -- roughly half of the cheaper list "
             "anchor -- because list price is not what a hospital pays and no source "
             "reports a negotiated figure. The price axis was widened from $600 to $900 "
             "so BOTH real products now sit ON the reported axis: at multilevel open "
             "posterior lumbar fusion the incision frequently exceeds 20 cm making the "
             "$830 customizable system the product-appropriate price. verify stays True: "
             "the gate is an institutional procurement figure that exists in no source "
             "and the list prices come from a paper three of whose four authors are "
             "employees of the manufacturer.")
    c_dressing_standard = Param(
        "c_dressing_standard", 0, 0, 50, "fixed",
        "assumption, corroborated by Mueller 2026 [9]",
        verify=False, evidence_status="derived_assumption", price_year=2025,
        source_locator="Mueller 2026, PDF p. 9 (journal p. 189)",
        note="ALREADY 2025 USD -- do not inflate. $0 base case means arm B pays the "
             "full device cost with no offset, which biases AGAINST ciNPWT and is in "
             "the conservative-bias register. The $0-$50 sensitivity charges it as a "
             "cycle-0 toll in ARM A. A2 NOTE: Mueller 2026 [9] independently costs "
             "conventional dressings at $0 on the same conservatism rationale so the "
             "assumption is now corroborated rather than free-standing.")
    c_decol_bundle = Param(
        "c_decol_bundle", 60, 40, 80, "gamma",
        "Kline 2018 [1] (derived: midpoint of the implied $40-$80)",
        verify=False, evidence_status="derived", price_year=2025,
        note="ALREADY 2025 USD -- do not inflate. CANCELS from every base-case result "
             "(decolonization is background in both arms); used only in the "
             "decolonization secondary analysis and the budget-impact table. Immaterial "
             "to every headline number, hence verify=False. A2: Kline 2018 [1] has no "
             "full text so this parameter remains unverifiable -- one of only two such "
             "in the model and it touches no headline result.")

    # ---- Medicare perspective scenario -- 2019 USD, MUST be inflated ------- #
    c_ssi_sup_0_6mo_mcr = Param(
        "c_ssi_sup_0_6mo_mcr", 11044, 6690, 15716, "gamma",
        "Edmiston 2023 [12] Medicare", verify=False, evidence_status="reported",
        price_year=2019, source_locator="Table 3, PDF p. 7 (Infect Control Hosp Epidemiol 2023;44(1):94)",
        note="2019 USD -- inflate via cost_2025(). A2-CORRECTED LABEL: this is NOT the "
             "only Medicare superficial window Edmiston reports -- all three are "
             "reported. Range is the published 95% CI ($6,690-$15,716); at n=585 this "
             "is the least precise cell in Table 3 (implied CV 0.209), the one place "
             "where the retired +/-50% assumption was nearly right.")
    c_ssi_sup_6_12mo_mcr = Param(
        "c_ssi_sup_6_12mo_mcr", 6923, 4457, 9389, "gamma",
        "Edmiston 2023 [12] Medicare (derived)", verify=False,
        evidence_status="derived", price_year=2019, source_locator="Table 3, PDF p. 7 (Infect Control Hosp Epidemiol 2023;44(1):94)",
        note="2019 USD -- inflate via cost_2025(). **A2 CORRECTION, +112%.** This build "
             "claimed the Medicare 12-month superficial value was NOT REPORTED and "
             "extrapolated it as 11,044 x (26,937/20,800) = 14,303, tranche 3,259. It "
             "IS reported -- $17,967 -- in the Abstract, the Results, Table 3 and the "
             "Discussion. MARGINAL tranche 17,967 - 11,044 = 6,923. The extrapolation "
             "also failed in principle: Medicare payments grow FASTER than commercial "
             "from 6 to 12 months (1.627 vs 1.295), so a transferred commercial ratio "
             "understates them. Range via SE = sqrt(2624.0^2 - 2302.6^2) = 1258.4.")
    c_ssi_sup_12_24mo_mcr = Param(
        "c_ssi_sup_12_24mo_mcr", 6129, 1648, 10610, "gamma",
        "Edmiston 2023 [12] Medicare (derived)", verify=False,
        evidence_status="derived", price_year=2019, source_locator="Table 3, PDF p. 7 (Infect Control Hosp Epidemiol 2023;44(1):94)",
        note="2019 USD -- inflate via cost_2025(). **A2 CORRECTION, +96%.** Was "
             "extrapolated to 3,124 on a false NOT-REPORTED premise. Reported 24-month "
             "Medicare superficial cumulative is $24,096, so the MARGINAL tranche is "
             "24,096 - 17,967 = 6,129. The build's implied cumulative of $17,427 was "
             "$6,669 (27.7%) too low. Range via SE = sqrt(3480.1^2 - 2624.0^2) = 2286.0; "
             "this is the widest relative range in the table and is honest about a cell "
             "estimated on 585 Medicare superficial cases.")
    c_ssi_deep_0_6mo_mcr = Param(
        "c_ssi_deep_0_6mo_mcr", 48662, 45251, 52209, "gamma",
        "Edmiston 2023 [12] Medicare", verify=False, evidence_status="reported",
        price_year=2019, source_locator="Table 3, PDF p. 7 (Infect Control Hosp Epidemiol 2023;44(1):94)",
        note="2019 USD -- inflate via cost_2025(). Range is the published 95% CI. Also "
             "the DENOMINATOR of the attribution factor k_attrib_pennington, which A2 "
             "re-derived on a same-basis footing to 0.492 (was 0.427).")
    c_ssi_deep_6_12mo_mcr = Param(
        "c_ssi_deep_6_12mo_mcr", 5095, 3382, 6808, "gamma",
        "Edmiston 2023 [12] Medicare (derived)", verify=False, evidence_status="derived",
        price_year=2019, source_locator="Table 3, PDF p. 7 (Infect Control Hosp Epidemiol 2023;44(1):94)",
        note="2019 USD -- inflate via cost_2025(). MARGINAL tranche 53,757 - 48,662. "
             "A2-VERIFIED: range via SE = sqrt(1978.6^2 - 1775.0^2) = 874.1.")
    c_ssi_deep_12_24mo_mcr = Param(
        "c_ssi_deep_12_24mo_mcr", 20046, 16097, 23995, "gamma",
        "Edmiston 2023 [12] Medicare (derived)", verify=False, evidence_status="derived",
        price_year=2019, source_locator="Table 3, PDF p. 7 (Infect Control Hosp Epidemiol 2023;44(1):94)",
        note="2019 USD -- inflate via cost_2025(). MARGINAL tranche 73,803 - 53,757. "
             "A2-VERIFIED: range via SE = sqrt(2824.0^2 - 1978.6^2) = 2015.0. Weighted "
             "Medicare 24-month cost per SSI = 0.745 x 73,803 + 0.255 x 24,096 = "
             "$61,128 (2019 USD), up from the $59,258 this build reported on the "
             "extrapolated superficial series.")


# --------------------------------------------------------------------------- #
# 9. util -- utilities (4 parameters)                                          #
#    DECREMENTS ARE POSITIVE MAGNITUDES, SUBTRACTED FROM u_well IN THE ENGINE. #
# --------------------------------------------------------------------------- #
class util:
    """Health-state utility and the tunnel decrements. proposed-inputs.md section 5."""

    u_well = Param(
        "u_well", 0.70, 0.55, 0.80, "beta",
        "Gates 2021 [22]", verify=False, evidence_status="reported",
        source_locator="Table 3 row 'Lumbar Fusion', PDF p. 6 (J Clin Neurosci 2021;93:165)",
        note="**A2: THE LARGEST LABELLED ASSUMPTION IN THIS MODEL IS RETIRED, AND THE "
             "VALUE DOES NOT CHANGE.** This build carried u_well = 0.70 as a "
             "derived_assumption with the source string 'NR -- no absolute post-fusion "
             "EQ-5D in the evidence base'. That string is false. Gates 2021 Table 3 "
             "prints a 'Lumbar Fusion' row -- 6 studies 782 patients -- reading "
             "0.45 preoperative -> 0.70 postoperative gain 0.25 under a column headed "
             "'Postoperative EQ-5D Index at 1-Year Mean (SD)'. Column assignment was "
             "proved geometrically (the header token and the 0.70 share x0 = 356.1 to "
             "0.1 pt while 'QALY Gained' is a separate column at x0 = 509.5) and "
             "arithmetically (0.70 - 0.45 = 0.25 and the identity holds for all five "
             "rows). Patient-weighted pooling of the six studies reproduces 0.45/0.70/"
             "0.25 to two decimals. Corroborated independently by Pennington 2019 [14] "
             "Table 2 whose matched NON-INFECTED arm reads 0.67 at 6 months in exactly "
             "this model's population -- agreement within 0.03 from an independent "
             "design and in the expected direction. So this is a PROVENANCE correction "
             "not a numeric one and must not be filed alongside value changes.\n"
             "RANGE CORRECTED 0.60-0.80 -> 0.55-0.80: the old 0.60 floor was sourced by "
             "nobody. Both bounds are now printed cells -- 0.55 from Rivero-Arias "
             "(n = 176) and 0.80 from the two US studies in the same pool. CAVEATS that "
             "stay: Gates' 0.70 is an all-comers mean including infected patients; the "
             "'at 1-Year' header is unreliable because ~53% of the fusion patient-weight "
             "carries an undefined superscript mapping to 2-year studies and Gates "
             "defines no footnote legend anywhere; and no EQ-5D version or country value "
             "set is stated in either source so this must NOT be labelled a US-tariff "
             "utility. Blind spot-check (2026-08-17) adds: Gates prints NO confidence "
             "interval and no heterogeneity statistic for this cell, its six constituent "
             "study values span 0.49-0.80, and a single study carries about 47% of the "
             "patient weight -- so 0.70 is a weighted point estimate with no published "
             "uncertainty, which is why the 0.55-0.80 range is assembled from other "
             "printed cells rather than from a reported interval. u_well is identical in both arms and enters the incremental QALY "
             "only through the small fraction of cohort-time that differs so it scales "
             "each arm's QALY LEVEL but very nearly cancels from the DIFFERENCE. "
             "Base case: RECOV also takes u_well (full recovery).")
    du_sup = Param(
        "du_sup", 0.08, 0.05, 0.11, "normal",
        "McFarland 2023 [19]", verify=False, evidence_status="derived",
        source_locator="Fig. 2, PDF p. 4 (Br J Surg 2023;110(8):945)",
        note="POSITIVE MAGNITUDE, SUBTRACTED from u_well by the 3-month superficial "
             "tunnel. Pooled ANY-SSI EQ-5D mean difference -0.08 (95% CI -0.11 to "
             "-0.05). Anti-conservative caveat (run_log objection 2): the pooled "
             "any-SSI figure already contains deep SSI, so using it as the "
             "superficial-specific decrement is slightly generous to ciNPWT -- "
             "immaterial at under 0.01 percentage points of threshold, but disclose it. "
             "**A2: the run-log objection is UPHELD, not retired, and is now quantified.** "
             "McFarland ran no superficial subgroup meta-analysis ('No data were "
             "available for the other planned subgroup analyses') so there is no pooled "
             "superficial value. But a single-study superficial estimate DOES exist: "
             "Guirro (ref 43) -0.03 (95% CI -0.09 to 0.03) k = 1 TKA >=12 months, "
             "statistically null -- 2.7x smaller than the 0.08 substituted for it. The "
             "pooled -0.08 is 36% weighted by deep-SSI estimates. 0.08 remains "
             "defensible for this model's ACUTE 3-cycle superficial tunnel (McFarland's "
             "<4-month all-type band is -0.09 -0.14 to -0.04) but a scenario at 0.03 is "
             "now mandatory because the substitution biases toward the intervention. "
             "PSA SE = 0.06/3.92 = 0.0153. Do NOT sample independently of du_deep: "
             "Fig. 3's deep pool is a strict subset of Fig. 2's any-SSI pool.")
    du_deep = Param(
        "du_deep", 0.10, 0.06, 0.14, "normal",
        "McFarland 2023 [19]", verify=False, evidence_status="reported",
        source_locator="Fig. 3, PDF p. 4 (Br J Surg 2023;110(8):945)",
        note="POSITIVE MAGNITUDE, SUBTRACTED from u_well by the 12-month deep tunnel. "
             "Deep-SSI EQ-5D mean difference -0.10 (95% CI -0.14 to -0.06), I2=0%. "
             "Implied QALY loss per SSI = 0.742 x (0.10 x 12/12) + 0.258 x "
             "(0.08 x 3/12) = 0.0794 i.e. ~$7,936 at WTP $100,000/QALY -- the second "
             "half of the threshold arithmetic (0.745/0.255 after the A2 split "
             "correction gives 0.0796). **A2-CONFIRMED EXACTLY.** PSA SE = 0.08/3.92 = "
             "0.0204; normal is defensible (P(draw < 0) ~ 1e-7). Comparator is a "
             "contemporaneous NON-INFECTED surgical control group at similar time "
             "points -- not a pre-infection baseline and not population norms -- so "
             "subtracting from u_well is the right operation. Instruments EQ-5D SF-36 "
             "SF-12 all mapped to EQ-5D on the UK Dolan tariff which in a US-costed "
             "model is a disclosed mismatch. ONE lumbar-fusion study contributes to "
             "neither pooled estimate: of six spine studies in the review only Haddad "
             "(adult spinal deformity) entered the meta-analysis, the rest excluded for "
             "not reporting SD/SE/CI. The authors warn the estimates 'may not be "
             "appropriate for use in other surgical settings.'\n"
             "**BLIND SPOT-CHECK FINDING (2026-08-17), disclosed rather than buried.** "
             "'Pooled' here means TWO studies -- Haddad (adult spinal deformity) and "
             "Parker (open lower-limb fracture), roughly 46 infected patients -- entered "
             "as SEVEN non-independent repeated measures of the same patients. The "
             "I2 = 0% is therefore substantially an artifact of that double-counting "
             "rather than evidence of consistency between studies, and the CI is "
             "correspondingly too narrow. McFarland's own text says the deep analysis "
             "used 'five time points' while Fig. 3 plots seven with Q(6) -- cite the "
             "figure. This is the weakest evidential footing of any base-case parameter "
             "in the model and Methods must say so.")
    du_recovered_residual = Param(
        "du_recovered_residual", 0.00, 0.00, 0.06, "fixed",
        "Urquhart 2021 [21] base case / McFarland 2023 [19] scenario",
        verify=False, evidence_status="derived",
        source_locator="base: Urquhart Table III PDF p. 6; scenario: McFarland Fig. 4 PDF p. 5 (Br J Surg 2023;110(8):946)",
        note="POSITIVE MAGNITUDE. **BASE CASE 0.00 A2-CONFIRMED; SCENARIO CORRECTED "
             "0.02 -> 0.06.** Base case: Urquhart 2021 -- prospective and matched to "
             "exactly this model's population -- finds all five PROMs null at 24 months "
             "with EVERY point estimate favouring the SSI group (ODI -0.7 leg -0.5 back "
             "-0.2 PCS +1.1 MCS +1.0; smallest P = 0.249) so full recovery in RECOV "
             "stands and the status rises from derived_assumption to derived. Scenario: "
             "McFarland's >=12-month pooled decrement is -0.06 (95% CI -0.10 to -0.02) "
             "I2 = 0.02% k = 5. **0.02 was the OPTIMISTIC EDGE of that CI, not its point "
             "estimate**, and a scenario constant should carry a point estimate. "
             "CAVEATS: McFarland's >=12-month band is ALL SSI TYPES, not deep-specific, "
             "and 44% of its weight is the review's single SUPERFICIAL study; deep-only "
             "rows in that band re-pool to about -0.088 and McFarland's published "
             "deep-specific figure is -0.10 with NO time attenuation (Q(6) = 2.09 "
             "P = 0.91). Equally, McFarland establishes NO time trend at all -- the test "
             "of subgroup differences is Qb(2) = 1.73 P = 0.42 and the band profile is "
             "non-monotone (-0.09 -> -0.11 -> -0.06) -- so no decay argument supports "
             "either 0.02 or 0.06; 0.06 is chosen as the printed >=12-month point "
             "estimate. The Urquhart-vs-McFarland conflict is real (population-matched "
             "PROMs vs a cross-surgical utility meta-analysis with essentially no lumbar "
             "fusion in it) and is the only reason RECOV is kept distinct from WELL.")


# --------------------------------------------------------------------------- #
# 10. mortality -- perioperative death and cohort composition (3 parameters)   #
#     Background mortality itself is CSV-backed: background_mortality() above.  #
# --------------------------------------------------------------------------- #
class mortality:
    """Perioperative mortality and the cohort characteristics that index the life
    table. proposed-inputs.md sections 0 and 3.6.

    NOTE: this holder MUST be in the all_params() tuple -- Table 1 is built from
    all_params(), and the superseded (tp, cost, util) tuple would have silently
    omitted these three, two of which are now sourced to Urquhart 2021 [21] at A2.
    """

    p_periop_death = Param(
        "p_periop_death", 0.0015, 0.0012, 0.0026, "beta",
        "Pumberger 2012 [24]", verify=False, evidence_status="derived",
        source_locator="Table 1, PDF p. 4 (Eur Spine J 2012;21:1636)",
        note="**A2 CORRECTION 0.0020 -> 0.0015.** Pumberger prints '0.2%' five times so "
             "the provenance was not invented -- but it prints the count in the same "
             "clause: 'Of those 0.2% (n = 1938) patients died during their in-hospital "
             "stay.' Against Table 1's denominators that is 1938/1285934 = 0.1507% "
             "(unweighted check 393/264728 = 0.1485%). For 1938 deaths to equal exactly "
             "0.200% the denominator would have to be 969000 which appears nowhere. "
             "0.2% is a one-significant-figure display of 0.1507% and 0.002 overstated "
             "the paper's own countable rate by 32.7% -- it cannot be carried as a "
             "'reported' transcription in a published inputs table. DSA low bound "
             "0.0014 -> 0.0012 from the ELECTIVE-ONLY subgroup derivable from Table 1's "
             "admission-type strata: 1374/1158254 = 0.1186% (rounding envelope only "
             "+/-0.13% relative) which matters because this model's population is "
             "elective and Pumberger's is all-comers (90.1% elective). COUNTERVAILING "
             "BIAS stated plainly: the estimand is IN-HOSPITAL index-admission mortality "
             "and the authors warn it 'almost certainly represents an underestimation'. "
             "The high bound 0.0026 is retained to hold that headroom. MATERIALITY: the "
             "published tornado ranks this 12th of 13 with a $3.01 swing on a $2251 base "
             "INMB -- a reporting-integrity correction not a results-changing one. "
             "n=1,288,496 primary posterior lumbar fusion (ICD-9-CM 81.08) NIS "
             "1998-2008. Applied in CYCLE 0 ONLY and "
             "IDENTICALLY in both arms, so it very nearly cancels from the incremental "
             "result. Range low = the elective-only subgroup 0.0012 (was Goz 2013 [25] "
             "0.0014, which has no full text and stays unverifiable); range high = a symmetric "
             "reflection of the Goz bound about the base (0.0020 + 0.0006 = 0.0026) "
             "because no higher published estimate exists in the evidence base -- an "
             "explicit construction, not a free-hand number. NO SSI-ATTRIBUTABLE "
             "MORTALITY DIFFERENCE is assumed anywhere; that omission is conservative "
             "against ciNPWT and A2 confirms it is also the right call -- Pumberger "
             "contains no SSI data at all (the words infection, wound and surgical site "
             "appear nowhere) and its in-hospital SEPSIS case fatality of 10.0% must NOT "
             "be transplanted as an SSI mortality increment: it is index-admission "
             "multi-organ failure overlapping the same patients' 43.7% pulmonary and "
             "36.2% cardiac complications and using it would make SSI roughly 66x more "
             "lethal than the entire perioperative period.")
    age_start = Param(
        "age_start", 60, 54, 70, "fixed",
        "Urquhart 2021 [21]", verify=False, evidence_status="derived",
        source_locator="Table I, PDF p. 4 (J Bone Joint Surg Am 2021;103:2108)",
        note="**A2: THE VALUE STANDS AT 60 AND THE PROVENANCE CHANGES.** This build "
             "carried age as not_reported. Three sources report a cohort age and they "
             "disagree so the choice is which POPULATION indexes the life table. "
             "Urquhart 2021 is the only source that both matches this model's population "
             "(open posterior instrumented thoracolumbar/lumbar fusion for degenerative "
             "disease, tumour and trauma excluded) and prints a real mean with an SD: "
             "pooled 59.33 y (SD 14.17) from 456 non-infected at 59.4 +/- 14.3 and 79 "
             "infected at 58.9 +/- 13.5, which rounds to the 60 already in use. The two "
             "competing candidates were examined and REJECTED as the base: Edmiston "
             "reports NO mean age at all -- only seven bands, from which a mean of "
             "54.2-54.7 can be reconstructed depending on the completed-years convention "
             "-- and its cohort is 49.1% pure-anterior i.e. substantially ACDF-like and "
             "younger than a posterior lumbar fusion population; Pumberger prints 54.3 y "
             "but on a 1998-2008 all-comers NIS cohort that is 27.8% under age 45 and "
             "includes trauma tumour and scoliosis. Adopting ~54 would have moved a "
             "lifetime-horizon parameter 5.8 years in the wrong direction. DSA range "
             "55-70 -> 54-70 so the span now REACHES the Edmiston/Pumberger values "
             "rather than excluding them. Indexes the life table via "
             "background_mortality(). Near-immaterial to the incremental result: with "
             "background-only mortality and no SSI-attributable mortality difference age "
             "enters identically in both arms -- the published tornado ranks it 11th of "
             "13 at a $5.33 swing. qx(60) = 0.011457 male / 0.007034 female (2023 US "
             "life table).")
    frac_male = Param(
        "frac_male", 0.533, 0.44, 0.55, "fixed",
        "Urquhart 2021 [21]", verify=False, evidence_status="derived",
        source_locator="Table I, PDF p. 4 (J Bone Joint Surg Am 2021;103:2108)",
        note="**A2 CORRECTION 0.50 -> 0.533.** Was carried as not_reported; three "
             "sources report it. Urquhart 2021 is chosen on the same population-match "
             "grounds as age_start: 535 - 250 female = 285 male, 285/535 = 0.5327, in "
             "open posterior instrumented thoracolumbar fusion for degenerative disease. "
             "The assumed 0.40-0.60 range is replaced by 0.44-0.55, a SOURCED span: "
             "Edmiston 2023 [12] Table 1a gives 94241/210019 = 0.449 and Pumberger 2012 "
             "[24] Table 1 gives 0.445, and those two converge independently on the low "
             "bound. The default sex_mix for background_mortality(). Sex mix moves "
             "background mortality by roughly a third in relative terms at age 60 and is "
             "worth one DSA row even though it very nearly cancels between arms -- the "
             "published tornado ranks it LAST of 13 at a $0.37 swing.")


# --------------------------------------------------------------------------- #
# 11. Registry                                                                 #
# --------------------------------------------------------------------------- #
#: The four parameter holders, in the order all_params() iterates them.
#: There is NO `cpi` holder -- CPI is CSV-backed and owns no Param objects
#: (BUILD-CONTRACT section 7, superseding blockquote of 2026-08-09).
PARAM_HOLDERS = (tp, cost, util, mortality)


def all_params() -> list[Param]:
    """Every Param across (tp, cost, util, mortality), in declaration order."""
    out: list[Param] = []
    for holder in PARAM_HOLDERS:
        for key in vars(holder):
            v = getattr(holder, key)
            if isinstance(v, Param):
                out.append(v)
    return out


def params_needing_verification() -> list[Param]:
    """The Phase-A2 audit list: every Param still gating on a full-text check."""
    return [p for p in all_params() if p.verify]


def param_by_name(name: str) -> Param:
    """Look up a Param by its `name` field across all four holders."""
    for p in all_params():
        if p.name == name:
            return p
    raise KeyError(f"no parameter named {name!r}")


# --------------------------------------------------------------------------- #
# 12. Settings and SETTINGS (BUILD-CONTRACT sections 3 and 4)                  #
# --------------------------------------------------------------------------- #
#: Sweep axis 1 -- baseline 90-day all-SSI risk. 291 points, step 0.0005.
RISK_GRID = np.round(np.arange(0.005, 0.15 + 1e-12, 0.0005), 10)
#: Sweep axis 2 -- ciNPWT unit cost (2025 USD). 161 points, step $5.
#: A2 WIDENED $600 -> $900. Mueller 2026 [9] publishes 2023 LIST prices of $495 for the
#: Prevena Peel-and-Place 20 cm system and $830 for the Prevena Plus Customizable system.
#: The old ceiling put the second product OFF the reported axis, and at multilevel open
#: posterior lumbar fusion the incision frequently exceeds 20 cm, which makes the
#: customizable system the product-appropriate one. Both list prices now sit on the axis.
PRICE_GRID = np.round(np.arange(100.0, 900.0 + 1e-12, 5.0), 10)
#: Sweep axis 3 -- ciNPWT relative risk. 61 points, step 0.01.
RR_GRID = np.round(np.arange(0.40, 1.00 + 1e-12, 0.01), 10)
#: CEAC willingness-to-pay grid, $0-$200,000 step $2,500.
CEAC_GRID = np.arange(0, 200_001, 2_500)

#: Reported risk strata (Losina idiom).
#: A2 CORRECTION: low 0.015 -> 0.020. See RISK_ANCHORS below.
STRATA = {"low": 0.020, "intermediate": 0.066, "high": 0.114}

#: Labelled anchors on the risk axis (plotted on Figs 2-4). These do NOT share
#: p_base_ssi's 90-day ascertainment window -- Edmiston is 3-180 d and Mueller 2021 is
#: 60 d, while Freire-Archer states NO window anywhere -- and they are plotted as
#: positions without being rescaled to a common window. Methods must say so.
#:
#: A2 CORRECTION, low anchor 0.015 -> 0.020. Freire-Archer pools DEEP SSI ONLY: the word
#: "superficial" does not appear anywhere in its 17 pages in an SSI sense, inclusion
#: required a reported DEEP SSI outcome, and the extracted field is "No. deep SSI".
#: Plotting a deep-only rate on an ALL-SSI axis was a category error. Rescaled onto this
#: model's own composition: 0.015 / 0.745 = 0.0201 -> 0.020, which is on the risk grid.
#: TWO CAVEATS THAT STAY. (1) 1.5% is the TRIM-AND-FILL BIAS-ADJUSTED estimate; the
#: observed random-effects value is 2.2% (1.8-2.8, 3111/209,347, I2 95.5%, k=37), which
#: would rescale to ~0.030. The adjusted value is retained for consistency with this
#: build's use of adjusted estimates elsewhere, and the observed value is carried as a
#: scenario. (2) 58.2% of Freire-Archer's denominator is two Scoliosis Research Society
#: voluntary surgeon self-report registries, both rated higher-risk-of-bias on outcome
#: ascertainment AND follow-up by the review's own CLARITY table; leaving them out raises
#: the crude rate to 1.86%, and the authors write "we suspect that the actual incidence
#: is underreported."
#:
#: Mueller 2021's 11.4% is CDC superficial + deep POOLED (all-SSI) over 60 days, so it
#: belongs on this axis with Edmiston's 6.6% and NOT on a deep-only axis; its counts are
#: 14/123, not the 18/156 this build recorded.
RISK_ANCHORS = [
    (0.020, "Freire-Archer pooled deep SSI, rescaled to all-SSI", "[17]"),
    (0.066, "Edmiston all-SSI", "[12]"),
    (0.114, "Mueller 2021 control (14/123, instrumented, 60 d)", "[6]"),
]

# Dirichlet concentration for the ssi_split pair (frac_deep, frac_sup).
# Method of moments on the plan's own DSA range, using the SE = (hi - lo)/3.92 idiom:
#     SE    = (0.80 - 0.30) / 3.92        = 0.127551
#     n_eff = 0.745 x 0.255 / SE**2 - 1   = 10.767
#     alpha = (0.745 x 10.767, 0.255 x 10.767) = (8.02, 2.75)
# NOT the raw Edmiston counts (~10,291 / ~3,570), which would make PSA treat the split
# as known when the real uncertainty is claims-coding validity.
DIRICHLET_KAPPA_SSI_SPLIT = 10.767
DIRICHLET_ALPHA_SSI_SPLIT = (
    round(0.745 * DIRICHLET_KAPPA_SSI_SPLIT, 4),
    round(0.255 * DIRICHLET_KAPPA_SSI_SPLIT, 4),
)


class Settings:
    """Frozen run settings (BUILD-CONTRACT section 3). `SETTINGS` mirrors these as a
    dict; both are exported and must stay in step."""

    perspective: str = "US commercial payer (payments basis)"
    price_year: int = PRICE_YEAR

    cycle_years: float = 1.0 / 12.0       # 1 month
    horizon_cycles: int = 24              # 2-year base case
    horizons = (3, 12, 24, 60)            # 90 d / 1 y / 2 y (base) / 5 y
    half_cycle: bool = True               # trapezoidal, DWELL streams only

    disc_cost: float = 0.03
    disc_qaly: float = 0.03
    disc_scenarios = (0.0, 0.05)

    wtp_primary: float = 100_000.0
    wtp_report = (50_000.0, 150_000.0)
    ceac_grid = CEAC_GRID

    psa_iterations: int = 10_000
    seed: int = 12345                     # np.random.default_rng(12345) everywhere

    #: Cost tolls fall at onset + these cycle offsets (BUILD-CONTRACT section 6.2).
    #: Onset sits at cycles 1-3, so the three tranches land at 1-3, 7-9 and 13-15.
    toll_offset_cycles = (0, 6, 12)

    #: Sweep axes.
    risk_grid = RISK_GRID
    price_grid = PRICE_GRID
    rr_grid = RR_GRID
    strata = STRATA
    risk_anchors = RISK_ANCHORS


SETTINGS: dict = {
    # ---- perspective / currency ---- #
    "perspective": Settings.perspective,
    "price_year": PRICE_YEAR,
    # ---- time ---- #
    "cycle_length_months": 1,
    "cycle_years": Settings.cycle_years,
    "horizon_cycles": Settings.horizon_cycles,
    "horizons": Settings.horizons,
    "half_cycle": Settings.half_cycle,
    "toll_offset_cycles": Settings.toll_offset_cycles,
    # ---- discounting ---- #
    "disc_cost": Settings.disc_cost,
    "disc_qaly": Settings.disc_qaly,
    "disc_scenarios": Settings.disc_scenarios,
    # ---- decision rule ---- #
    "wtp_primary": Settings.wtp_primary,
    "wtp_report": Settings.wtp_report,
    "ceac_grid": CEAC_GRID,
    # ---- stochastic ---- #
    "psa_iterations": Settings.psa_iterations,
    "seed": Settings.seed,
    "dirichlet_kappa_ssi_split": DIRICHLET_KAPPA_SSI_SPLIT,
    "dirichlet_alpha_ssi_split": DIRICHLET_ALPHA_SSI_SPLIT,
    # ---- the three sweep axes (BUILD-CONTRACT section 4) ---- #
    "risk_grid": RISK_GRID,
    "price_grid": PRICE_GRID,
    "rr_grid": RR_GRID,
    "strata": STRATA,
    "risk_anchors": RISK_ANCHORS,
    # ---- cohort: MIRRORS of the Params, which remain the source of truth ---- #
    "age0": float(mortality.age_start),
    "sex_mix": {"M": float(mortality.frac_male), "F": 1.0 - float(mortality.frac_male)},
}


# --------------------------------------------------------------------------- #
# 13. Results banner (line 1 of every results CSV, as a '#' comment)           #
# --------------------------------------------------------------------------- #
RESULTS_BANNER = (
    "VERIFIED RESULTS -- Phase A2 full-text verification completed 2026-08-17: 46 of 48 "
    "parameters were closed out against the 14 source full texts and 43 carry a "
    "table/page locator; 2 remain unresolvable assumptions (the ciNPWT institutional net "
    "acquisition price and the superficial tunnel length). "
    "SSI/ciNPWT risk-threshold Markov CUA; US commercial payer; 2025 USD; monthly "
    "cycles; 24-cycle base horizon; 3%/yr discounting; WTP $100000/QALY."
)
#: Alias kept for import compatibility -- run scripts and the manuscript builder
#: reference this name. It is no longer a "provisional" banner.
PROVISIONAL_BANNER = RESULTS_BANNER


# --------------------------------------------------------------------------- #
# 14. Import-time structural checks                                            #
#     Every one of these guards a trap that would otherwise fail silently.     #
# --------------------------------------------------------------------------- #
def _check_imports() -> None:
    # -- slot vector ------------------------------------------------------- #
    assert len(SLOT_NAMES) == N_SLOTS == 18, f"slot vector is {len(SLOT_NAMES)}, expected 18"
    assert SLOT_NAMES[WELL] == "WELL" and SLOT_NAMES[RECOV] == "RECOV"
    assert SLOT_NAMES[DEAD] == "DEAD" and DEAD == N_SLOTS - 1
    assert SUP_START + SUP_LEN == DEEP_START, "superficial tunnel does not abut the deep tunnel"
    assert DEEP_START + DEEP_LEN == RECOV, "deep tunnel does not abut RECOV"
    assert [slot_to_clinical(s) for s in range(N_SLOTS)] == (
        [0] + [1] * SUP_LEN + [2] * DEEP_LEN + [3, 4]
    ), "slot_to_clinical does not collapse the 18 slots onto the 5 clinical states"
    assert len(CLINICAL_STATES) == 5

    # -- trap 9: tunnel-length Params must match the frozen slot vector ----- #
    assert int(tp.n_tunnel_sup.value) == SUP_LEN == 3, "n_tunnel_sup must equal SUP_LEN = 3"
    assert int(tp.n_tunnel_deep.value) == DEEP_LEN == 12, "n_tunnel_deep must equal DEEP_LEN = 12"

    # -- registry ---------------------------------------------------------- #
    params = all_params()
    assert len(params) == 48, f"expected 48 parameters, found {len(params)}"
    names = [p.name for p in params]
    assert len(names) == len(set(names)), "duplicate Param names"
    for p in params:
        assert p.name == p.name.strip() and p.name, "Param name is blank"
        assert p.dist in DISTRIBUTIONS, f"{p.name}: bad dist {p.dist!r}"
        assert p.evidence_status in EVIDENCE_STATUSES, \
            f"{p.name}: bad evidence_status {p.evidence_status!r}"
        assert p.source, f"{p.name}: empty source"
        assert p.note, f"{p.name}: empty note"
        assert p.low <= p.value <= p.high, \
            f"{p.name}: value {p.value} outside [{p.low}, {p.high}]"
    n_holders = {h.__name__: sum(1 for k in vars(h) if isinstance(getattr(h, k), Param))
                 for h in PARAM_HOLDERS}
    assert n_holders == {"tp": 24, "cost": 17, "util": 4, "mortality": 3}, n_holders
    # A2 (2026-08-17): 43 -> 7. Every parameter closed out against a full text carries a
    # non-empty source_locator; the 7 that remain are assumptions no source can resolve.
    assert len(params_needing_verification()) == 2, \
        f"expected 2 parameters still gating after A2, found {len(params_needing_verification())}"
    assert {p.name for p in params_needing_verification()} == {
        "c_device_cinpwt", "n_tunnel_sup"
    }, f"unexpected residual verify=True set: {sorted(p.name for p in params_needing_verification())}"
    # A2 AUDITABILITY GATE: verify=False must mean "checked against a full text", so every
    # such parameter needs a locator. The four parameters whose gating source has no full
    # text, or which are pure modelling constants, are exempted by name.
    #   - rr_cinpwt_attenuated: a scenario constant, not an estimate from any paper
    #   - c_dressing_standard: an assumed $0 (Mueller 2026 corroborates but does not
    #     "report" it as this model's input)
    #   - c_decol_bundle: Kline 2018 [1] has no full text
    #   - rr_readmit_sa / rr_decol_broad: Rosenthal [13] / Seidelman [28], Tier 3, not
    #     requested at A2 and gating nothing
    _locator_exempt = {"rr_cinpwt_attenuated", "c_dressing_standard", "c_decol_bundle",
                       "rr_readmit_sa", "rr_decol_broad"}
    for _p in params:
        if not _p.verify and _p.name not in _locator_exempt:
            assert _p.source_locator, (
                f"{_p.name}: verify=False but no source_locator -- a parameter cannot be "
                f"closed out without recording WHERE it was read from")

    # -- trap 1: the price-year split -------------------------------------- #
    costs = [p for p in params if p.price_year is not None]
    y2019 = [p.name for p in costs if p.price_year == 2019]
    y2025 = [p.name for p in costs if p.price_year == PRICE_YEAR]
    assert len(y2019) == 12, f"expected 12 cost params in 2019 USD, found {len(y2019)}: {y2019}"
    assert len(y2025) == 3, f"expected 3 cost params in 2025 USD, found {len(y2025)}: {y2025}"
    assert set(y2025) == {"c_device_cinpwt", "c_dressing_standard", "c_decol_bundle"}, y2025
    assert len(costs) == 15, "every dollar-valued Param must carry a price_year"
    assert all(p.price_year is None for p in params if p not in costs)
    assert cost_2025(cost.c_device_cinpwt) == 250.0, \
        "a 2025-USD parameter was inflated -- trap 1"

    # -- trap 2: marginal (already-differenced) cost tranches --------------- #
    #    A2 added the FOURTH identity: the Medicare superficial series is no longer
    #    extrapolated, so it must now close on Edmiston's reported $24,096 like the
    #    other three.
    assert 20800 + 6137 + 5884 == 32821, "superficial tranches do not sum to $32,821"
    assert 59766 + 15109 + 18866 == 93741, "deep tranches do not sum to $93,741"
    assert 48662 + 5095 + 20046 == 73803, "Medicare deep tranches do not sum to $73,803"
    assert 11044 + 6923 + 6129 == 24096, "Medicare superficial tranches do not sum to $24,096"
    assert (cost.c_ssi_sup_0_6mo.value + cost.c_ssi_sup_6_12mo.value
            + cost.c_ssi_sup_12_24mo.value) == 32821
    assert (cost.c_ssi_deep_0_6mo.value + cost.c_ssi_deep_6_12mo.value
            + cost.c_ssi_deep_12_24mo.value) == 93741
    assert (cost.c_ssi_deep_0_6mo_mcr.value + cost.c_ssi_deep_6_12mo_mcr.value
            + cost.c_ssi_deep_12_24mo_mcr.value) == 73803
    assert (cost.c_ssi_sup_0_6mo_mcr.value + cost.c_ssi_sup_6_12mo_mcr.value
            + cost.c_ssi_sup_12_24mo_mcr.value) == 24096

    # -- trap 3: decrements are stored as POSITIVE magnitudes --------------- #
    for p in (util.du_sup, util.du_deep, util.du_recovered_residual):
        assert p.value >= 0.0 and p.low >= 0.0, f"{p.name} must be a positive magnitude"

    # -- splits and weights ------------------------------------------------- #
    assert abs(tp.frac_deep.value + tp.frac_sup.value - 1.0) < 1e-12, \
        "frac_deep + frac_sup must equal 1"
    assert tp.frac_deep.group == tp.frac_sup.group == "ssi_split"
    for triple in ((tp.w_onset_deep_m1, tp.w_onset_deep_m2, tp.w_onset_deep_m3),
                   (tp.w_onset_sup_m1, tp.w_onset_sup_m2, tp.w_onset_sup_m3)):
        s = sum(float(x) for x in triple)
        assert abs(s - 1.0) < 1e-9, f"onset weights sum to {s}, expected 1"

    # -- grids: a floating-point arange endpoint is the classic silent bug --- #
    assert len(RISK_GRID) == 291, f"risk_grid has {len(RISK_GRID)} points, expected 291"
    assert len(PRICE_GRID) == 161, f"price_grid has {len(PRICE_GRID)} points, expected 161"
    assert len(RR_GRID) == 61, f"rr_grid has {len(RR_GRID)} points, expected 61"
    assert len(CEAC_GRID) == 81, f"ceac_grid has {len(CEAC_GRID)} points, expected 81"
    assert (RISK_GRID[0], RISK_GRID[-1]) == (0.005, 0.15), "risk_grid endpoints moved"
    assert (PRICE_GRID[0], PRICE_GRID[-1]) == (100.0, 900.0), "price_grid endpoints moved"
    assert (RR_GRID[0], RR_GRID[-1]) == (0.40, 1.00), "rr_grid endpoints moved"
    for label, p in STRATA.items():
        assert np.any(np.isclose(RISK_GRID, p)), f"stratum {label} (p={p}) is off the risk grid"
    assert [a[0] for a in RISK_ANCHORS] == [0.020, 0.066, 0.114]
    assert float(cost.c_device_cinpwt.low) == PRICE_GRID[0]
    assert float(cost.c_device_cinpwt.high) == PRICE_GRID[-1]
    assert float(tp.p_base_ssi.low) == RISK_GRID[0]
    assert float(tp.p_base_ssi.high) == RISK_GRID[-1]

    # -- banner ------------------------------------------------------------- #
    assert "\n" not in RESULTS_BANNER and "\r" not in RESULTS_BANNER, \
        "RESULTS_BANNER must be one line"
    assert "," not in RESULTS_BANNER, "RESULTS_BANNER must contain no commas (CSV safety)"
    # A2: the banner no longer claims the results are provisional, and must not.
    assert "PROVISIONAL" not in RESULTS_BANNER.upper(), \
        "RESULTS_BANNER still calls the results PROVISIONAL after A2 verification"
    assert "pdf/ is empty" not in RESULTS_BANNER
    assert "VERIFIED" in RESULTS_BANNER and "A2" in RESULTS_BANNER


_check_imports()


# --------------------------------------------------------------------------- #
# 15. Self-report                                                              #
# --------------------------------------------------------------------------- #
def _self_report() -> None:
    params = all_params()
    flagged = params_needing_verification()
    print("inputs.py -- SSI/ciNPWT risk-threshold Markov CUA (A2-VERIFIED 2026-08-17)")
    print("-" * 76)
    print(f"Parameters      : {len(params)} total across "
          f"{', '.join(h.__name__ for h in PARAM_HOLDERS)}")
    for h in PARAM_HOLDERS:
        n = sum(1 for k in vars(h) if isinstance(getattr(h, k), Param))
        nv = sum(1 for k in vars(h)
                 if isinstance(getattr(h, k), Param) and getattr(h, k).verify)
        print(f"                  {h.__name__:<10s} {n:>3d} params   {nv:>3d} need A2 verification")
    print(f"Needing verify  : {len(flagged)} of {len(params)} "
          f"(the {len(params) - len(flagged)} exceptions: "
          f"{', '.join(p.name for p in params if not p.verify)})")
    print(f"Slot vector     : N_SLOTS={N_SLOTS}, len(SLOT_NAMES)={len(SLOT_NAMES)}, "
          f"WELL={WELL} SUP={SUP_START}..{SUP_START + SUP_LEN - 1} "
          f"DEEP={DEEP_START}..{DEEP_START + DEEP_LEN - 1} RECOV={RECOV} DEAD={DEAD}")
    print(f"Sweep grids     : risk {len(RISK_GRID)} pts [{RISK_GRID[0]:g}, {RISK_GRID[-1]:g}] | "
          f"price {len(PRICE_GRID)} pts [{PRICE_GRID[0]:g}, {PRICE_GRID[-1]:g}] | "
          f"rr {len(RR_GRID)} pts [{RR_GRID[0]:g}, {RR_GRID[-1]:g}]")
    print(f"CEAC grid       : {len(CEAC_GRID)} pts [{CEAC_GRID[0]:,}, {CEAC_GRID[-1]:,}] "
          f"step {CEAC_GRID[1] - CEAC_GRID[0]:,}")

    f = inflation_factor(2019)
    cpi = load_cpi()
    print(f"CPI med-care    : {min(cpi)}-{max(cpi)}; 2019={cpi[2019]:.3f} "
          f"{PRICE_YEAR}={cpi[PRICE_YEAR]:.3f}; 2019->{PRICE_YEAR} factor {f:.6f}")

    w2019 = (float(tp.frac_deep) * 93741.0) + (float(tp.frac_sup) * 32821.0)
    print(f"Weighted 24-mo cost per SSI: ${w2019:,.0f} (2019 USD) -> "
          f"${inflate(w2019, 2019):,.0f} ({PRICE_YEAR} USD)   [target $78,024 / $90,812]")
    print(f"Price-year split: cost_2025(c_device_cinpwt)   = "
          f"{cost_2025(cost.c_device_cinpwt):,.2f}  (2025 USD -- NOT inflated)")
    print(f"                  cost_2025(c_ssi_deep_0_6mo) = "
          f"{cost_2025(cost.c_ssi_deep_0_6mo):,.2f}  (2019 USD -- inflated)")

    a0 = int(mortality.age_start.value)
    lt = load_life_table()
    print(f"Life table      : ages {int(lt.index.min())}-{int(lt.index.max())}; "
          f"qx({a0}) male={lt.loc[a0, 'qx_male']:.6f} female={lt.loc[a0, 'qx_female']:.6f} "
          f"blended={_qx_blended(a0, float(mortality.frac_male)):.6f}")
    print(f"                  background_mortality({a0}, t=1)    = "
          f"{background_mortality(a0, t=1.0):.8f}")
    print(f"                  background_mortality({a0}, t=1/12) = "
          f"{background_mortality(a0, t=1 / 12):.8f}   "
          f"(qx/12 would be {_qx_blended(a0, float(mortality.frac_male)) / 12:.8f})")
    print("Import checks   : PASS (48 params; 12/3 price-year split; marginal-tranche "
          "identities; 291/101/61 grids; 18-slot vector; one-line banner)")
    print("-" * 76)
    print("BANNER: " + RESULTS_BANNER)
    print("-" * 76)
    print(f"STILL NEEDING A2 FULL-TEXT VERIFICATION ({len(flagged)}):")
    for p in flagged:
        py = f" [{p.price_year} USD]" if p.price_year is not None else ""
        print(f"  [VERIFY] {p.name:<24s} = {p.value:<10g} {p.evidence_status:<20s} "
              f"{p.source}{py}")


if __name__ == "__main__":
    _self_report()
