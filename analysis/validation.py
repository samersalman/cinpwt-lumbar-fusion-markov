"""
validation.py -- Internal validation suite for the SSI / ciNPWT risk-threshold Markov CUA.

Adapted (structure only) from `endoscopic vs MIS-TLIF/v2/analysis/validation.py`: the
HARD-vs-SANITY split and the PASS/FAIL report rendering. Every number here is this
project's own, computed live against `model.py` / `inputs.py`.

=============================================================================
GATE STATUS: VERIFIED -- see inputs.RESULTS_BANNER. Phase A2 full-text
verification completed 2026-08-17; 46 of 48 parameters are closed out against the
14 source full texts and 43 carry a table/page locator. The 2 that remain are the
ciNPWT institutional net acquisition price and the superficial tunnel length.
=============================================================================

TWO CLASSES OF CHECK, AND THE DIFFERENCE IS NOT COSMETIC
--------------------------------------------------------------------------------
HARD / STRUCTURAL (plan Decision 8, "hard / structural (fatal)"). These test that the
engine is a correct Markov process. They `assert`, they are `fatal=True`, and any
failure exits NON-ZERO. They cannot be traded off against anything.

CLINICAL / FACE VALIDITY (plan Decision 8, "clinical / face validity -- reported, not
fatal"). These compare model output with external literature. They are EMITTED AS DATA
with a PASS / FAIL / OUTSIDE / PLUMBING result and NEVER asserted. A FAIL or an OUTSIDE
here is a FINDING, not a build failure, so the run still exits ZERO -- but the report
makes it unmissable. BUILD-CONTRACT section 10 was corrected on 2026-08-09 precisely
because it had hardened one of these (the Mueller band) into an assertion that is
unsatisfiable alongside the contract's base threshold.

THE HEADLINE ROW
--------------------------------------------------------------------------------
`S1_nnt_intermediate` / `S2_nnt_zwanenburg` are the paper's thesis in miniature. With
the relative effect FROZEN at RR 0.61, the model returns NNT 39.0 at a 6.6% baseline
90-day SSI risk and 19.1 at 13.5%. Zwanenburg's published NNT of 19 [11] is therefore
reproduced ONLY at a control rate of ~13.5% -- which is exactly the paper's claim that
baseline risk, not the relative effect, decides who benefits. That row goes in the
manuscript.

WHAT IS *NOT* INDEPENDENT CORROBORATION -- SAY SO OUT LOUD
--------------------------------------------------------------------------------
Two of the face-validity targets are ALGEBRAIC IDENTITIES under plan
Decision 3's counter structure and CANNOT fail:

    n_reop       = n_ssi      * p_reop_ssi          -> n_reop/n_ssi      == 0.286 always
    n_persistent = n_ssi_deep * p_persistent_deep   -> ratio             == 0.25  always

`model.run_arm` computes each counter as exactly that product and neither counter
touches a cost. Reporting "the model reproduces Karamian's 28.6% reoperation rate" as if
it corroborated anything would be circular. They are labelled PLUMBING: they verify that
the counter wiring is intact and the parameter is not silently dropped -- nothing more.
A reviewer who spots the circularity unaided will distrust the whole suite, so it is
stated here rather than discovered there.

PRICE YEARS -- THE ERROR SECTION 10 WAS CORRECTED FOR, TWICE
--------------------------------------------------------------------------------
Every dollar comparison in this file is computed on ONE consistent price basis. The
2019-USD weighted SSI cost ($78,024) must never be divided into a 2025-USD device price
($250) or multiplied by a 2025-USD WTP ($100,000). `S8_price_year_invariance`
demonstrates the invariance directly: deflating EVERYTHING to 2019 USD (device $214.79,
WTP $85,918, both computed live from the CPI CSV) reproduces the 2025-USD hand
calculation to machine precision. No CPI index value is hard-coded anywhere -- the
weighted-cost target is `78,024 x CPI[2025]/CPI[2019]` read from
`analysis/data/cpi_medical.csv`, so the check follows the index if BLS ever revises it.

INDEPENDENCE FROM THE OTHER RUN SCRIPTS
--------------------------------------------------------------------------------
This module imports `inputs.py` and `model.py` ONLY. It does NOT import
`run_threshold.py` (written concurrently) and reads no other run script's output: the
break-even solve is done here, directly against the engine, with `scipy.optimize.brentq`.
Validating the engine against another module's CSV would validate the CSV, not the model.

TWO BASES, BOTH LABELLED (BUILD-CONTRACT section 10, "BASIS LABELS -- added 2026-08-09")
--------------------------------------------------------------------------------
Every break-even risk this module publishes is emitted on BOTH reporting conventions and
each one is named, because an unlabelled p* is ambiguous:

    undiscounted, no half-cycle          p* = 0.6515% at $100k -- comparison with the
                                         closed-form hand calculation ONLY (`S7`)
    reporting base case (24 cycles,      p* = 0.660%  at $100k -- THE MANUSCRIPT
    3%/yr, half-cycle)                   HEADLINE (Table 6, Results)

`S9`-`S12` carry the undiscounted pair in `observed` and the reporting-basis value in the
parallel column `observed_reporting_basis`; `validation_report.md` section 3.2 renders
both and marks which column the manuscript must use. No band verdict differs between the
bases -- this is a labelling requirement, not a numerical one -- but quoting one number
from each convention in one sentence would put two different values of one quantity into
one manuscript, which is exactly what the contract forbids.

OUTPUTS (BUILD-CONTRACT section 8, exact filenames)
--------------------------------------------------------------------------------
    v2/results/validation_report.csv   banner line 1, header line 2 ->
                                       pd.read_csv(path, skiprows=1)
                                       columns: check_id, category, description, target,
                                       observed, observed_reporting_basis, tolerance,
                                       result, fatal, note
    v2/results/validation_report.md    results banner, HARD table, SANITY table,
                                       narrative

EXIT CODE
--------------------------------------------------------------------------------
    non-zero  iff a HARD check fails
    zero      even when a SANITY check is FAIL or OUTSIDE (those are findings)

Run with cwd = v2/analysis/ :  python3 validation.py
"""

from __future__ import annotations

import os
import sys
import traceback

# Runnable from anywhere: this script's own dir (= v2/analysis) goes on the path so
# `import inputs` / `import model` resolve (model.py does `import inputs`).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd
from scipy.optimize import brentq

import inputs as I
import model

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V1_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
RESULTS_DIR = os.path.join(V1_DIR, "results")

S = I.SETTINGS
ARMS = I.ARMS
STRATA = S["strata"]
RISK_GRID = S["risk_grid"]
WTP = float(S["wtp_primary"])
BASE_HORIZON = int(S["horizon_cycles"])
RR_BASE = float(I.tp.rr_cinpwt)

#: **A2 RECLASSIFICATION, 2026-08-17.** This build treated "1.0-15.0%" as Mueller 2026's
#: published break-even BAND and reported its own threshold as falling OUTSIDE it, below
#: a "1.0% floor". Full-text verification found that 1.0% and 15.0% are simply the second
#: and eighth of NINE SWEPT INPUT RATES in Mueller's Methods -- verbatim: "results are
#: displayed in tables at nine pre-specified rates: 0.5%, 1.0%, 2.0%, 3.0%, 4.0%, 5.0%,
#: 10.0%, 15.0%, and 20.0%". They are column headers, not estimates, and 1.0% is not a
#: floor of anything: at 1.0% Mueller's own Table 1 shows universal ciNPT LOSING money.
#: Comparing a break-even risk against a swept input range was a category error.
#:
#: The range is retained ONLY as a coverage check -- "does our threshold land inside the
#: risk range Mueller thought worth examining?" -- and is renamed accordingly. The real
#: comparator is MUELLER_BREAKEVEN below.
MUELLER_BAND = (0.010, 0.150)          # Mueller 2026 [9] SWEPT INPUT RATES, not a band
MUELLER_REF = "Mueller 2026 [9]"
MUELLER_BAND_LABEL = "the 1.0-15.0% span of SSI rates Mueller 2026 [9] swept"

#: Mueller 2026's ACTUAL break-even SSI rates, recomputed during A2 from its own
#: cost-offset identity at its own published inputs (device $495 / $830 at 2023 list,
#: 70% effectiveness, SSI cost $18,899) and cross-checked against all 36 published
#: cutoff-0 cells of its Tables 1-2. Its Conclusions state ">=7% SSI and/or SSC-AR risk".
#: These are ANOTHER PAPER'S OUTPUT -- literal, with an inline citation, the same status
#: as Table 6's comparator column (BUILD-CONTRACT section 9's single exception).
MUELLER_BREAKEVEN = {"$495 device": 0.0374, "$830 device": 0.0627}

#: BUILD-CONTRACT section 10 CROSS-CHECK TARGETS. These are deliberately literal:
#: they are the contract's own numbers, and deriving them from `inputs.py` would make
#: every assertion compare the parameter store with itself and delete the only check that
#: ties the store to the contract. (Same status as model.py's smoke-test targets -- see
#: run_log.d/model.md, post-review fix 3.) The 2025-USD target is NOT among them: it is
#: computed live as 78,024 x CPI[2025]/CPI[2019].
# A2 UPDATE 2026-08-17. Was 78_024.0. The A2 audit moved the SSI split from the rounded
# rates (4.9/6.6, 1.7/6.6) to the counts Edmiston's own prose uses (10,296 and 3,517 over
# 13,813 -- "of these, 75% were deep-incisional SSIs"), i.e. frac_deep 0.742 -> 0.745:
#     0.745 x $93,741 + 0.255 x $32,821 = $69,837.0 + $8,369.4 = $78,206.4
# This is the ALL-CAUSE Edmiston cost per SSI and is what w2019 rebuilds from the Params.
TARGET_SSI_COST_2019 = 78_206.0        # BUILD-CONTRACT section 10 anchor, 2019 USD
TOL_SSI_COST_2019 = 50.0
TOL_SSI_COST_2025 = 100.0
TARGET_REOP = 0.286                    # Karamian 2022 [20]
TARGET_PERSISTENT = 0.25               # Freire-Archer 2023 [17]
TOL_COUNTER = 0.05
TOL_NNT = 0.15                         # BUILD-CONTRACT section 10
# A2 FIX 2026-08-17. The second target was p = 0.135, described as Zwanenburg's "implied"
# pooled control rate -- a value BACK-SOLVED from its printed NNT of 19 at RR 0.61
# (1/(19 x 0.39) = 0.13495). Zwanenburg REPORTS its control rate directly, four times
# (text, Table 2, Table 5, Fig. 2): 14.3% = 315/2205. The string "13.5" appears nowhere in
# the paper. Cross-checking the model against a number its own source contradicts is
# circular, and the sentence this row generated reached the manuscript.
NNT_TARGETS = ((0.066, "Edmiston [12] intermediate stratum"),
               (0.143, "Zwanenburg [11] reported pooled control rate 315/2205"))

RESULT_VOCAB = ("PASS", "FAIL", "OUTSIDE", "PLUMBING")

#: THE TWO BASES, NAMED ONCE (BUILD-CONTRACT section 10, "BASIS LABELS -- added
#: 2026-08-09"). Both are correct; they answer different questions and must never be
#: conflated or averaged. Every threshold this module publishes carries one of these two
#: strings, because an unlabelled p* is ambiguous now that two conventions are in play.
#:
#:   undiscounted, no half-cycle  -> comparison with the closed-form hand calc ONLY
#:   reporting base case          -> THE MANUSCRIPT HEADLINE (Table 6, Results)
#:
#: The discounted value is HIGHER by a factor of ~1.0126, and the direction is structural
#: rather than noise: the device toll falls at cycle 0 where the discount factor is
#: exactly 1, while every benefit is discounted. Read from SETTINGS, never re-declared.
BASIS_UNDISC = f"undiscounted, no half-cycle, {BASE_HORIZON} cycles"
BASIS_REPORT = (f"reporting base case: {BASE_HORIZON} cycles, "
                f"{float(S['disc_cost']):.0%}/yr discounting, half-cycle correction")
BASIS_UNDISC_SHORT = "undiscounted, no half-cycle"
BASIS_REPORT_SHORT = (f"reporting base case ({BASE_HORIZON} cycles, "
                      f"{float(S['disc_cost']):.0%}/yr, half-cycle)")


# --------------------------------------------------------------------------- #
# 0. Report collector                                                          #
# --------------------------------------------------------------------------- #
class Report:
    """Accumulates one row per check and renders the two BUILD-CONTRACT section-8 files.

    A row is appended BEFORE a HARD check asserts, so a fatal failure still lands in the
    report instead of vanishing with the traceback.
    """

    COLUMNS = ["check_id", "category", "description", "target", "observed",
               "observed_reporting_basis", "tolerance", "result", "fatal", "note"]

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, check_id, category, description, target, observed, tolerance,
            result, fatal, note, observed_reporting_basis="") -> dict:
        """`observed_reporting_basis` is the PARALLEL REPORTING-BASIS reading of the same
        quantity (24 cycles, 3%/yr, half-cycle), for the rows whose `observed` is on the
        undiscounted hand-calc basis. It is empty for every check to which the
        distinction does not apply. BUILD-CONTRACT section 10 requires the reporting
        basis to be the manuscript headline, so it is carried explicitly rather than left
        for a downstream module to re-derive."""
        if category not in ("HARD", "SANITY"):
            raise ValueError(f"category must be HARD or SANITY, got {category!r}")
        if result not in RESULT_VOCAB:
            raise ValueError(f"result must be one of {RESULT_VOCAB}, got {result!r}")
        row = {
            "check_id": str(check_id),
            "category": str(category),
            "description": str(description),
            "target": str(target),
            "observed": str(observed),
            "observed_reporting_basis": str(observed_reporting_basis),
            "tolerance": str(tolerance),
            "result": str(result),
            "fatal": bool(fatal),
            "note": " ".join(str(note).split()),
        }
        self.rows.append(row)
        tag = {"PASS": "PASS", "FAIL": "FAIL", "OUTSIDE": "OUTSIDE",
               "PLUMBING": "PLUMB"}[result]
        kind = "" if category == "HARD" else "   [face validity -- reported, not fatal]"
        print(f"  [{tag:<7}] {check_id}: {description}{kind}")
        print(f"            target {target}  |  observed {observed}  |  tol {tolerance}")
        return row

    # -- convenience ------------------------------------------------------- #
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=self.COLUMNS)

    def hard(self) -> list[dict]:
        return [r for r in self.rows if r["category"] == "HARD"]

    def sanity(self) -> list[dict]:
        return [r for r in self.rows if r["category"] == "SANITY"]

    def hard_failures(self) -> list[dict]:
        return [r for r in self.hard() if r["result"] == "FAIL"]


def _run(rep: Report, fn) -> None:
    """Run one check function; a HARD assertion failure is caught so that the report is
    still written and the failure is visible, rather than dying with a traceback and no
    `validation_report.csv`. The non-zero exit is then driven off the report itself."""
    before = len(rep.rows)
    try:
        fn(rep)
    except Exception as exc:                       # noqa: BLE001 -- deliberate
        detail = f"{type(exc).__name__}: {exc}"
        print(f"  [FAIL   ] {fn.__name__} raised -- {detail}")
        traceback.print_exc(limit=3)
        if len(rep.rows) == before:                # the check died before recording
            rep.add(fn.__name__, "HARD", f"{fn.__name__} (raised before reporting)",
                    "no exception", detail, "n/a", "FAIL", True,
                    "The check itself raised. Treated as a fatal structural failure.")
        else:
            for r in rep.rows[before:]:
                if r["result"] == "PASS":
                    r["result"] = "FAIL"
                    r["note"] = (r["note"] + " ASSERTION FAILED: " + detail).strip()


# --------------------------------------------------------------------------- #
# 1. Shared engine fixtures                                                    #
# --------------------------------------------------------------------------- #
#: `p` values used for the matrix / trace sweeps. The full 291-point grid (subsampled
#: every 10th point) plus the three reported strata plus the extremes -- including
#: p = 0.5, far outside the study range, because a row-stochasticity bug shows up at the
#: extremes first.
_P_SWEEP = sorted({0.005, 0.15, 0.5, *(float(v) for v in STRATA.values()),
                   *(float(x) for x in RISK_GRID[::10])})
_P_TRACE = (0.005, float(STRATA["intermediate"]), 0.15)
_HORIZONS = tuple(S["horizons"])                   # (3, 12, 24, 60)

_TRACE_CACHE: list[tuple] | None = None


def _runs() -> list[tuple]:
    """[(arm, p, horizon, result_dict), ...] -- the fixture every trace-level HARD check
    shares, built once. Base-case 3/12 tunnel layout throughout (BUILD-CONTRACT section 2:
    only run_dsa / run_scenarios may instantiate another width)."""
    global _TRACE_CACHE
    if _TRACE_CACHE is None:
        _TRACE_CACHE = [
            (arm, p, h, model.run_arm(arm, p_base=p, horizon_cycles=h))
            for arm in ARMS for p in _P_TRACE for h in _HORIZONS
        ]
    return _TRACE_CACHE


def _life_table_survival(n_cycles: int) -> np.ndarray:
    """Pure life-table survivorship over `n_cycles` monthly cycles, from the same cohort
    age / sex mix and the same `inputs.background_mortality` the engine uses.

    S[0] = 1; S[t+1] = S[t] * (1 - q_bg(age0 + t/12, t = 1/12)).

    It carries background mortality ONLY. The cohort additionally faces perioperative
    mortality (0.15% at cycle 0 after the A2 correction, both arms), so cohort survival
    must sit AT OR BELOW this
    curve at every cycle -- if it ever rose above it, the engine would be manufacturing
    survivors.
    """
    age0 = float(I.mortality.age_start)
    sex_mix = float(I.mortality.frac_male)
    cy = float(S["cycle_years"])
    surv = np.ones(int(n_cycles) + 1, dtype=float)
    for t in range(int(n_cycles)):
        surv[t + 1] = surv[t] * (1.0 - I.background_mortality(age0 + t * cy, sex_mix, t=cy))
    return surv


def _undisc_pair(p, **kw):
    """Both arms, 24 cycles, UNDISCOUNTED and NO half-cycle correction -- the basis the
    contract's hand calculation is defined on (`BASIS_UNDISC`, BUILD-CONTRACT
    section 10)."""
    ra = model.run_arm("A_standard", p_base=p, horizon_cycles=BASE_HORIZON,
                       half_cycle=False, disc_cost=0.0, disc_qaly=0.0, **kw)
    rb = model.run_arm("B_cinpwt", p_base=p, horizon_cycles=BASE_HORIZON,
                       half_cycle=False, disc_cost=0.0, disc_qaly=0.0, **kw)
    return ra, rb


def _report_pair(p, **kw):
    """Both arms on the REPORTING BASE CASE (`BASIS_REPORT`): 24 cycles, 3%/yr on costs
    and QALYs, half-cycle correction on. This is the basis the manuscript headline is
    quoted on (BUILD-CONTRACT section 10, "BASIS LABELS"), and it is the parallel of
    `_undisc_pair` -- same call, same horizon, only the convention differs. Every setting
    is READ from `inputs.SETTINGS`, never re-declared."""
    ra = model.run_arm("A_standard", p_base=p, horizon_cycles=BASE_HORIZON,
                       half_cycle=bool(S["half_cycle"]),
                       disc_cost=float(S["disc_cost"]), disc_qaly=float(S["disc_qaly"]),
                       **kw)
    rb = model.run_arm("B_cinpwt", p_base=p, horizon_cycles=BASE_HORIZON,
                       half_cycle=bool(S["half_cycle"]),
                       disc_cost=float(S["disc_cost"]), disc_qaly=float(S["disc_qaly"]),
                       **kw)
    return ra, rb


def _inmb(p, *, basis: str = BASIS_UNDISC, **kw) -> float:
    ra, rb = (_report_pair(p, **kw) if basis == BASIS_REPORT else _undisc_pair(p, **kw))
    return model.incremental(rb, ra, wtp=WTP)["inmb"]


def engine_threshold(*, basis: str = BASIS_UNDISC, **kw) -> float:
    """Break-even baseline 90-day SSI risk p* where INMB = 0, solved DIRECTLY against the
    engine with `scipy.optimize.brentq`.

    `basis` selects one of the two conventions named at the top of this module and MUST
    be one of them -- there is no third, and an unlabelled p* is not a publishable
    quantity (BUILD-CONTRACT section 10). The default is `BASIS_UNDISC` because the only
    caller that may omit it is the hand-calculation cross-check `S7`, which is defined on
    that basis; every other caller passes the basis explicitly.

    Deliberately NOT imported from `run_threshold.py`: that module is being written
    concurrently, and a cross-import would make this suite validate another unfinished
    module's arithmetic instead of the engine's. INMB is smooth and monotone increasing in
    `p` over the whole range, so a wide bracket converges without tricks.
    """
    if basis not in (BASIS_UNDISC, BASIS_REPORT):
        raise ValueError(f"basis must be BASIS_UNDISC or BASIS_REPORT, got {basis!r}")
    return float(brentq(lambda p: _inmb(p, basis=basis, **kw),
                        1e-9, 0.9, xtol=1e-14, rtol=1e-14))


def qaly_loss_per_ssi(sample=None) -> float:
    """QALY loss per incident SSI, undiscounted, on the hand-calculation basis:
    `frac_deep * du_deep * (deep tunnel / 12) + frac_sup * du_sup * (sup tunnel / 12)`.
    Decrements are stored as POSITIVE magnitudes (run_log.d/inputs.md trap 3)."""
    return (model.val(I.tp.frac_deep, sample) * model.val(I.util.du_deep, sample)
            * (I.DEEP_LEN / 12.0)
            + model.val(I.tp.frac_sup, sample) * model.val(I.util.du_sup, sample)
            * (I.SUP_LEN / 12.0))


def hand_calc_threshold(device_cost: float, ssi_cost: float, wtp: float,
                        rr: float, dq: float) -> float:
    """`p* = C / [(1 - RR) * (SSI cost + WTP * QALY loss)]` -- BUILD-CONTRACT section 10,
    undiscounted, no half-cycle correction.

    EVERY ARGUMENT MUST BE ON THE SAME PRICE BASIS. Mixing a 2019-USD `ssi_cost` with a
    2025-USD `device_cost` / `wtp` is the error section 10 was corrected for twice; the
    function cannot detect it, so `S8_price_year_invariance` demonstrates it explicitly.
    """
    return float(device_cost) / ((1.0 - float(rr)) * (float(ssi_cost) + float(wtp) * float(dq)))


def weighted_ssi_cost_2019(*, attributable: bool = False) -> float:
    """Weighted 24-month cost per SSI in the tranches' OWN stored price year (2019 USD).

    Built from the six `inputs.cost.*` marginal tranche Params, never from the literals
    93,741 / 32,821 -- otherwise the check compares two literals and keeps passing after
    a Phase-A2 revision of Edmiston.

    A2 2026-08-17. This used to multiply unconditionally by `m_ssi_cost`, which was
    harmless while `m_ssi_cost` was 1.000 and silently wrong the moment the audit
    re-anchored the base case on Pennington's matched increment (0.492). The two
    quantities are now separated explicitly:

        attributable=False  ->  Edmiston's ALL-CAUSE increment per SSI  ($78,206)
        attributable=True   ->  what the model actually charges         ($38,477)

    S3 checks the ALL-CAUSE value against the contract anchor, because that anchor is an
    Edmiston number. The attributable value is reported alongside it so no reader can
    mistake one for the other.
    """
    deep = sum(float(p) for p in (I.cost.c_ssi_deep_0_6mo, I.cost.c_ssi_deep_6_12mo,
                                  I.cost.c_ssi_deep_12_24mo))
    sup = sum(float(p) for p in (I.cost.c_ssi_sup_0_6mo, I.cost.c_ssi_sup_6_12mo,
                                 I.cost.c_ssi_sup_12_24mo))
    m = float(I.cost.m_ssi_cost) if attributable else 1.0
    f_deep, f_sup = float(I.tp.frac_deep), float(I.tp.frac_sup)
    tot = f_deep + f_sup
    f_deep, f_sup = f_deep / tot, f_sup / tot
    return m * (f_deep * deep + f_sup * sup)


# =========================================================================== #
#                    HARD / STRUCTURAL CHECKS  (fatal, assert)                #
# =========================================================================== #
def h1_matrix_rowsums(rep: Report) -> None:
    """Every transition-matrix row sums to 1 (tol 1e-9), both arms, cycles 0-29,
    across a wide range of `p`."""
    TOL = 1e-9
    worst, n_mat = 0.0, 0
    for arm in ARMS:
        for p in _P_SWEEP:
            for t in range(0, 30):
                P = model.transition_matrix(arm, t, p_base=p, rr=None)
                worst = max(worst, float(np.max(np.abs(P.sum(axis=1) - 1.0))))
                n_mat += 1
    ok = worst <= TOL
    rep.add("H1_matrix_rowsum", "HARD",
            "Every transition-matrix row sums to 1",
            "1.0 exactly", f"max |rowsum - 1| = {worst:.2e}", f"{TOL:.0e}",
            "PASS" if ok else "FAIL", True,
            f"{n_mat} matrices: both arms x {len(_P_SWEEP)} values of p "
            f"(0.005-0.5, the 291-point grid subsampled every 10th point plus the three "
            f"reported strata and p=0.5 well outside the study range) x cycles 0-29. "
            f"Row stochasticity is what makes the object a Markov chain at all.")
    assert ok, f"transition-matrix rows deviate from 1 by {worst:.3e} (tol {TOL:.0e})"


def h2_trace_rowsums(rep: Report) -> None:
    """Every trace row sums to 1 (tol 1e-8) -- cohort mass is conserved."""
    TOL = 1e-8
    worst = 0.0
    for arm, p, h, r in _runs():
        worst = max(worst, float(np.max(np.abs(r["trace"].sum(axis=1) - 1.0))))
    ok = worst <= TOL
    rep.add("H2_trace_rowsum", "HARD",
            "Every trace row sums to 1 (cohort mass conserved)",
            "1.0 exactly", f"max |rowsum - 1| = {worst:.2e}", f"{TOL:.0e}",
            "PASS" if ok else "FAIL", True,
            f"{len(_runs())} runs: both arms x p in {_P_TRACE} x horizons "
            f"{_HORIZONS}. Mass is neither created nor destroyed at any cycle.")
    assert ok, f"trace rows deviate from 1 by {worst:.3e} (tol {TOL:.0e})"


def h3_occupancy_nonnegative(rep: Report) -> None:
    """Occupancy is non-negative in every slot at every cycle."""
    TOL = -1e-12
    worst = 0.0
    for arm, p, h, r in _runs():
        worst = min(worst, float(np.min(r["trace"])))
    ok = worst >= TOL
    rep.add("H3_occupancy_nonneg", "HARD",
            "Occupancy is non-negative in every slot at every cycle",
            ">= 0", f"min occupancy = {worst:.2e}", f"{TOL:.0e}",
            "PASS" if ok else "FAIL", True,
            "A negative occupancy would mean a probability leaked below zero and would "
            "silently corrupt every cost and QALY accumulated from that slot.")
    assert ok, f"negative occupancy {worst:.3e} in the trace"


def h4_survival_le_life_table(rep: Report) -> None:
    """Cohort survival <= pure life-table survival at every cycle, both arms.

    The cohort carries background mortality PLUS 0.15% perioperative mortality at cycle 0
    (0.2% before the A2 correction),
    so it must sit at or below the life-table curve everywhere. This is the check that
    would catch an engine quietly failing to apply mortality inside the tunnels.
    """
    TOL = 1e-12
    worst = -np.inf
    detail = ""
    for arm, p, h, r in _runs():
        alive = 1.0 - r["trace"][:, I.DEAD]
        slt = _life_table_survival(h)
        gap = float(np.max(alive - slt))
        if gap > worst:
            worst, detail = gap, f"{arm}, p={p:.3f}, horizon {h}"
    where = f" (at {detail})" if worst > 0.0 else " (the cohort never exceeds it anywhere)"
    ok = worst <= TOL
    ra = model.run_arm("A_standard", p_base=float(STRATA["intermediate"]),
                       horizon_cycles=BASE_HORIZON)
    alive24 = float(1.0 - ra["trace"][-1, I.DEAD])
    slt24 = float(_life_table_survival(BASE_HORIZON)[-1])
    rep.add("H4_survival_le_lifetable", "HARD",
            "Cohort survival <= life-table survival at every cycle",
            "cohort alive(t) <= S_lifetable(t)",
            f"max (cohort - life table) = {worst:.2e}{where}",
            f"{TOL:.0e}", "PASS" if ok else "FAIL", True,
            f"Life-table curve built from the same inputs.background_mortality, cohort "
            f"age {float(I.mortality.age_start):.0f}, {float(I.mortality.frac_male)*100:.0f}% "
            f"male, 1 - (1 - qx) ** (1/12) per cycle. At 24 cycles the cohort is "
            f"{alive24:.6f} alive against a life-table {slt24:.6f}; the "
            f"{(slt24 - alive24)*1000:.2f} per 1,000 shortfall is the "
            f"{float(I.mortality.p_periop_death)*100:.2f}% perioperative "
            f"mortality (Pumberger [24]) net of its own survivorship. Identical in both "
            f"arms -- no SSI-attributable mortality difference is assumed.")
    assert ok, f"cohort survival exceeds life-table survival by {worst:.3e}"


def h5_dead_monotone(rep: Report) -> None:
    """The DEAD column is non-decreasing -- death is absorbing."""
    TOL = -1e-15
    worst = 0.0
    for arm, p, h, r in _runs():
        worst = min(worst, float(np.min(np.diff(r["trace"][:, I.DEAD]))))
    ok = worst >= TOL
    rep.add("H5_dead_monotone", "HARD",
            "DEAD occupancy is non-decreasing (death is absorbing)",
            "diff(trace[:, DEAD]) >= 0", f"min diff = {worst:.2e}", f"{TOL:.0e}",
            "PASS" if ok else "FAIL", True,
            "P[DEAD, DEAD] = 1 with no outgoing mass; a resurrection would inflate life "
            "years and every dwell-based QALY.")
    assert ok, f"trace[:, DEAD] decreases by {worst:.3e}"


def h6_tunnels_drain(rep: Report) -> None:
    """Both SSI tunnels drain -- nobody is stuck in a tunnel at the horizon.

    Onset sits at cycles 1..3, the deep tunnel is 12 cycles wide, so the last entrant
    leaves DEEP_11 into RECOV at cycle 3 + 12 = 15. Tunnel occupancy must be exactly zero
    from cycle 15 onward, and in particular at the 24- and 60-cycle horizons.
    """
    TOL = 1e-12
    tun = slice(I.SUP_START, I.DEEP_START + I.DEEP_LEN)
    expected_empty = model.ONSET_CYCLES + int(I.DEEP_LEN)      # 3 + 12 = 15
    worst_tail, worst_final, first_empty = 0.0, 0.0, -1
    for arm, p, h, r in _runs():
        occ = r["trace"][:, tun].sum(axis=1)
        worst_final = max(worst_final, float(occ[-1]) if h >= expected_empty else 0.0)
        if h >= expected_empty:
            worst_tail = max(worst_tail, float(np.max(occ[expected_empty:])))
            empties = np.nonzero(occ <= TOL)[0]
            # first cycle at which the tunnels are empty AND stay empty
            for c in empties:
                if c > 0 and np.all(occ[c:] <= TOL):
                    first_empty = max(first_empty, int(c))
                    break
    ok = (worst_tail <= TOL) and (worst_final <= TOL)
    rep.add("H6_tunnels_drain", "HARD",
            "Both SSI tunnels drain to RECOV (nobody is stuck in a tunnel)",
            f"tunnel occupancy = 0 from cycle {expected_empty} onward",
            f"max tunnel occupancy from cycle {expected_empty} = {worst_tail:.2e} "
            f"(first permanently empty at cycle {first_empty})",
            f"{TOL:.0e}", "PASS" if ok else "FAIL", True,
            f"Onset is confined to cycles 1-{model.ONSET_CYCLES}; the deep tunnel is "
            f"{I.DEEP_LEN} cycles wide, so the last entrant exits to RECOV at cycle "
            f"{expected_empty}. A tunnel that never empties would keep charging its "
            f"utility decrement forever and would break the 12-24-month cost tranche.")
    assert ok, f"tunnel occupancy {worst_tail:.3e} persists past cycle {expected_empty}"


def h7_onset_hazard_zero(rep: Report) -> None:
    """The SSI onset hazard is exactly zero from cycle 4 onward (the transition out of
    cycle index 3), so the pivotal hazard is never extrapolated."""
    worst_h, worst_p = 0.0, 0.0
    for arm in ARMS:
        for p in (0.005, float(STRATA["intermediate"]), 0.15, 0.5):
            for t in range(model.ONSET_CYCLES, 40):
                r = model._cycle_rates(arm, t, p_base=p, rr=None)
                worst_h = max(worst_h, abs(float(r["h_total"])))
                P = model.transition_matrix(arm, t, p_base=p, rr=None)
                worst_p = max(worst_p, float(P[I.WELL, I.SUP_START]),
                              float(P[I.WELL, I.DEEP_START]))
    ok = (worst_h == 0.0) and (worst_p == 0.0)
    rep.add("H7_onset_hazard_zero", "HARD",
            "SSI onset hazard is zero from cycle 4 onward (90-day window only)",
            "h_total = 0 and P[WELL, SUP_0] = P[WELL, DEEP_0] = 0 for cycle >= 3",
            f"max h_total = {worst_h:.1e}, max WELL->tunnel probability = {worst_p:.1e}",
            "0 exactly (bitwise)", "PASS" if ok else "FAIL", True,
            "Onset acts on the transitions out of cycles 0-2, so entrants occupy "
            "SUP_0/DEEP_0 at cycles 1-3 and the hazard is bitwise zero thereafter. This "
            "is why the 2-year horizon needs NO extrapolation of the pivotal hazard -- a "
            "real strength of the design, and it must be true bitwise, not to a "
            "tolerance. Checked for cycles 3-39, both arms, p in "
            "{0.005, 0.066, 0.15, 0.5}.")
    assert ok, f"onset hazard is not zero after the 90-day window ({worst_h:.3e})"


def h8_cumulative_onset_risk(rep: Report) -> None:
    """Cumulative 90-day onset RISK = `p` (arm A) and `p x RR` (arm B) across the FULL
    291-point risk grid, tol 1e-12.

    THE well-posed form of "does the model reproduce the input risk". The cohort COUNT
    `n_ssi` does NOT equal `p` and must never be asserted to (BUILD-CONTRACT section 10):
    competing perioperative and background mortality attenuate it by a constant 0.322%
    relative. The RISK -- what the conditional-onset formula
    `h_k = p*w_k / (1 - p*sum(w_<k))` is FOR -- is exact.
    """
    TOL = 1e-12
    worst, where = 0.0, ""
    for p in RISK_GRID:
        for arm in ARMS:
            surv_no_ssi = 1.0
            for t in range(model.ONSET_CYCLES):
                surv_no_ssi *= (1.0 - model._cycle_rates(arm, t, p_base=float(p),
                                                         rr=None)["h_total"])
            cum = 1.0 - surv_no_ssi
            target = float(p) * (1.0 if arm == "A_standard" else RR_BASE)
            dev = abs(cum - target)
            if dev > worst:
                worst, where = dev, f"{arm}, p={float(p):.4f}"
    ok = worst <= TOL
    # the documented count attenuation, reported alongside so the distinction is explicit
    r_int = model.run_arm("A_standard", p_base=float(STRATA["intermediate"]),
                          horizon_cycles=BASE_HORIZON)
    atten = (float(STRATA["intermediate"]) - r_int["n_ssi"]) / float(STRATA["intermediate"])
    rep.add("H8_onset_risk_equals_p", "HARD",
            "Cumulative 90-day onset RISK = p (arm A) and p x RR (arm B), full risk grid",
            "p and p x 0.61 exactly",
            f"max |cum risk - target| = {worst:.2e} (worst at {where})",
            f"{TOL:.0e}", "PASS" if ok else "FAIL", True,
            f"All {len(RISK_GRID)} grid points (0.005-0.15), both arms. This is the "
            f"correct form of the check: the cohort COUNT n_ssi sits {atten*100:.3f}% "
            f"relative BELOW p because perioperative "
            f"({float(I.mortality.p_periop_death)*100:.2f}%, cycle 0) and background "
            f"mortality compete with onset in exactly the cycles where onset happens. "
            f"Asserting n_ssi == p would be unsatisfiable by any correct competing-risks "
            f"engine (run_log.d/model.md objection 2).")
    assert ok, f"cumulative onset risk deviates from p by {worst:.3e} (tol {TOL:.0e})"


def h9_streams_sum_to_totals(rep: Report) -> None:
    """`cost_stream` / `qaly_stream` sum EXACTLY to `cost` / `qaly`."""
    TOL = 1e-9
    worst_c, worst_q = 0.0, 0.0
    for arm, p, h, r in _runs():
        worst_c = max(worst_c, abs(float(r["cost_stream"].sum()) - float(r["cost"])))
        worst_q = max(worst_q, abs(float(r["qaly_stream"].sum()) - float(r["qaly"])))
    ok = (worst_c <= TOL) and (worst_q <= TOL)
    rep.add("H9_streams_sum_to_totals", "HARD",
            "cost_stream / qaly_stream sum to cost / qaly",
            "sum(stream) = reported total",
            f"max |sum - total| = {worst_c:.2e} (cost), {worst_q:.2e} (QALY)",
            f"{TOL:.0e}", "PASS" if ok else "FAIL", True,
            "Both streams are DISCOUNTED and post-half-cycle. Figures 2 and S3 and the "
            "accrual disclosure are drawn from the streams while the tables quote the "
            "totals; if they disagreed, the figures would contradict the tables.")
    assert ok, f"stream/total mismatch: cost {worst_c:.3e}, qaly {worst_q:.3e}"


HARD_CHECKS = (h1_matrix_rowsums, h2_trace_rowsums, h3_occupancy_nonnegative,
               h4_survival_le_life_table, h5_dead_monotone, h6_tunnels_drain,
               h7_onset_hazard_zero, h8_cumulative_onset_risk, h9_streams_sum_to_totals)


# =========================================================================== #
#             CLINICAL / FACE-VALIDITY CHECKS  (reported, NEVER assert)       #
# =========================================================================== #
def s1_s2_nnt(rep: Report) -> None:
    """THE HEADLINE. NNT reproduction at 6.6% and at Zwanenburg's implied 13.5%.

    NNT = 1 / (n_ssi_A - n_ssi_B) from the engine's own cohort counts, undiscounted and
    without half-cycle correction. The risk-basis target is `1 / (p x (1 - RR))` computed
    live, which is what the plan writes; BUILD-CONTRACT section 10 rounds it to 38.9 / 19.
    """
    for k, (p, label) in enumerate(NNT_TARGETS, start=1):
        ra, rb = _undisc_pair(p)
        nnt = 1.0 / (ra["n_ssi"] - rb["n_ssi"])
        risk_basis = 1.0 / (p * (1.0 - RR_BASE))
        dev = abs(nnt - risk_basis)
        ok = dev <= TOL_NNT
        excess = (nnt / risk_basis - 1.0) * 100.0
        atten = (p - ra["n_ssi"]) / p * 100.0        # count attenuation, in percent
        if k == 1:
            note = (f"HEADLINE ROW -- the paper's thesis in miniature, and the row that "
                    f"goes in the manuscript. Risk-basis target = 1/({p:.3f} x "
                    f"{1.0 - RR_BASE:.2f}) = {risk_basis:.2f}; BUILD-CONTRACT section 10 "
                    f"gives it rounded to 38.9. The engine returns {nnt:.2f}, "
                    f"{excess:.4f}% ABOVE the risk basis. That excess is NOT slop and it "
                    f"is not a tolerance being spent: perioperative mortality "
                    f"({float(I.mortality.p_periop_death)*100:.2f}%, "
                    f"cycle 0) and background mortality compete with onset in exactly the "
                    f"cycles where onset happens, so the cohort COUNT n_ssi sits "
                    f"{atten:.4f}% relative below the onset RISK in BOTH arms, and the "
                    f"NNT -- one over a difference of counts -- rises by exactly "
                    f"a/(1 - a) = {excess:.4f}%. Correct competing-risks behaviour, "
                    f"stated rather than hidden.")
        else:
            note = (f"HEADLINE ROW, part 2. **A2 re-based this row.** It is evaluated at "
                    f"Zwanenburg [11]'s REPORTED pooled control rate of {p*100:.1f}% "
                    f"(315/2205, printed in the text, Table 2, Table 5 and Fig. 2), not "
                    f"at the 13.5% this build previously back-solved from the paper's "
                    f"printed NNT of 19 -- the string '13.5' appears nowhere in that "
                    f"paper. Zwanenburg's own 19 comes from a different estimator, "
                    f"1/(control risk - intervention risk) on crude arm risks rounded up, "
                    f"which reproduces all four of its published NNTs; this model applies "
                    f"the pooled RR multiplicatively and returns "
                    f"{nnt:.2f} at p = {p:.3f} -- NOT at the 6.6% intermediate stratum, "
                    f"where the same device at the same RR 0.61 needs "
                    f"{1.0/(0.066*(1.0-RR_BASE)):.1f} patients. Same {excess:.4f}% "
                    f"competing-mortality excess over the risk basis {risk_basis:.2f} "
                    f"(count attenuation {atten:.4f}%, identical at every p because the "
                    f"onset weights are conditional). Reproducing a published NNT only at "
                    f"the control rate that published NNT was computed from IS the "
                    f"paper's argument: baseline risk, not the relative effect, decides "
                    f"who benefits.")
        rep.add(f"S{k}_nnt_{'intermediate' if k == 1 else 'zwanenburg'}", "SANITY",
                f"Model NNT at p = {p*100:.1f}% ({label})",
                f"{risk_basis:.2f} (= 1 / (p x (1 - RR)); contract section 10 gives "
                f"{'38.9' if k == 1 else '17.9 at the reported 14.3% control rate'})",
                f"{nnt:.2f}", f"{TOL_NNT:.2f} (|dev| = {dev:.3f})",
                "PASS" if ok else "FAIL", False, note)


def s3_s4_weighted_ssi_cost(rep: Report) -> None:
    """Weighted 24-month cost per SSI, in BOTH price years, on ONE consistent basis each.

    NO DOLLAR TARGET IS HARD-CODED FOR 2025. The 2025-USD target is
    `78,024 x CPI[2025]/CPI[2019]` read live from `analysis/data/cpi_medical.csv`, so the
    check follows the index if BLS ever revises it. (BUILD-CONTRACT section 10 originally
    the contract's original "$88.5k-$89k", which is the 2019->2024 factor mislabelled 2025; it was
    corrected to $90,812 on 2026-08-09. Hard-coding the old band would fail on correct
    arithmetic -- phase-a1.md says so explicitly.)
    """
    cpi = I.load_cpi()
    factor = I.inflation_factor(I.EDMISTON_PRICE_YEAR)          # 2019 -> 2025, from CSV
    w2019 = weighted_ssi_cost_2019()
    # A2 2026-08-17 -- TWO BASES, AND S4 NEARLY CHECKED THE WRONG ONE.
    # `ssi_cost_per_case()` applies BOTH the CPI factor AND m_ssi_cost, the attribution
    # multiplier the audit moved from 1.000 (Edmiston's all-cause increment at face value)
    # to 0.492 (Pennington's matched increment). So its "weighted" value is the model's
    # SSI-ATTRIBUTABLE cost, while TARGET_SSI_COST_2019 is an ALL-CAUSE anchor. Comparing
    # them directly would have made S4 fail on a units mismatch rather than on anything
    # wrong. Both bases are now checked explicitly.
    m_attrib = float(I.cost.m_ssi_cost)
    w2025_attrib = model.ssi_cost_per_case()["weighted"]     # attributable, 2025 USD
    w2025 = w2025_attrib / m_attrib                          # all-cause, 2025 USD
    target2025 = TARGET_SSI_COST_2019 * factor

    d19 = abs(w2019 - TARGET_SSI_COST_2019)
    ok19 = d19 <= TOL_SSI_COST_2019
    rep.add("S3_ssi_cost_2019", "SANITY",
            "Weighted 24-month cost per SSI, 2019 USD (Edmiston [12] basis)",
            f"${TARGET_SSI_COST_2019:,.0f}", f"${w2019:,.2f}",
            f"+/- ${TOL_SSI_COST_2019:,.0f} (|dev| = ${d19:,.2f})",
            "PASS" if ok19 else "FAIL", False,
            f"Rebuilt from the six inputs.cost.* MARGINAL tranche Params "
            f"({float(I.tp.frac_deep):.3f} x $93,741 deep + {float(I.tp.frac_sup):.3f} x "
            f"$32,821 superficial), never from the literals -- so a Phase-A2 revision of "
            f"Edmiston moves the observed value instead of being masked. This is a "
            f"GENUINELY INDEPENDENT check: it ties the parameter store to the contract's "
            f"own anchor. BASIS: this is Edmiston's ALL-CAUSE increment. The model "
            f"charges the SSI-ATTRIBUTABLE value, "
            f"${weighted_ssi_cost_2019(attributable=True):,.2f} in 2019 USD, after the "
            f"A2 re-anchoring on Pennington [14]'s matched increment "
            f"(m_ssi_cost = {float(I.cost.m_ssi_cost):.3f}). Do not quote the two "
            f"interchangeably.")

    d25 = abs(w2025 - target2025)
    ok25 = d25 <= TOL_SSI_COST_2025
    rep.add("S4_ssi_cost_2025", "SANITY",
            "Weighted 24-month cost per SSI, 2025 USD (target computed LIVE from the CPI CSV)",
            f"${target2025:,.2f} (= ${TARGET_SSI_COST_2019:,.0f} x {factor:.6f})",
            f"${w2025:,.2f}",
            f"+/- ${TOL_SSI_COST_2025:,.0f} (|dev| = ${d25:,.2f})",
            "PASS" if ok25 else "FAIL", False,
            f"NO 2025 DOLLAR TARGET IS HARD-CODED. factor = CPI[2025]/CPI[2019] = "
            f"{cpi[2025]:.3f}/{cpi[I.EDMISTON_PRICE_YEAR]:.3f} = {factor:.6f}, read from "
            f"analysis/data/cpi_medical.csv (BLS CUUR0000SAM, medical care, annual "
            f"averages), so the check follows the index if it is ever revised. The "
            f"${d25:,.2f} residual is the rounding of the ${TARGET_SSI_COST_2019:,.0f} "
            f"anchor, which is itself rounded from ${w2019:,.2f}. BOTH VALUES HERE ARE "
            f"ALL-CAUSE. The model does NOT charge ${w2025:,.0f} per SSI: it charges "
            f"${w2025_attrib:,.0f}, the SSI-ATTRIBUTABLE cost, because A2 re-anchored the "
            f"base case on Pennington [14]'s matched increment (m_ssi_cost = "
            f"{m_attrib:.3f}) rather than on Edmiston [12]'s regression-adjusted "
            f"all-cause increment, which is an upper bound on attributable cost by "
            f"construction. Quoting ${w2025:,.0f} as the model's cost per SSI would "
            f"overstate it by {1.0/m_attrib:.2f}x. Contract section 10 originally "
            f"reported $88.5k-$89k -- the 2019->2024 factor 1.131273 mislabelled as "
            f"2025 -- and was corrected on 2026-08-09.")


def s5_s6_counter_plumbing(rep: Report) -> None:
    """Reoperation 28.6% and persistent infection 25% -- ALGEBRAIC IDENTITIES, not
    corroboration. Labelled PLUMBING and explained as such.

    Under plan Decision 3 the engine computes
        n_reop       = n_ssi      * p_reop_ssi
        n_persistent = n_ssi_deep * p_persistent_deep
    and neither counter multiplies a cost anywhere. The ratios are therefore the input
    parameters returned to us, exactly, at every horizon and every `p`. They CANNOT fail
    while the wiring is intact, and they corroborate nothing about the model's behaviour.
    Presenting them as external validation would be circular; a reviewer who spots that
    unaided would rightly distrust the rest of the suite.
    """
    r1y = model.run_arm("A_standard", p_base=float(STRATA["intermediate"]),
                        horizon_cycles=12)                       # Karamian's 1-year frame
    reop = float(r1y["n_reop"]) / float(r1y["n_ssi"])
    persist = float(r1y["n_persistent"]) / float(r1y["n_ssi_deep"])

    # horizon- and p-invariance, which is the point: it is an identity, not a result
    ratios = {(p, h): (model.run_arm("A_standard", p_base=p, horizon_cycles=h)["n_reop"]
                       / model.run_arm("A_standard", p_base=p, horizon_cycles=h)["n_ssi"])
              for p in (0.005, 0.066, 0.15) for h in (3, 12, 24, 60)}
    spread = max(ratios.values()) - min(ratios.values())

    d = abs(reop - TARGET_REOP)
    rep.add("S5_reoperation_plumbing", "SANITY",
            "Reoperation among SSI patients at 1 year (Karamian [20]) -- PLUMBING CHECK",
            f"{TARGET_REOP:.3f}", f"{reop:.6f}",
            f"{TOL_COUNTER:.2f} (|dev| = {d:.2e})",
            "PLUMBING" if d <= TOL_COUNTER else "FAIL", False,
            f"NOT INDEPENDENT CORROBORATION -- an algebraic identity under plan Decision "
            f"3. model.run_arm computes n_reop = n_ssi x p_reop_ssi and the counter "
            f"multiplies no cost, so this ratio returns the input parameter "
            f"{TARGET_REOP} exactly, at every horizon and every p (observed spread across "
            f"p in {{0.005, 0.066, 0.15}} x horizons {{3, 12, 24, 60}} = {spread:.1e}). It "
            f"verifies that the counter is wired up and not silently dropped -- nothing "
            f"more. It cannot fail while the wiring is intact and must not be reported as "
            f"external validation.")

    d2 = abs(persist - TARGET_PERSISTENT)
    rep.add("S6_persistent_plumbing", "SANITY",
            "Persistent infection among deep SSI (Freire-Archer [17]) -- PLUMBING CHECK",
            f"{TARGET_PERSISTENT:.3f}", f"{persist:.6f}",
            f"{TOL_COUNTER:.2f} (|dev| = {d2:.2e})",
            "PLUMBING" if d2 <= TOL_COUNTER else "FAIL", False,
            "Same class as S5 and stated with the same bluntness: n_persistent = "
            "n_ssi_deep x p_persistent_deep by construction, so the ratio is the input "
            "parameter returned to us. Wiring check only; not corroboration of anything.")


def s7_s8_hand_calc(rep: Report) -> None:
    """Hand-calculated threshold vs the engine, and the price-year invariance proof.

    `p* = C / [(1 - RR) x (SSI cost + WTP x QALY loss)]`, undiscounted, no half-cycle.
    Computed on ONE consistent price basis, twice: everything in 2025 USD, and everything
    deflated to 2019 USD (device $214.79, WTP $85,918, both derived live from the CPI CSV,
    never hard-coded). The two MUST agree -- that identity is the arithmetic proof that
    the contract's original 0.746% is a mixed-price-year artifact and 0.649% is the real value.
    """
    factor = I.inflation_factor(I.EDMISTON_PRICE_YEAR)
    dq = qaly_loss_per_ssi()
    c2025 = I.cost_2025(I.cost.c_device_cinpwt)
    c2019 = c2025 / factor
    wtp2019 = WTP / factor
    w2025 = model.ssi_cost_per_case()["weighted"]
    # A2 2026-08-17: BOTH sides must be on the ATTRIBUTABLE basis. `w2025` comes through
    # ssi_cost_per_case(), which applies m_ssi_cost; the 2019 side must therefore apply it
    # too, or this check compares an attributable 2025 cost against an all-cause 2019 cost
    # and "fails" on a units mismatch rather than on a price-year error -- which is the
    # exact confusion the check exists to catch. `attributable=True` is not optional here.
    w2019 = weighted_ssi_cost_2019(attributable=True)

    hand2025 = hand_calc_threshold(c2025, w2025, WTP, RR_BASE, dq)
    hand2019 = hand_calc_threshold(c2019, w2019, wtp2019, RR_BASE, dq)
    hand_mixed = hand_calc_threshold(c2025, w2019, WTP, RR_BASE, dq)   # the OLD error
    eng = engine_threshold()

    rel = abs(eng - hand2025) / hand2025
    TOL_REL = 0.01
    ok = rel <= TOL_REL
    rep.add("S7_handcalc_vs_engine", "SANITY",
            "Hand-calculated break-even risk p* reproduces the engine (undiscounted, no half-cycle)",
            f"{hand2025*100:.4f}% (hand-calc, all 2025 USD)",
            f"{eng*100:.4f}% (engine, brentq on INMB = 0)",
            f"{TOL_REL*100:.0f}% relative (observed {rel*100:.2f}%)",
            "PASS" if ok else "FAIL", False,
            f"p* = C / [(1 - RR) x (SSI cost + WTP x QALY loss)] = ${c2025:,.2f} / "
            f"[{1.0 - RR_BASE:.2f} x (${w2025:,.0f} + ${WTP:,.0f} x {dq:.5f})]. The "
            f"engine sits {(eng/hand2025 - 1)*100:.2f}% ABOVE the closed form, in the "
            f"expected direction and of the expected size: the closed form has neither "
            f"the discrete three-cycle onset split nor the competing mortality that "
            f"attenuates incidence by 0.322%. Solved HERE with scipy.optimize.brentq "
            f"directly against model.py -- run_threshold.py is deliberately NOT imported, "
            f"so this validates the engine and not another module's CSV.")

    rel2 = abs(hand2025 - hand2019) / hand2025
    TOL_INV = 1e-9
    ok2 = rel2 <= TOL_INV
    rep.add("S8_price_year_invariance", "SANITY",
            "Hand-calc is price-year invariant: all-2025-USD == all-2019-USD",
            f"{hand2025*100:.4f}% (device ${c2025:,.2f}, SSI ${w2025:,.0f}, WTP ${WTP:,.0f})",
            f"{hand2019*100:.4f}% (device ${c2019:,.4f}, SSI ${w2019:,.0f}, WTP ${wtp2019:,.0f})",
            f"{TOL_INV:.0e} relative (observed {rel2:.1e})",
            "PASS" if ok2 else "FAIL", False,
            f"THE PROOF, and the reason BUILD-CONTRACT section 10 was corrected TWICE. "
            f"Both price bases give {hand2025*100:.4f}% because numerator and denominator "
            f"scale by the same CPI factor {factor:.6f}. Mixing them does not: a 2025-USD "
            f"device price over a 2019-USD SSI cost with a 2025-USD WTP gives "
            f"{hand_mixed*100:.4f}%, which is an ARTIFACT of mixing price years and "
            f"corresponds to nothing. (Before A2 this mixed-basis figure was the "
            f"contract's original 0.746%; it moved with the parameters, so it is computed "
            f"live here and no longer carries that label.) The 2019-USD device "
            f"(${c2019:,.4f}) and WTP (${wtp2019:,.2f}) are "
            f"DEFLATED LIVE from the CPI CSV, never hard-coded; the contract quotes them "
            f"truncated as $214.79 and $85,918. Consequence: the base-case threshold "
            f"sits slightly LOWER, which is the direction that favours ciNPWT and is "
            f"therefore reported rather than relied on.")


def s9_s12_mueller_band(rep: Report) -> None:
    """External cross-comparison against the SSI-rate span Mueller 2026 [9] examined.

    REPORTED, NEVER ASSERTED -- and REFRAMED BY THE A2 AUDIT, 2026-08-17.

    What this check used to say: that "1.0-15.0%" was Mueller's published break-even BAND
    and that this model's threshold fell OUTSIDE it, below a "1.0% floor", which was then
    written up as a substantive result arguing for broad use.

    What the full text actually says (Methods, PDF p. 2 / journal p. 182): "results are
    displayed in tables at nine pre-specified rates: 0.5%, 1.0%, 2.0%, 3.0%, 4.0%, 5.0%,
    10.0%, 15.0%, and 20.0%". 1.0% and 15.0% are the SECOND and EIGHTH of nine SWEPT INPUT
    RATES -- column headers, not estimates, and not endpoints of anything. 1.0% is not a
    floor: at 1.0% Mueller's own Table 1 shows universal ciNPT LOSING money. Comparing a
    solved break-even risk against a swept input range was a category error, and the
    "OUTSIDE, below the floor" finding was an artifact of it.

    What this check is now: a COVERAGE test -- does our threshold land inside the span of
    SSI rates Mueller thought worth examining? That is worth knowing and is honest about
    what it is. The substantive comparison is against MUELLER_BREAKEVEN (3.74% at its $495
    device price, 6.27% at $830; its Conclusions say ">=7%"), reported separately.

    A further caveat that must travel with any use of this family: Mueller 2026 is a
    three-equation static cost-offset identity with NO QALYs, no discounting, no time
    horizon, no stated perspective and no uncertainty analysis. It cannot externally
    validate a Markov cost-utility model; it is a loose concordance check on the cost side.

    EVERY THRESHOLD BELOW IS PUBLISHED ON BOTH BASES AND BOTH ARE LABELLED. `observed`
    carries the closed-form-comparable UNDISCOUNTED / no-half-cycle pair (hand calc and
    engine, which is the only basis on which the two are commensurable);
    `observed_reporting_basis` carries the REPORTING BASE CASE engine value, which is the
    one Table 6 and the Results section must quote (BUILD-CONTRACT section 10, "BASIS
    LABELS -- added 2026-08-09"). The two differ by ~1.3% relative and NO VERDICT FLIPS
    between them -- every scenario keeps its INSIDE / OUTSIDE status -- which is exactly
    why this is a labelling requirement and not a numerical one.
    """
    lo, hi = MUELLER_BAND
    factor = I.inflation_factor(I.EDMISTON_PRICE_YEAR)
    dq = qaly_loss_per_ssi()
    c2025 = I.cost_2025(I.cost.c_device_cinpwt)
    w2025 = model.ssi_cost_per_case()["weighted"]
    k = float(I.cost.k_attrib_pennington)          # 0.492 -- now IN the base case
    rr_att = float(I.tp.rr_cinpwt_attenuated)      # 0.85
    # A2: the "premium device" price was `price_grid[-1] - 100`, i.e. defined relative to
    # the top of the price axis. Widening that axis to $900 silently moved it from $500 to
    # $800 and pushed this scenario across a band edge -- exactly the kind of coupling
    # that should never decide a published verdict. It is now pinned to a SOURCED value:
    # Mueller 2026 [9]'s own 2023 list price for the 3M Prevena Peel-and-Place 20 cm
    # system. The $830 Prevena Plus price is also on the axis and readable from Table 3.
    c_premium = 495.0                              # Mueller 2026 [9] list, 2023 USD

    # A2 RESTRUCTURE. `w2025` already carries m_ssi_cost = 0.492, because the MATCHED
    # attribution anchor is now the base case. Multiplying by k again -- as this family
    # did before A2, when the base case was all-cause -- would apply the factor twice and
    # charge 0.242 of Edmiston's increment. The sensitivity now runs the OTHER way:
    # dividing by k restores Edmiston's all-cause increment, which is the upper bound.
    w2025_allcause = w2025 / k
    scenarios = [
        ("S9_mueller_base", "Matched-attribution base case (Pennington [14])",
         c2025, w2025, RR_BASE, {}, None, None),
        ("S10_mueller_attribution", "All-cause SSI cost (Edmiston [12] at face value)",
         c2025, w2025_allcause, RR_BASE, {"m_ssi_cost": 1.0}, None, None),
        ("S11_mueller_attr_attenuated", f"All-cause cost + attenuated RR {rr_att:.2f}",
         c2025, w2025_allcause, rr_att, {"m_ssi_cost": 1.0}, rr_att, None),
        ("S12_mueller_attr_att_premium",
         f"All-cause cost + RR {rr_att:.2f} + list-price device ${c_premium:,.0f}",
         c_premium, w2025_allcause, rr_att, {"m_ssi_cost": 1.0}, rr_att, c_premium),
    ]

    family = []
    for cid, label, C, sc, rr, samp, rr_kw, dc in scenarios:
        hand = hand_calc_threshold(C, sc, WTP, rr, dq)
        kw = {}
        if samp:
            kw["sample"] = dict(samp)
        if rr_kw is not None:
            kw["rr"] = rr_kw
        if dc is not None:
            kw["device_cost"] = dc
        eng = engine_threshold(basis=BASIS_UNDISC, **kw)
        eng_rep = engine_threshold(basis=BASIS_REPORT, **kw)
        inside = (lo <= hand <= hi) and (lo <= eng <= hi)
        below = (hand < lo) or (eng < lo)
        # The reporting basis must not move the verdict -- if it ever did, the two bases
        # would no longer be a labelling question and this suite would be hiding it.
        inside_rep = lo <= eng_rep <= hi
        assert inside_rep == inside, (
            f"{cid}: the band verdict DIFFERS between bases (undiscounted "
            f"{eng*100:.4f}% -> {'INSIDE' if inside else 'OUTSIDE'}, reporting basis "
            f"{eng_rep*100:.4f}% -> {'INSIDE' if inside_rep else 'OUTSIDE'}). That is no "
            f"longer a labelling difference and must be reported as a substantive split.")
        family.append((cid, label, hand, eng, eng_rep, inside, below))

    n_inside = sum(1 for *_, inside, _ in family if inside)
    for cid, label, hand, eng, eng_rep, inside, below in family:
        if inside:
            result, verdict = "PASS", "INSIDE the band"
        elif below:
            result, verdict = "OUTSIDE", "BELOW the examined span"
        else:
            result, verdict = "OUTSIDE", "OUTSIDE (above ceiling)"
        basis_note = (
            f"BASIS: the hand-calc and engine values in `observed` are BOTH "
            f"{BASIS_UNDISC_SHORT} -- the only basis on which a closed form and the "
            f"engine are commensurable. `observed_reporting_basis` = "
            f"{eng_rep*100:.4f}% is the SAME quantity on the {BASIS_REPORT_SHORT}, and "
            f"THAT is the value Table 6 and the Results section must quote "
            f"(BUILD-CONTRACT section 10). The reporting basis is higher by "
            f"{(eng_rep/eng - 1)*100:.2f}% because the device toll falls at cycle 0 "
            f"where the discount factor is exactly 1 while every benefit is discounted; "
            f"the band verdict is identical on both bases ({verdict}).")
        if cid == "S9_mueller_base":
            note = (f"REPORTED, NEVER ASSERTED. Engine {eng*100:.4f}% and the consistent "
                    f"hand-calc {hand*100:.4f}% -- {verdict} relative to "
                    f"{MUELLER_BAND_LABEL}. **READ THE VERDICT NARROWLY.** A2 "
                    f"verification established that 1.0% and 15.0% are two of NINE SWEPT "
                    f"INPUT RATES in Mueller's Methods, not the endpoints of a published "
                    f"break-even band, so 'inside' here means only that our threshold "
                    f"falls within the span of SSI rates Mueller examined -- it is a "
                    f"coverage check, not agreement. The pre-A2 build reported this "
                    f"comparison as falling OUTSIDE, 'below Mueller's 1.0% floor'; that "
                    f"finding was an artifact of treating a swept input range as a "
                    f"result, and 1.0% is not a floor -- at 1.0% Mueller's own Table 1 "
                    f"shows universal ciNPT losing money. The substantive comparison is "
                    f"against Mueller's ACTUAL break-even, recomputed from its own "
                    f"cost-offset identity at its own inputs: "
                    + " and ".join(f"{v*100:.2f}% ({k})"
                                   for k, v in MUELLER_BREAKEVEN.items())
                    + f", with its Conclusions stating >=7%. Substituting only this "
                    f"model's SSI cost into Mueller's own equation returns a sub-1% "
                    f"break-even, so the two agree once put on one basis. {n_inside} of "
                    f"{len(family)} scenarios fall inside the examined span. Mueller has "
                    f"no QALYs, no discounting, no horizon and no uncertainty analysis, "
                    f"so this is a loose cost-side concordance check and NOT external "
                    f"validation of a cost-utility model.")
        else:
            note = (f"Reported scenario, same reported-not-fatal status as S9. "
                    f"Hand-calc {hand*100:.4f}% / engine {eng*100:.4f}%, {verdict}. The "
                    f"engine sits slightly above the closed form throughout (competing "
                    f"mortality plus the discrete onset split), which never changes a "
                    f"verdict here.")
        rep.add(cid, "SANITY",
                f"Break-even risk vs {MUELLER_BAND_LABEL} -- "
                f"{label} ({BASIS_UNDISC_SHORT})",
                f"inside the examined span [{lo*100:.1f}%, {hi*100:.1f}%] ({MUELLER_REF}"
                f" -- swept input rates, NOT a published break-even band)",
                f"{hand*100:.4f}% hand-calc | {eng*100:.4f}% engine -- {verdict}",
                "band membership (not a numeric tolerance)",
                result, False, note + " " + basis_note,
                observed_reporting_basis=(f"{eng_rep*100:.4f}% engine "
                                          f"[{BASIS_REPORT_SHORT}] -- {verdict}"))


SANITY_CHECKS = (s1_s2_nnt, s3_s4_weighted_ssi_cost, s5_s6_counter_plumbing,
                 s7_s8_hand_calc, s9_s12_mueller_band)


# =========================================================================== #
#                              Report rendering                               #
# =========================================================================== #
def write_csv(rep: Report) -> str:
    """`validation_report.csv` -- banner as '#'-prefixed line 1, header line 2, so
    `pd.read_csv(path, skiprows=1)` works (BUILD-CONTRACT section 7)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "validation_report.csv")
    with open(path, "w", newline="") as fh:
        fh.write("# " + I.RESULTS_BANNER + "\n")
        rep.frame().to_csv(fh, index=False)
    return path


def _md_cell(s) -> str:
    return str(s).replace("|", "\\|")


def _md_table(rows: list[dict]) -> list[str]:
    cols = ["check_id", "description", "target", "observed", "tolerance", "result"]
    head = ["check_id", "description", "target", "observed", "tolerance", "result"]
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join(["---"] * len(head)) + "|"]
    for r in rows:
        mark = {"PASS": "**PASS**", "FAIL": "**FAIL**",
                "OUTSIDE": "**OUTSIDE**", "PLUMBING": "*PLUMBING*"}[r["result"]]
        cells = [_md_cell(r[c]) for c in cols[:-1]] + [mark]
        out.append("| " + " | ".join(cells) + " |")
    return out


def write_md(rep: Report) -> str:
    """`validation_report.md` -- results banner, HARD table, SANITY table, narrative."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "validation_report.md")

    hard, sanity = rep.hard(), rep.sanity()
    n_hard_pass = sum(1 for r in hard if r["result"] == "PASS")
    counts: dict[str, int] = {}
    for r in sanity:
        counts[r["result"]] = counts.get(r["result"], 0) + 1
    by_id = {r["check_id"]: r for r in rep.rows}

    L: list[str] = []
    L.append("# Internal validation report -- SSI / ciNPWT risk-threshold Markov CUA")
    L.append("")
    L.append("> **" + I.RESULTS_BANNER + "**")
    L.append("")
    L.append(f"**HARD / structural checks: {n_hard_pass}/{len(hard)} PASS** "
             f"(fatal -- any failure exits non-zero). "
             f"**Clinical / face-validity checks: "
             + ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
             + f" of {len(sanity)}** (reported, not fatal -- these never change the "
               f"exit code).")
    L.append("")
    L.append("Two classes of check, per plan Decision 8. **HARD / structural** checks "
             "test that the engine is a correct Markov process; they `assert` and are "
             "fatal. **Clinical / face-validity** checks compare model output with "
             "external literature; they are emitted as data with a "
             "`PASS` / `FAIL` / `OUTSIDE` / `PLUMBING` result and are *never* asserted, "
             "because a disagreement with the literature is a finding about the world, "
             "not a bug in the code. One of them (`S10`, the ALL-CAUSE upper bound) sits "
             "`OUTSIDE` the span Mueller examined and is "
             "reported as such.")
    L.append("")
    L.append("This module imports `inputs.py` and `model.py` only. It does **not** import "
             "`run_threshold.py` and reads no other run script's output: the break-even "
             "solve is done here with `scipy.optimize.brentq` directly against the "
             "engine, so a PASS is a statement about the model rather than about another "
             "module's CSV.")
    L.append("")

    # ---- HARD ---------------------------------------------------------------
    L.append("## 1. HARD / structural checks (fatal)")
    L.append("")
    L.extend(_md_table(hard))
    L.append("")
    L.append("**Notes**")
    L.append("")
    for r in hard:
        L.append(f"- **{r['check_id']}** -- {_md_cell(r['note'])}")
    L.append("")

    # ---- SANITY -------------------------------------------------------------
    L.append("## 2. Clinical / face-validity checks (reported, NOT fatal)")
    L.append("")
    L.extend(_md_table(sanity))
    L.append("")
    L.append("**Notes**")
    L.append("")
    for r in sanity:
        L.append(f"- **{r['check_id']}** -- {_md_cell(r['note'])}")
    L.append("")

    # ---- Narrative ----------------------------------------------------------
    L.append("## 3. Narrative")
    L.append("")

    n1, n2 = by_id["S1_nnt_intermediate"], by_id["S2_nnt_zwanenburg"]
    nnt1 = float(n1["observed"])
    nnt2 = float(n2["observed"])
    _p_int = float(STRATA["intermediate"])
    atten_pct = ((_p_int - model.run_arm("A_standard", p_base=_p_int,
                                         horizon_cycles=BASE_HORIZON)["n_ssi"])
                 / _p_int * 100.0)
    L.append("### 3.1 The headline row -- NNT reproduction")
    L.append("")
    L.append(
        f"With the relative effect held **fixed** at RR {RR_BASE:.2f}, the model returns "
        f"an NNT of **{nnt1:.1f}** at a baseline 90-day SSI risk of 6.6% "
        f"(risk basis `1/(0.066 x 0.39)` = 38.9) and **{nnt2:.1f}** at "
        f"{NNT_TARGETS[1][0]*100:.1f}%, Zwanenburg's REPORTED pooled control rate "
        f"(315/2205; risk basis {1.0/(NNT_TARGETS[1][0]*(1.0-RR_BASE)):.1f}). "
        f"**A2 CORRECTED THIS ROW.** It used to be evaluated at 13.5%, a control rate "
        f"BACK-SOLVED from Zwanenburg's printed NNT of 19; the string \"13.5\" appears "
        f"nowhere in that paper. Zwanenburg's own printed NNT of 19 comes from a "
        f"different estimator -- `1/(control risk - intervention risk)` on crude arm "
        f"risks, rounded up, which reproduces all four of its published NNTs -- whereas "
        f"this model applies the pooled RR multiplicatively and returns "
        f"{nnt2:.1f}. That gap is crude-vs-pooled RR and is disclosed rather than "
        f"engineered away by keeping a control rate the source contradicts. The "
        f"comparison across strata is the whole paper in one row. Nothing about the device changed between those two "
        f"numbers: same RR, same price, same horizon, same cost tranches. Only the "
        f"baseline risk moved, and the number of patients who must be treated to prevent "
        f"one SSI roughly **doubled** ({nnt1/nnt2:.2f}x). A trial-level NNT is not a "
        f"property of an intervention; it is a property of the population the trial "
        f"enrolled. Quoting NNT 19 to a surgeon whose patients carry a 6.6% risk "
        f"overstates the benefit by a factor of two, which is exactly the reasoning error "
        f"a risk-threshold analysis exists to prevent. The engine values sit "
        f"{(nnt1/(1.0/(0.066*(1.0-RR_BASE))) - 1)*100:.4f}% above the pure risk-basis "
        f"targets of {1.0/(0.066*(1.0-RR_BASE)):.2f} and "
        f"{1.0/(NNT_TARGETS[1][0]*(1.0-RR_BASE)):.2f}. That excess is not a tolerance "
        f"being spent: perioperative mortality "
        f"({float(I.mortality.p_periop_death)*100:.2f}% at cycle 0) and background "
        f"mortality compete with "
        f"SSI onset in exactly the cycles where onset happens, so the cohort *count* of "
        f"SSIs sits {atten_pct:.4f}% relative below the onset *risk* in both arms and the "
        f"NNT rises by exactly a/(1 - a). It is correct competing-risks behaviour and is "
        f"stated here rather than absorbed into a loose tolerance."
    )
    L.append("")

    L.append("### 3.2 External comparison with Mueller 2026 [9] -- reported, and a "
             "COVERAGE check, not validation")
    L.append("")
    fam = [by_id[c] for c in ("S9_mueller_base", "S10_mueller_attribution",
                              "S11_mueller_attr_attenuated", "S12_mueller_attr_att_premium")]
    n_in = sum(1 for r in fam if r["result"] == "PASS")
    # A2: hoisted above section 3.2, which now quotes the live base-case and
    # all-cause values in its narrative rather than asserting a fixed direction.
    def _pct_of(row: dict, key: str) -> float:
        """The leading `NN.NNNN%` of a formatted cell, back as a proportion."""
        cell = row[key]
        if key == "observed":                        # "0.6492% hand-calc | 0.6515% ..."
            cell = cell.split(" hand-calc | ", 1)[1]
        return float(cell.split("%", 1)[0]) / 100.0

    _base_rep = _pct_of(by_id["S9_mueller_base"], "observed_reporting_basis")
    _attr_rep = _pct_of(by_id["S10_mueller_attribution"], "observed_reporting_basis")
    _base_und = _pct_of(by_id["S9_mueller_base"], "observed")
    L.append(
        f"The derived break-even risk -- reported on the **{BASIS_REPORT_SHORT}**, and "
        f"cross-checked on the **{BASIS_UNDISC_SHORT}** basis the closed form is defined "
        f"on -- is compared against {MUELLER_BAND_LABEL}. **A2 CORRECTED BOTH HALVES OF "
        f"WHAT THIS SECTION USED TO SAY.** It described 1.0-15.0% as a break-even BAND "
        f"reported by {MUELLER_REF} and reported the base case as falling OUTSIDE it, "
        f"below a 1.0% floor. Neither is true. Those two figures are the second and "
        f"eighth of NINE SWEPT INPUT RATES in Mueller's Methods, so there is no band and "
        f"no floor; and the base case now sits at "
        f"{_base_rep*100:.4f}%, INSIDE the examined span. What falls below the span is "
        f"the ALL-CAUSE row, not the base case -- see the table below and read the "
        f"verdicts narrowly. BUILD-CONTRACT section 10's "
        f"original wording -- \"must fall inside\" -- was corrected on 2026-08-09 because "
        f"it is unsatisfiable alongside the contract's base threshold printed on the "
        f"same line, and because the plan files this comparison under *clinical / face "
        f"validity (reported, not fatal)* (Decision 8). The whole family, "
        f"**on both bases, each one labelled**:"
    )
    L.append("")
    L.append(
        "> **Which column does the manuscript use?** The **last numeric column** -- "
        "`p* (engine, REPORTING BASE CASE)`. BUILD-CONTRACT section 10 (\"BASIS LABELS "
        "-- added 2026-08-09\") makes the reporting base case "
        f"({BASE_HORIZON} cycles, {float(S['disc_cost']):.0%}/yr discounting, half-cycle "
        "correction) **the manuscript headline**, and `run_threshold.py` authoritative "
        "for every published threshold. The two hand-calc / engine columns before it are "
        f"**{BASIS_UNDISC_SHORT}** and exist for **one** purpose: they are the only basis "
        "on which the closed form and the engine are commensurable, so they are the "
        "`S7` cross-check, not a reportable result. **Table 6 and the Results section "
        "must quote the reporting-basis column and name the basis; do not mix the two "
        "and never average them.**")
    L.append("")
    L.append(f"| scenario | p* (hand-calc, 2025 USD, {BASIS_UNDISC_SHORT}) "
             f"| p* (engine, {BASIS_UNDISC_SHORT}) "
             f"| **p* (engine, REPORTING BASE CASE: {BASE_HORIZON} cycles, "
             f"{float(S['disc_cost']):.0%}/yr, half-cycle) -- USE THIS** "
             f"| vs 1.0-15.0% band |")
    L.append("|---|---|---|---|---|")
    _suffix = f" ({BASIS_UNDISC_SHORT})"
    for r in fam:
        obs = r["observed"]
        hand_s, rest = obs.split(" hand-calc | ", 1)
        eng_s, verdict = rest.split(" engine -- ", 1)
        label = r["description"].split(" -- ", 1)[-1]
        if label.endswith(_suffix):                  # the basis lives in the header now
            label = label[: -len(_suffix)]
        rep_s = r["observed_reporting_basis"].split(" engine ", 1)[0]
        L.append(f"| {_md_cell(label)} | {hand_s} | {eng_s} | **{rep_s}** | "
                 f"{'**' + verdict + '**' if r['result'] == 'OUTSIDE' else verdict} |")
    L.append("")
    L.append(
        f"The verdict column is identical on both bases -- **no scenario changes its "
        f"INSIDE / OUTSIDE status** -- so the choice of basis here is a labelling "
        f"obligation rather than a substantive one. It is still an obligation. On the "
        f"reporting base case the headline break-even risk is **{_base_rep*100:.3f}%** "
        f"({_base_rep*100:.4f}% to four decimals); the {BASIS_UNDISC_SHORT} value is "
        f"{_base_und*100:.4f}%, which is {(_base_rep/_base_und - 1)*100:.2f}% lower and "
        f"belongs only beside the closed form. The like-for-like attribution-adjusted "
        f"pair **on the reporting basis** is **{_base_rep*100:.3f}% -> "
        f"{_attr_rep*100:.3f}%** -- quote that pair, not the reporting-basis base case "
        f"against an undiscounted attribution-adjusted comparator, which would put two "
        f"conventions in one sentence."
    )
    L.append("")
    L.append(
        f"{n_in} of the {len(fam)} scenarios sit inside the examined span. **The "
        f"direction here is the opposite of what this build reported before A2, and it "
        f"matters.** The base case is now anchored on Pennington's MATCHED attributable "
        f"increment and sits at {_base_rep*100:.4f}%, inside the span. The row that falls "
        f"BELOW it is the ALL-CAUSE upper bound ({_attr_rep*100:.4f}% on the reporting "
        f"basis) -- charging Edmiston's regression-adjusted all-cause increment at face "
        f"value makes ciNPWT appear to pay off sooner, which is exactly why that estimand "
        f"is treated as a bound and not as the headline. Read none of this as external "
        f"validation: Mueller 2026 has no QALYs, no discounting, no horizon and no "
        f"uncertainty analysis, so containment in its swept input range is a coverage "
        f"statement about the risks it chose to examine, not agreement between two models."
    )
    L.append("")

    L.append("### 3.3 What is genuinely independent, and what is plumbing")
    L.append("")
    L.append(
        "Not every face-validity row carries the same evidential weight, and the "
        "difference is stated here rather than left for a reviewer to discover. "
        "**Genuinely independent:** the two NNT rows (`S1`, `S2`), which test the "
        "engine's incidence arithmetic against an externally published NNT; the weighted "
        "24-month cost per SSI in both price years (`S3`, `S4`), which ties the parameter "
        "store to the plan's own reported anchor and to a live BLS index; the "
        "hand-calc-versus-engine agreement (`S7`) and the price-year invariance proof "
        "(`S8`); and the Mueller band comparison (`S9`-`S12`), which is the only check "
        "here against a *different research group's* published output. "
        "**Plumbing, not corroboration:** the reoperation (`S5`) and persistent-infection "
        "(`S6`) rows. Under plan Decision 3 the engine computes `n_reop = n_ssi x "
        "p_reop_ssi` and `n_persistent = n_ssi_deep x p_persistent_deep`, and neither "
        "counter multiplies a cost. Those ratios therefore return the input parameters "
        "exactly, at every horizon and every baseline risk; they cannot fail while the "
        "wiring is intact, and they say nothing about whether the model resembles "
        "Karamian's or Freire-Archer's cohorts. They are worth running -- a counter that "
        "was silently dropped or double-applied would show up immediately -- but they are "
        "labelled `PLUMBING` so that they are never mistaken for external validation."
    )
    L.append("")

    L.append("### 3.4 Price years")
    L.append("")
    inv = by_id["S8_price_year_invariance"]
    L.append(
        f"Every dollar comparison in this suite is computed on **one** price basis. The "
        f"2025-USD cost target is not a hard-coded number: it is "
        f"`${TARGET_SSI_COST_2019:,.0f} x "
        f"CPI[2025]/CPI[2019]` read live from `analysis/data/cpi_medical.csv`, so it "
        f"follows the index if BLS revises it. `S8` demonstrates the invariance directly "
        f"-- deflating *everything* to 2019 USD reproduces the 2025-USD threshold exactly "
        f"(relative difference {inv['tolerance'].split('(observed ')[-1].rstrip(')')}) "
        f"-- which is the "
        f"arithmetic proof that the contract's original 0.746% was a mixed-basis "
        f"artifact (a 2025-USD device price over a 2019-USD SSI cost). BUILD-CONTRACT "
        f"section 10 was corrected twice for this class of error, once for the cost per "
        f"SSI and once for the thresholds; the checks above are written so that neither "
        f"correction can silently regress."
    )
    L.append("")

    if rep.hard_failures():
        L.append("### 3.5 HARD FAILURES")
        L.append("")
        L.append("**The engine is not structurally sound. Do not use these results.**")
        for r in rep.hard_failures():
            L.append(f"- `{r['check_id']}` -- {_md_cell(r['description'])}: "
                     f"observed {_md_cell(r['observed'])}")
        L.append("")
    else:
        L.append("### 3.5 Overall")
        L.append("")
        L.append(
            f"All {len(hard)} HARD / structural checks pass, so the engine is a correct "
            f"Markov process on its own terms: rows and traces sum to one, occupancy is "
            f"non-negative, death absorbs and never reverses, the cohort can never "
            f"out-survive the life table, the tunnels drain, the pivotal hazard is "
            f"bitwise zero outside the 90-day window, cumulative onset risk reproduces "
            f"the input `p` to 2e-16 across all 291 grid points, and the reported totals "
            f"are the sums of the streams the figures are drawn from. "
            + ("Every face-validity target reproduces within its reported "
               "tolerance, with one deliberate exception: the all-cause upper-bound row "
               "of the Mueller coverage comparison, "
               "which is `OUTSIDE` and is reported as a substantive finding."
               if not [r for r in sanity if r["result"] == "FAIL"] else
               "One or more face-validity checks FAILED -- see section 2; these do not "
               "affect the exit code but must be resolved before publication.")
        )
        L.append("")

    L.append("---")
    L.append("")
    L.append("Generated by `v2/analysis/validation.py`. Machine-readable form: "
             "`v2/results/validation_report.csv` "
             "(`pd.read_csv(path, skiprows=1)`).")
    L.append("")

    with open(path, "w") as fh:
        fh.write("\n".join(L))
    return path


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> int:
    print("=" * 78)
    print("validation.py -- INTERNAL VALIDATION SUITE")
    print(I.RESULTS_BANNER)
    print("=" * 78)
    print("HARD / structural checks are FATAL (assert; non-zero exit).")
    print("Clinical / face-validity checks are REPORTED, never asserted: a FAIL or an")
    print("OUTSIDE is a finding about the world, not a build failure.\n")

    rep = Report()

    print("-" * 78)
    print("HARD / STRUCTURAL (fatal)")
    print("-" * 78)
    for fn in HARD_CHECKS:
        _run(rep, fn)

    print()
    print("-" * 78)
    print("CLINICAL / FACE VALIDITY (reported, NOT fatal)")
    print("-" * 78)
    for fn in SANITY_CHECKS:
        _run(rep, fn)

    csv_path = write_csv(rep)
    md_path = write_md(rep)

    hard, sanity = rep.hard(), rep.sanity()
    n_hard_pass = sum(1 for r in hard if r["result"] == "PASS")
    counts: dict[str, int] = {}
    for r in sanity:
        counts[r["result"]] = counts.get(r["result"], 0) + 1

    print()
    print("=" * 78)
    print(f"HARD / structural   : {n_hard_pass}/{len(hard)} PASS  "
          f"(these and only these determine the exit code)")
    print(f"Face validity       : "
          + ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
          + f"  of {len(sanity)}  (reported; never changes the exit code)")
    for r in sanity:
        if r["result"] in ("FAIL", "OUTSIDE"):
            print(f"    >>> {r['result']:<8} {r['check_id']}: {r['observed']} "
                  f"[{BASIS_UNDISC_SHORT}]")
            if r["observed_reporting_basis"]:
                print(f"        {'':<8} {r['observed_reporting_basis']}"
                      f"  <- MANUSCRIPT HEADLINE")
    print("=" * 78)
    _hl = next((r for r in sanity if r["check_id"] == "S9_mueller_base"), None)
    if _hl is not None:
        print("HEADLINE break-even risk, BOTH BASES (BUILD-CONTRACT section 10, do not "
              "conflate):")
        print(f"  {BASIS_UNDISC_SHORT:<50} "
              f"{_hl['observed'].split(' hand-calc | ')[1]}")
        print(f"  {BASIS_REPORT_SHORT:<50} "
              f"{_hl['observed_reporting_basis'].split(' [', 1)[0]} -- "
              f"{_hl['observed_reporting_basis'].rsplit('-- ', 1)[-1]}")
        print("  -> the REPORTING BASE CASE value is the one Table 6 and the Results "
              "section must quote.")
        print("=" * 78)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")

    if rep.hard_failures():
        print("\nOVERALL: FAIL -- a HARD / structural check failed; exiting non-zero.")
        for r in rep.hard_failures():
            print(f"    {r['check_id']}: {r['description']} -> {r['observed']}")
        return 1

    print("\nOVERALL: PASS -- every HARD / structural check passed.")
    if any(r["result"] == "OUTSIDE" for r in sanity):
        print("NOTE: the engine is structurally sound AND a face-validity")
        print("      comparison is OUTSIDE its external band. That is a reported finding")
        print("      (plan Decision 8; BUILD-CONTRACT section 10 as corrected 2026-08-09),")
        print("      not a build failure -- see validation_report.md section 3.2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
