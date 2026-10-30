"""
run_psa.py -- Probabilistic sensitivity analysis for the SSI / ciNPWT risk-threshold
Markov cost-utility analysis, including the study's novel ACCEPTABILITY-BY-RISK output
(Fig 4): P(ciNPWT cost-effective) as a function of the baseline 90-day SSI risk `p`.

Imports the engine (`inputs.py` / `model.py`) READ-ONLY and writes the five BUILD-CONTRACT
section-8 deliverables to `v2/results/`:

    psa_iterations.csv         one row per draw at the INTERMEDIATE stratum (p = 0.066):
                               every sampled parameter + per-arm cost/QALY + incrementals
    psa_summary.csv            per stratum: mean / median / 2.5th / 97.5th percentile of
                               inc_cost, inc_qaly and INMB at each WTP; P(cost-effective)
                               at each WTP; P(dominant); P(dominated); CE-plane quadrants
    ceac.csv                   conventional CEAC over np.arange(0, 200_001, 2_500),
                               per stratum; probabilities in [0,1] and summing to 1
    acceptability_by_risk.csv  THE NOVEL OUTPUT (Fig 4): P(cost-effective | p) at
                               $50k / $100k / $150k per QALY over a risk subgrid
    psa_summary.md             results banner, results, the three BUILD-CONTRACT
                               section-9 sampling-rule disclosures, the realized cost-CV
                               diagnostic, and the full exclusion log

Every CSV carries `inputs.RESULTS_BANNER` as a '#'-prefixed LINE 1, followed by the header
row -- downstream readers use `pd.read_csv(path, skiprows=1)` (BUILD-CONTRACT section 7).

=============================================================================
PSA DISTRIBUTION CONVENTION (method of moments; [low, high] = approx 95% CI)
=============================================================================
For every sampled Param, SE = (high - low) / 3.92 and the MEAN is preserved at `value`,
EXCEPT relative effects, which are sampled on the LOG scale preserving the MEDIAN:

  beta      probabilities and utilities in (0,1)
            m = value, var = SE**2 (guarded below m*(1-m));
            alpha = m * (m*(1-m)/var - 1),  beta = (1-m) * (m*(1-m)/var - 1)
  gamma     costs > 0
            shape k = (m/SE)**2,  scale theta = SE**2 / m
  lognormal relative effects
            sdlog = (ln(high) - ln(low)) / 3.92,  meanlog = ln(value)
            MEDIAN-PRESERVING, so the median draw reproduces the base case exactly and
            (with RR < 1 in every draw) arm ordering is preserved draw by draw. The
            mean-preserving alternative meanlog = ln(value) - sdlog**2/2 is NOT used.
  normal    utility DECREMENTS, TRUNCATED AT 0 BY REJECTION (see disclosure 2 below)
            NOTE: `c_device_cinpwt` (gamma) is likewise TRUNCATED, to its declared
            $100-$900 range, by the same rejection idiom -- see decision A below
  dirichlet destination splits, grouped by `group`; concentration from
            SETTINGS['dirichlet_alpha_ssi_split'], NOT the raw Edmiston counts
  fixed     NOT sampled -- the base value is used

THE THREE BINDING SAMPLING RULES (BUILD-CONTRACT section 9, settled 2026-08-09)
------------------------------------------------------------------------------
1. `p_base_ssi` IS NEVER SAMPLED. It carries dist='beta' but it is the study's x-axis,
   not an uncertain input: P(cost-effective | p) CONDITIONS on p, and a generic "sample
   every non-fixed Param" loop would integrate over it and destroy
   acceptability_by_risk.csv. It is excluded BY NAME and the exclusion is logged in
   psa_summary.md rather than hidden.
2. `du_sup` / `du_deep` are `normal` on a quantity that must be non-negative
   (a negative draw is a utility GAIN from SSI). They are TRUNCATED AT 0 BY REJECTION
   SAMPLING -- resampled, not clipped, so no point mass is created at exactly 0. The
   number of rejected draws is reported.
3. `m_ssi_cost` STAYS `fixed`. SSI-attributable-cost attribution is structural
   between-source disagreement (Pennington's matched increment vs Edmiston's
   regression-adjusted increment, a ~3.9-fold span) and is handled by the all-cause
   x1.000 sensitivity and by the DSA, not by sampling. (A2 2026-08-17 inverted this:
   the MATCHED anchor m_ssi_cost = 0.492 is now the base case, because Edmiston's
   increment is regression-adjusted ALL-CAUSE spend with no matching and is therefore an
   upper bound on attributable cost by construction.) Two consequences are DISCLOSED, not
   hidden: (a) the CEAC does not carry attribution uncertainty and is therefore more
   confident than the evidence supports; (b) the six cost tranches are sampled as
   INDEPENDENT gammas, so their weighted sum has a smaller CV than any single tranche.
   The REALIZED coefficient of variation of the total per-SSI cost across draws is
   reported as a diagnostic so the reader can see how much cost uncertainty the PSA
   actually carries.

TWO ORCHESTRATOR DECISIONS ADDED 2026-08-09 (both disclosure/completeness, no redesign)
--------------------------------------------------------------------------------------
A. `c_device_cinpwt` IS TRUNCATED TO ITS DECLARED $100-$900 RANGE by rejection resampling
   (`TRUNCATE_TO_RANGE`), consistent with how `du_sup` / `du_deep` are handled. That range
   is analysis axis 2 -- the reported `SETTINGS['price_grid']` -- and the untruncated
   method-of-moments gamma put draws outside the axis at both ends. Truncation
   does NOT resolve the deeper issue that the device price is represented twice (sampled
   here, swept there), so `psa_summary.md` additionally reports P(cost-effective) at the
   low stratum BOTH with the device sampled AND with it held fixed at its base value --
   `device_fixed_counterfactual()` -- so a reader can see how much of that stratum's
   width is device-price assumption rather than evidence.
B. THE PSA MEANS DO NOT REPRODUCE THE DETERMINISTIC BASE CASE, and `psa_vs_deterministic()`
   says why in the summary instead of letting a reader find an unexplained ~1% gap between
   Table 2 and the PSA table. The mandated median-preserving lognormal on `rr_cinpwt` has
   mean 0.61 x exp(sdlog^2/2) = 0.6138 against the deterministic 0.61; a higher mean RR
   averts fewer SSIs, so the mean INMB across draws sits about 1% below the deterministic
   value at the intermediate and high strata (relatively more at the low stratum, where
   decision A's device shift is a large share of a small INMB). It is a property of the
   fit, not Monte-Carlo noise -- several Monte-Carlo standard errors at every stratum --
   and the reconciliation is MEASURED, by re-running the engine at the mean of the draws.

CORRELATION / ARM ORDERING
--------------------------
The engine applies `p_ciNPWT = p_control x RR` INTERNALLY, so the two arms are correlated
by construction and the RR is never pre-multiplied here. Zero arm-order inversions
(`n_ssi_B < n_ssi_A`) are asserted across every draw at every risk point.

PRICE YEARS
-----------
A cost draw in a `sample` dict is expressed in that parameter's OWN stored price year --
the engine inflates it exactly once inside `_cost()`. The six commercial SSI tranches are
drawn in 2019 USD; `c_device_cinpwt` is drawn in 2025 USD. Costs are never read with
`val()`.

RUNTIME
-------
`run_pair` is ~320 us, so 10,000 draws cost ~6 s per risk point. ONE SAMPLE SET of 10,000
draws is generated and REUSED at every risk point (common random numbers): this is what
makes the acceptability curve smooth and MONOTONE -- for a fixed draw INMB is strictly
increasing in `p`, so P(INMB > 0) cannot decrease -- and it keeps the sweep to a single
pass. The risk subgrid is every 5th point of SETTINGS['risk_grid'] plus the three reported
strata (61 of 291 points), which is ~3.3 min in total.

Run with cwd = v2/analysis/ :  python3 run_psa.py
"""

from __future__ import annotations

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import inputs as I
import model

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V1_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
RESULTS_DIR = os.path.join(V1_DIR, "results")

# --------------------------------------------------------------------------- #
# Run settings -- every one of them read from inputs.py (BUILD-CONTRACT 9:      #
# only inputs.py may own a number)                                             #
# --------------------------------------------------------------------------- #
N_ITER = int(I.SETTINGS["psa_iterations"])                 # 10,000
SEED = int(I.SETTINGS["seed"])                             # 12345
HORIZON = int(I.SETTINGS["horizon_cycles"])                # 24 cycles = 2 years
DISC_COST = float(I.SETTINGS["disc_cost"])
DISC_QALY = float(I.SETTINGS["disc_qaly"])
HALF_CYCLE = bool(I.SETTINGS["half_cycle"])
WTP = float(I.SETTINGS["wtp_primary"])                     # 100,000
WTP_LO, WTP_HI = (float(w) for w in I.SETTINGS["wtp_report"])   # 50,000 / 150,000
WTP_REPORTED = (WTP_LO, WTP, WTP_HI)
CEAC_GRID = np.asarray(I.SETTINGS["ceac_grid"], dtype=float)
RISK_GRID = np.asarray(I.SETTINGS["risk_grid"], dtype=float)
STRATA = dict(I.SETTINGS["strata"])                        # low / intermediate / high
BASE_STRATUM = "intermediate"                              # p = 0.066, the iterations file

#: The acceptability-by-risk subgrid: every 5th point of the 291-point risk grid, PLUS the
#: three reported strata (which do not all land on a multiple of 5 -- 0.066 is index 122
#: and 0.114 is index 218). 61 points. Stated in psa_summary.md.
RISK_SUBGRID_STEP = 5

#: Method-of-moments divisor: [low, high] read as an approximate 95% CI.
Z95 = 3.92
EPS = 1e-9

#: Maximum rejection-sampling passes for a truncated normal before giving up. ~1e-7 of
#: draws are negative, so one pass is essentially always enough; the cap exists so a
#: mis-specified parameter fails loudly instead of spinning.
MAX_REJECTION_PASSES = 100


# --------------------------------------------------------------------------- #
# 1. Which parameters are sampled -- and, explicitly, which are not            #
# --------------------------------------------------------------------------- #
#: BUILD-CONTRACT section 9 rule 1. Excluded BY NAME, never by a dist test.
AXIS_EXCLUSIONS = {
    "p_base_ssi": (
        "THE STUDY'S X-AXIS. Carries dist='beta' and a (0.005, 0.15) range, but "
        "P(cost-effective | p) conditions on p; sampling it would integrate over the "
        "x-axis and destroy acceptability_by_risk.csv and Fig 4 "
        "(BUILD-CONTRACT section 9 rule 1; run_log.d/inputs.md objection 2)."
    ),
}

#: The parameters `run_arm` actually reads (run_log.d/model.md, "sample dicts and val()").
#: VERIFIED EMPIRICALLY at run time by `_probe_engine_read()` -- a declared list that
#: silently fell out of step with the engine would put inert columns in
#: psa_iterations.csv and make the PSA look broader than it is.
ENGINE_READ_PARAMS = frozenset({
    "rr_cinpwt",
    "frac_deep", "frac_sup",
    "w_onset_deep_m1", "w_onset_deep_m2", "w_onset_deep_m3",
    "w_onset_sup_m1", "w_onset_sup_m2", "w_onset_sup_m3",
    "p_reop_ssi", "p_persistent_deep",
    "m_ssi_cost",
    "c_ssi_sup_0_6mo", "c_ssi_sup_6_12mo", "c_ssi_sup_12_24mo",
    "c_ssi_deep_0_6mo", "c_ssi_deep_6_12mo", "c_ssi_deep_12_24mo",
    "c_device_cinpwt", "c_dressing_standard",
    "u_well", "du_sup", "du_deep", "du_recovered_residual",
    "p_periop_death", "age_start", "frac_male",
})

#: Utility DECREMENTS: `normal` on a must-be-non-negative magnitude
#: (BUILD-CONTRACT section 9 rule 2).
TRUNCATE_AT_ZERO = ("du_sup", "du_deep")

#: Parameters whose draws are TRUNCATED TO THEIR OWN DECLARED [low, high] BY REJECTION --
#: the same idiom as TRUNCATE_AT_ZERO, one step wider (both tails rather than one).
#:
#: ORCHESTRATOR DECISION 2026-08-09, axis widened by A2 2026-08-17 -- `c_device_cinpwt`
#: is truncated to $100-$900 (was $100-$600 until A2 put Mueller 2026 [9]'s $495 and $830
#: list prices on the reported axis).
#: The $100-$900 span is not a soft plausibility range: it is analysis axis 2, the
#: reported 161-point `SETTINGS['price_grid']` that Fig 3 and the threshold-price
#: results are built on. A method-of-moments gamma fitted to it (CV 0.51) ignores those
#: bounds entirely and put draws outside the axis at both ends, i.e. below and
#: above the range the study itself records as plausible -- a $17 and a $1,087
#: ciNPWT unit are wrong in both directions. Truncation makes the sampled price live on
#: the same axis the paper reports thresholds against. It does NOT fix the fact that the
#: device price is represented twice (sampled here, swept there); that is disclosed below.
TRUNCATE_TO_RANGE = ("c_device_cinpwt",)


def classify_params() -> dict:
    """Partition `inputs.all_params()` into sampled and excluded, with a reason for every
    exclusion. This IS the exclusion log printed into psa_summary.md -- the contract
    requires the exclusions be visible rather than hidden inside a loop condition."""
    sampled, excluded = [], []
    for p in I.all_params():
        if p.name in AXIS_EXCLUSIONS:
            excluded.append((p, "axis (never sampled)", AXIS_EXCLUSIONS[p.name]))
        elif p.dist == "fixed":
            reason = (
                "dist='fixed'. BUILD-CONTRACT section 9 rule 3: attribution is structural "
                "between-source disagreement, characterized DETERMINISTICALLY by the "
                "all-cause x1.000 sensitivity and by the DSA (where it must be the top "
                "tornado bar), not probabilistically. THE CEAC THEREFORE CARRIES NO "
                "ATTRIBUTION UNCERTAINTY."
                if p.name == "m_ssi_cost" else
                "dist='fixed' -- not sampled; the base value is used."
            )
            excluded.append((p, "fixed", reason))
        elif p.name not in ENGINE_READ_PARAMS:
            excluded.append((
                p, "not read by the engine",
                "the engine never reads this parameter (scenario / secondary-analysis / "
                "cross-check input), so a draw would be a silently inert column in "
                "psa_iterations.csv -- an over-broad sample that makes the PSA look "
                "wider than it is. Its uncertainty is carried by the scenario or DSA "
                "that does use it.",
            ))
        else:
            sampled.append(p)
    return {"sampled": sampled, "excluded": excluded}


def _probe_engine_read(p_probe: float) -> set:
    """Empirically determine which Params the engine actually reads, by perturbing each
    non-fixed parameter to its `high` bound one at a time and watching every output the
    engine reports (costs, QALYs and the two counters).

    ~56 `run_pair` calls, i.e. ~20 ms. It exists so `ENGINE_READ_PARAMS` cannot drift out
    of step with `model.py` without this script noticing -- the exact failure mode
    BUILD-CONTRACT section 9 rule 1 is about, one level down.
    """
    keys = ("cost", "qaly", "n_reop", "n_persistent", "n_ssi")
    base_a, base_b = model.run_pair(p_base=p_probe, horizon_cycles=HORIZON,
                                    disc_cost=DISC_COST, disc_qaly=DISC_QALY,
                                    half_cycle=HALF_CYCLE)
    base = np.array([base_a[k] for k in keys] + [base_b[k] for k in keys], dtype=float)
    read = set()
    for p in I.all_params():
        if p.dist == "fixed" or p.name in AXIS_EXCLUSIONS:
            continue
        bump = float(p.high) if float(p.high) != float(p.value) else float(p.value) * 2.0
        a, b = model.run_pair(p_base=p_probe, horizon_cycles=HORIZON,
                              disc_cost=DISC_COST, disc_qaly=DISC_QALY,
                              half_cycle=HALF_CYCLE, sample={p.name: bump})
        got = np.array([a[k] for k in keys] + [b[k] for k in keys], dtype=float)
        if np.max(np.abs(got - base)) > 1e-12 * max(1.0, np.max(np.abs(base))):
            read.add(p.name)
    return read


# --------------------------------------------------------------------------- #
# 2. Distribution parameterization (method of moments)                         #
# --------------------------------------------------------------------------- #
def _se(p) -> float:
    """SE implied by treating [low, high] as an approximate 95% CI."""
    return (float(p.high) - float(p.low)) / Z95


def _beta_ab(m: float, se: float) -> tuple[float, float]:
    """Beta (alpha, beta) from mean `m` and SE, guarding var < m*(1-m)."""
    var = se * se
    var_max = m * (1.0 - m)
    if var <= 0.0 or var_max <= 0.0:
        return max(m, 1e-6) * 1e4, max(1.0 - m, 1e-6) * 1e4      # near point mass
    if var >= var_max:
        var = 0.99 * var_max
    common = m * (1.0 - m) / var - 1.0
    return max(m * common, 1e-6), max((1.0 - m) * common, 1e-6)


def _gamma_k_theta(m: float, se: float) -> tuple[float, float]:
    """Gamma (shape k, scale theta) from mean `m` and SE."""
    if se <= 0.0 or m <= 0.0:
        return 1e6, max(m, 1e-9) / 1e6                            # near point mass at m
    return (m / se) ** 2, (se * se) / m


def _lognormal_params(p) -> tuple[float, float]:
    """(meanlog, sdlog), MEDIAN-preserving: meanlog = ln(value),
    sdlog = (ln(high) - ln(low)) / 3.92.

    Median- rather than mean-preserving because the parameter this matters for is
    `rr_cinpwt`: the median draw must reproduce the base RR of 0.61 exactly, and the
    whole draw distribution must stay below 1 so the arm ordering the engine builds in
    (`p_ciNPWT = p_control x RR`) is never inverted by the sampler.
    """
    return float(np.log(p.value)), (np.log(float(p.high)) - np.log(float(p.low))) / Z95


def _normal_truncated_at_zero(rng, m: float, se: float, n: int) -> tuple[np.ndarray, int]:
    """Normal(m, se) truncated at 0 BY REJECTION (resample, not clip).

    Clipping would pile every negative draw onto exactly 0, creating a spurious point
    mass at "SSI causes no utility loss at all"; resampling preserves the shape of the
    conditional distribution. Returns (draws, n_rejected).
    """
    draws = rng.normal(m, se, n)
    rejected = 0
    for _ in range(MAX_REJECTION_PASSES):
        bad = draws < 0.0
        n_bad = int(bad.sum())
        if n_bad == 0:
            return draws, rejected
        rejected += n_bad
        draws[bad] = rng.normal(m, se, n_bad)
    raise RuntimeError(
        f"rejection sampling for a truncated normal (mean {m}, SE {se}) did not converge "
        f"in {MAX_REJECTION_PASSES} passes -- the parameter is not plausibly non-negative"
    )


def _truncate_to_range(draw, redraw, lo: float, hi: float, n: int,
                       name: str) -> tuple[np.ndarray, dict]:
    """Draw `n` values from `draw(n)` and REJECTION-RESAMPLE until every one lies in
    [lo, hi]. Same idiom as `_normal_truncated_at_zero`, both tails.

    Resampled, never clipped: clipping would pile the out-of-range mass onto exactly the
    two bounds and manufacture spurious point masses at the ends of the reported
    price axis -- "the device costs exactly $900" is a different claim from "the device is
    expensive".

    **The replacements come from `redraw`, a SEPARATE seed-derived stream, not from the
    main sampler.** Rejection sampling consumes a variable number of random numbers, so
    drawing the replacements from the main generator would shift the stream for every
    parameter sampled after this one -- utilities, perioperative mortality and the
    Dirichlet split would all silently change draws because of a decision about the device
    price. With a separate stream the FIRST pass is bit-identical to an untruncated run
    and **the device column is the only column that changes**, which is what makes the
    effect of the truncation attributable rather than merely observed.

    Returns (draws, diagnostics); the diagnostics record the FIRST pass, i.e. the
    untruncated fit, so the summary can report how much mass that fit put outside the axis
    rather than merely asserting that the shipped draws are inside it.
    """
    x = np.asarray(draw(n), dtype=float)
    diag = {
        "lo": float(lo), "hi": float(hi), "n": int(n),
        "n_below_untruncated": int((x < lo).sum()),
        "n_above_untruncated": int((x > hi).sum()),
        "min_untruncated": float(x.min()),
        "max_untruncated": float(x.max()),
        "mean_untruncated": float(x.mean()),
        "median_untruncated": float(np.median(x)),
        "cv_untruncated": float(np.std(x, ddof=1) / np.mean(x)),
    }
    rejected = 0
    for _ in range(MAX_REJECTION_PASSES):
        bad = (x < lo) | (x > hi)
        n_bad = int(bad.sum())
        if n_bad == 0:
            diag["n_rejected"] = rejected
            return x, diag
        rejected += n_bad
        x[bad] = np.asarray(redraw(n_bad), dtype=float)
    raise RuntimeError(                                      # pragma: no cover
        f"rejection sampling of {name!r} into [{lo}, {hi}] did not converge in "
        f"{MAX_REJECTION_PASSES} passes -- the fitted distribution puts almost no mass "
        "inside the declared range"
    )


# --------------------------------------------------------------------------- #
# 3. Sampler                                                                   #
# --------------------------------------------------------------------------- #
def make_samples(n_iter: int = N_ITER, seed: int = SEED) -> dict:
    """Draw `n_iter` parameter sets. Returns a dict with the draws DataFrame, the list of
    per-draw `sample` dicts consumed by `model.run_arm(sample=...)`, the fitted
    distribution table, and the sampling diagnostics.

    Cost draws are in the parameter's OWN stored price year -- the engine inflates them
    exactly once. Nothing is pre-inflated and no cost is read with `val()`.
    """
    rng = np.random.default_rng(seed)
    #: Dedicated, seed-derived stream for rejection REPLACEMENTS in `TRUNCATE_TO_RANGE`.
    #: Rejection consumes a variable number of random numbers, so taking the replacements
    #: from `rng` would shift every parameter drawn after the truncated one. Keeping them
    #: on their own stream means the truncation changes exactly one column and nothing
    #: else, and the run stays fully reproducible from `seed` alone.
    rng_trunc = np.random.default_rng([int(seed), 1])
    split = classify_params()
    cols: dict[str, np.ndarray] = {}
    fitted: list[dict] = []
    n_rejected: dict[str, int] = {}
    trunc_diag: dict[str, dict] = {}

    # ---- Dirichlet groups, collected in declaration order ---------------------------- #
    dgroups: dict[str, list] = {}
    for p in split["sampled"]:
        if p.dist == "dirichlet":
            dgroups.setdefault(p.group, []).append(p)

    # ---- scalar draws ---------------------------------------------------------------- #
    for p in split["sampled"]:
        if p.dist == "dirichlet":
            continue
        se = _se(p)
        if p.dist == "beta":
            a, b = _beta_ab(float(p.value), se)
            cols[p.name] = rng.beta(a, b, n_iter)
            spec = f"beta(alpha={a:.4g}, beta={b:.4g})"
        elif p.dist == "gamma":
            k, theta = _gamma_k_theta(float(p.value), se)
            if p.name in TRUNCATE_TO_RANGE:
                draws, diag = _truncate_to_range(
                    lambda m: rng.gamma(k, theta, m),        # main stream, first pass
                    lambda m: rng_trunc.gamma(k, theta, m),  # separate stream, redraws
                    float(p.low), float(p.high), n_iter, p.name)
                cols[p.name] = draws
                n_rejected[p.name] = diag["n_rejected"]
                trunc_diag[p.name] = diag
                spec = (f"gamma(k={k:.4g}, theta={theta:.4g}) TRUNCATED TO "
                        f"[{float(p.low):g}, {float(p.high):g}] BY REJECTION")
            else:
                cols[p.name] = rng.gamma(k, theta, n_iter)
                spec = f"gamma(k={k:.4g}, theta={theta:.4g})"
        elif p.dist == "lognormal":
            ml, sl = _lognormal_params(p)
            cols[p.name] = rng.lognormal(ml, sl, n_iter)
            spec = f"lognormal(meanlog={ml:.6g}, sdlog={sl:.5g}) MEDIAN-preserving"
        elif p.dist == "normal":
            if p.name in TRUNCATE_AT_ZERO:
                draws, rej = _normal_truncated_at_zero(rng, float(p.value), se, n_iter)
                cols[p.name] = draws
                n_rejected[p.name] = rej
                spec = f"normal(mean={float(p.value):.6g}, sd={se:.5g}) TRUNCATED AT 0"
            else:
                cols[p.name] = rng.normal(float(p.value), se, n_iter)
                spec = f"normal(mean={float(p.value):.6g}, sd={se:.5g})"
        else:                                                    # pragma: no cover
            raise ValueError(f"{p.name}: unhandled dist {p.dist!r}")
        fitted.append({"parameter": p.name, "dist": p.dist, "value": float(p.value),
                       "low": float(p.low), "high": float(p.high), "se": se,
                       "fitted": spec, "price_year": p.price_year,
                       "note": "cost draw in this parameter's OWN price year"
                               if p.price_year is not None else ""})

    # ---- Dirichlet groups ------------------------------------------------------------ #
    # The `ssi_split` pair uses the DOWN-WEIGHTED concentration exposed by inputs.py
    # (kappa = 10.767, alpha = (7.99, 2.78)), NOT the raw Edmiston claims counts
    # (~10,291 deep / ~3,570 superficial) -- those would make the PSA treat the split as
    # known, when the real uncertainty is claims-coding validity.
    for g, members in dgroups.items():
        if g == "ssi_split":
            alpha = np.asarray(I.SETTINGS["dirichlet_alpha_ssi_split"], dtype=float)
            if len(alpha) != len(members):
                raise ValueError(
                    f"SETTINGS['dirichlet_alpha_ssi_split'] has {len(alpha)} entries but "
                    f"group 'ssi_split' has {len(members)} members"
                )
            # Order check: alpha is (deep, sup) and members are in declaration order.
            base = np.array([float(m.value) for m in members], dtype=float)
            base = base / base.sum()
            if not np.allclose(alpha / alpha.sum(), base, atol=5e-4):
                raise ValueError(
                    "dirichlet_alpha_ssi_split is not aligned with the declaration order "
                    f"of group 'ssi_split' ({[m.name for m in members]}): "
                    f"alpha implies {alpha / alpha.sum()}, params imply {base}"
                )
            kappa = float(I.SETTINGS["dirichlet_kappa_ssi_split"])
        else:                                                    # pragma: no cover
            base = np.array([float(m.value) for m in members], dtype=float)
            base = base / base.sum()
            kappa = float(I.SETTINGS["dirichlet_kappa_ssi_split"])
            alpha = base * kappa
        draws = rng.dirichlet(alpha, n_iter)
        for j, m in enumerate(members):
            cols[m.name] = draws[:, j]
            fitted.append({
                "parameter": m.name, "dist": "dirichlet", "value": float(m.value),
                "low": float(m.low), "high": float(m.high), "se": _se(m),
                "fitted": (f"dirichlet(alpha={tuple(round(float(a), 4) for a in alpha)}, "
                           f"kappa={kappa:g}) group={g!r}"),
                "price_year": m.price_year,
                "note": "DOWN-WEIGHTED concentration, NOT the raw Edmiston counts",
            })

    df = pd.DataFrame(cols)
    # Column order: declaration order of the sampled parameters.
    df = df[[p.name for p in split["sampled"]]]
    samples = df.to_dict(orient="records")

    # Every truncated parameter must actually be inside its declared range in the shipped
    # draws -- asserted here rather than trusted, because a silent failure would put a
    # $1,087 ciNPWT unit back into the CEAC.
    for name in TRUNCATE_TO_RANGE:
        if name in df.columns:
            p = I.param_by_name(name)
            x = df[name].to_numpy()
            assert x.min() >= float(p.low) and x.max() <= float(p.high), (
                f"{name}: truncation to [{float(p.low)}, {float(p.high)}] failed -- "
                f"realized range {x.min():.2f} to {x.max():.2f}"
            )

    return {
        "draws": df,
        "samples": samples,
        "fitted": pd.DataFrame(fitted),
        "sampled_params": split["sampled"],
        "excluded": split["excluded"],
        "n_rejected": n_rejected,
        "trunc_diag": trunc_diag,
        "n_iter": n_iter,
        "seed": seed,
    }


# --------------------------------------------------------------------------- #
# 4. Dominance / CE-plane classification (vectorized mirror of model.incremental) #
# --------------------------------------------------------------------------- #
_DOM_EPS = 1e-12


def classify_dominance(inc_cost: np.ndarray, inc_qaly: np.ndarray) -> np.ndarray:
    """Vectorized `dominance` label, in the SAME branch order as `model.incremental`.

    Reproduced here (rather than looping over 610,000 `incremental` calls) and then
    ASSERTED equal to the engine's own strings on all 10,000 intermediate-stratum draws,
    so the two can never silently disagree.
    """
    ic = np.asarray(inc_cost, dtype=float)
    iq = np.asarray(inc_qaly, dtype=float)
    cost_more, cost_less = ic > _DOM_EPS, ic < -_DOM_EPS
    qaly_more, qaly_less = iq > _DOM_EPS, iq < -_DOM_EPS
    conds = [
        ~(cost_more | cost_less | qaly_more | qaly_less),
        qaly_more & ~cost_more,
        qaly_less & ~cost_less,
        cost_more & qaly_more,
        cost_less & qaly_less,
        cost_less,
    ]
    choices = ["equal", "ciNPWT dominant", "ciNPWT dominated",
               "ciNPWT more costly & more effective",
               "ciNPWT less costly & less effective", "ciNPWT dominant"]
    return np.select(conds, choices, default="ciNPWT dominated")


# --------------------------------------------------------------------------- #
# 5. The Monte-Carlo sweep                                                     #
# --------------------------------------------------------------------------- #
def risk_subgrid() -> tuple[np.ndarray, np.ndarray, dict]:
    """The acceptability-by-risk subgrid: every `RISK_SUBGRID_STEP`-th point of the
    291-point `SETTINGS['risk_grid']`, UNION the three reported strata (0.066 and 0.114
    are not multiples of 5 on that grid). Returns (indices, p values, stratum -> position
    within the subgrid)."""
    idx = set(range(0, len(RISK_GRID), RISK_SUBGRID_STEP))
    strat_idx = {}
    for name, p in STRATA.items():
        j = int(np.argmin(np.abs(RISK_GRID - p)))
        if abs(RISK_GRID[j] - p) > 1e-12:
            raise ValueError(f"stratum {name} (p={p}) is not on SETTINGS['risk_grid']")
        idx.add(j)
        strat_idx[name] = j
    ordered = np.array(sorted(idx), dtype=int)
    pos = {name: int(np.where(ordered == j)[0][0]) for name, j in strat_idx.items()}
    return ordered, RISK_GRID[ordered], pos


def run_sweep(samples: list[dict], p_values: np.ndarray, strata_pos: dict,
              verbose: bool = True) -> dict:
    """Run `run_pair` for every (risk point x draw) using ONE reused sample set.

    Common random numbers across risk points is not just a speed-up: for a fixed draw
    INMB is strictly increasing in `p` (more SSI averted, same device toll), so reusing
    the draws is what makes the acceptability curve monotone rather than jittery.

    Returns arrays of shape (n_risk, n_iter) for inc_cost / inc_qaly, plus the per-arm
    cost/QALY at the three strata and the arm-ordering audit.
    """
    n_p, n_iter = len(p_values), len(samples)
    inc_cost = np.empty((n_p, n_iter), dtype=float)
    inc_qaly = np.empty((n_p, n_iter), dtype=float)
    arms = {name: {k: np.empty(n_iter) for k in ("cost_a", "qaly_a", "cost_b", "qaly_b")}
            for name in strata_pos}
    stratum_at_pos = {pos: name for name, pos in strata_pos.items()}

    n_inversions = 0
    worst_gap = np.inf                      # min over draws of (n_ssi_A - n_ssi_B)
    t0 = time.perf_counter()
    for i, p in enumerate(p_values):
        sname = stratum_at_pos.get(i)
        for j, s in enumerate(samples):
            res_a, res_b = model.run_pair(p_base=float(p), horizon_cycles=HORIZON,
                                          disc_cost=DISC_COST, disc_qaly=DISC_QALY,
                                          half_cycle=HALF_CYCLE, sample=s)
            inc_cost[i, j] = res_b["cost"] - res_a["cost"]
            inc_qaly[i, j] = res_b["qaly"] - res_a["qaly"]
            gap = res_a["n_ssi"] - res_b["n_ssi"]
            if gap <= 0.0:
                n_inversions += 1
            if gap < worst_gap:
                worst_gap = gap
            if sname is not None:
                arms[sname]["cost_a"][j] = res_a["cost"]
                arms[sname]["qaly_a"][j] = res_a["qaly"]
                arms[sname]["cost_b"][j] = res_b["cost"]
                arms[sname]["qaly_b"][j] = res_b["qaly"]
        if verbose and (i % 10 == 0 or i == n_p - 1):
            el = time.perf_counter() - t0
            print(f"    risk point {i + 1:>3d}/{n_p}  p = {p:.4f}   "
                  f"[{el:6.1f}s elapsed, ~{el / (i + 1) * (n_p - i - 1):5.0f}s left]",
                  flush=True)

    return {"inc_cost": inc_cost, "inc_qaly": inc_qaly, "arms": arms,
            "n_inversions": n_inversions, "min_ssi_gap": float(worst_gap),
            "seconds": time.perf_counter() - t0}


# --------------------------------------------------------------------------- #
# 6. Cost-uncertainty diagnostic (BUILD-CONTRACT section 9 rule 3)             #
# --------------------------------------------------------------------------- #
def cost_cv_diagnostic(draws: pd.DataFrame) -> dict:
    """Realized coefficient of variation of the TOTAL per-SSI cost across draws.

    The six commercial tranches are sampled as INDEPENDENT gammas, so their weighted sum
    has a materially smaller CV than any single tranche -- independence shrinks aggregate
    variance. With `m_ssi_cost` held `fixed`, this number IS the whole of the cost
    uncertainty the CEAC carries, so it is reported explicitly.

    Every tranche draw is converted to 2025 USD through `inputs.cost_2025(param, draw)`,
    vectorized via its exact price-year factor `cost_2025(param, 1.0)` (cost_2025 is
    linear in `value`; asserted below). Nothing is pre-inflated and no cost is `val()`ed.
    """
    sup_params = (I.cost.c_ssi_sup_0_6mo, I.cost.c_ssi_sup_6_12mo, I.cost.c_ssi_sup_12_24mo)
    deep_params = (I.cost.c_ssi_deep_0_6mo, I.cost.c_ssi_deep_6_12mo, I.cost.c_ssi_deep_12_24mo)

    def _f(param) -> float:
        f = float(I.cost_2025(param, 1.0))
        probe = float(param.value)
        assert abs(I.cost_2025(param, probe) - probe * f) < 1e-6, \
            f"cost_2025 is not linear in value for {param.name}"
        return f

    sup = np.zeros(len(draws))
    deep = np.zeros(len(draws))
    per_tranche = {}
    for p in sup_params:
        d = draws[p.name].to_numpy() * _f(p)
        sup += d
        per_tranche[p.name] = d
    for p in deep_params:
        d = draws[p.name].to_numpy() * _f(p)
        deep += d
        per_tranche[p.name] = d

    fd = draws["frac_deep"].to_numpy()
    fs = draws["frac_sup"].to_numpy()
    tot = fd + fs
    fd, fs = fd / tot, fs / tot
    weighted = fd * deep + fs * sup

    def _cv(x) -> float:
        return float(np.std(x, ddof=1) / np.mean(x))

    m_ssi = I.cost.m_ssi_cost
    return {
        "per_tranche_cv": {k: _cv(v) for k, v in per_tranche.items()},
        "mean_tranche_cv": float(np.mean([_cv(v) for v in per_tranche.values()])),
        "cv_deep_total": _cv(deep),
        "cv_sup_total": _cv(sup),
        "cv_weighted": _cv(weighted),
        "mean_weighted": float(np.mean(weighted)),
        "sd_weighted": float(np.std(weighted, ddof=1)),
        "p2_5_weighted": float(np.percentile(weighted, 2.5)),
        "p97_5_weighted": float(np.percentile(weighted, 97.5)),
        "cv_frac_deep": _cv(fd),
        # What the PSA does NOT carry: the attribution multiplier, held `fixed`. Read on
        # the same method-of-moments footing as everything else, its DSA range implies
        # this CV -- roughly as large as the entire cost CV the PSA does carry.
        "implied_cv_m_ssi_cost": float(
            (float(m_ssi.high) - float(m_ssi.low)) / Z95 / float(m_ssi.value)),
        "m_ssi_range": (float(m_ssi.low), float(m_ssi.value), float(m_ssi.high)),
        "weighted": weighted,
    }


# --------------------------------------------------------------------------- #
# 7. Deliverables                                                              #
# --------------------------------------------------------------------------- #
def build_iterations(sm: dict, sweep: dict, strata_pos: dict, p_values: np.ndarray):
    """psa_iterations.csv -- one row per draw at the INTERMEDIATE stratum (p = 0.066)."""
    pos = strata_pos[BASE_STRATUM]
    p = float(p_values[pos])
    a = sweep["arms"][BASE_STRATUM]
    ic = sweep["inc_cost"][pos]
    iq = sweep["inc_qaly"][pos]

    # Dominance from the engine itself (`model.incremental` reads only cost/qaly), then
    # the vectorized classifier is asserted to agree on every draw.
    dom = [model.incremental({"cost": a["cost_b"][j], "qaly": a["qaly_b"][j]},
                             {"cost": a["cost_a"][j], "qaly": a["qaly_a"][j]},
                             wtp=WTP)["dominance"] for j in range(len(ic))]
    dom = np.array(dom, dtype=object)
    vec = classify_dominance(ic, iq)
    assert np.all(vec == dom), "vectorized dominance disagrees with model.incremental"

    out = pd.DataFrame({"iteration": np.arange(len(ic), dtype=int)})
    for name in sm["draws"].columns:
        out[name] = sm["draws"][name].to_numpy()
    out["cost_a"] = a["cost_a"]
    out["qaly_a"] = a["qaly_a"]
    out["cost_b"] = a["cost_b"]
    out["qaly_b"] = a["qaly_b"]
    out["inc_cost"] = ic
    out["inc_qaly"] = iq
    out["inmb_100k"] = iq * WTP - ic
    out["dominance"] = dom
    out["quadrant"] = [model.DOMINANCE_TO_QUADRANT[d] for d in dom]
    assert "p_base_ssi" not in out.columns, \
        "CONTRACT section 9 rule 1 VIOLATION: p_base_ssi reached psa_iterations.csv"
    return out, p


def build_ceac(sweep: dict, strata_pos: dict, p_values: np.ndarray) -> pd.DataFrame:
    """ceac.csv -- conventional CEAC over the contract WTP grid, per stratum.

    P(ciNPWT cost-effective) = fraction of draws with INMB(wtp) > 0; P(standard) is its
    complement, so the two sum to 1 exactly. Both are asserted in [0,1] and summing to 1
    at every WTP, live, as the contract requires.
    """
    rows = []
    for name, pos in sorted(strata_pos.items(), key=lambda kv: p_values[kv[1]]):
        ic = sweep["inc_cost"][pos]
        iq = sweep["inc_qaly"][pos]
        for w in CEAC_GRID:
            inmb = iq * w - ic
            p_ci = float(np.mean(inmb > 0.0))
            p_std = 1.0 - p_ci
            assert -EPS <= p_ci <= 1.0 + EPS, f"P(ciNPWT CE) out of [0,1]: {p_ci}"
            assert -EPS <= p_std <= 1.0 + EPS, f"P(standard CE) out of [0,1]: {p_std}"
            assert abs(p_ci + p_std - 1.0) < EPS, "CEAC probabilities do not sum to 1"
            rows.append({"stratum": name, "p": float(p_values[pos]), "wtp": float(w),
                         "p_ciNPWT_ce": p_ci, "p_standard_ce": p_std})
    return pd.DataFrame(rows)


def build_summary_csv(sweep: dict, strata_pos: dict, p_values: np.ndarray,
                      n_iter: int) -> pd.DataFrame:
    """psa_summary.csv -- one row per stratum.

    ICERs are NOT averaged: `icer` is NaN wherever ciNPWT is dominant, which is most of
    the range. The summary is built on INMB and on the CE-plane / dominance frequencies,
    with `n_icer_defined` reported so the reader can see how few draws even have one.
    """
    rows = []
    for name, pos in sorted(strata_pos.items(), key=lambda kv: p_values[kv[1]]):
        ic = sweep["inc_cost"][pos]
        iq = sweep["inc_qaly"][pos]
        dom = classify_dominance(ic, iq)
        r = {"stratum": name, "p": float(p_values[pos]), "n_iter": int(n_iter),
             "horizon_cycles": HORIZON}
        for tag, arr in (("inc_cost", ic), ("inc_qaly", iq)):
            r[f"{tag}_mean"] = float(np.mean(arr))
            r[f"{tag}_median"] = float(np.median(arr))
            r[f"{tag}_p2_5"] = float(np.percentile(arr, 2.5))
            r[f"{tag}_p97_5"] = float(np.percentile(arr, 97.5))
        for w in WTP_REPORTED:
            tag = f"{int(w / 1000)}k"
            inmb = iq * w - ic
            r[f"inmb_{tag}_mean"] = float(np.mean(inmb))
            r[f"inmb_{tag}_median"] = float(np.median(inmb))
            r[f"inmb_{tag}_p2_5"] = float(np.percentile(inmb, 2.5))
            r[f"inmb_{tag}_p97_5"] = float(np.percentile(inmb, 97.5))
            r[f"p_ce_{tag}"] = float(np.mean(inmb > 0.0))
        r["p_dominant"] = float(np.mean(dom == "ciNPWT dominant"))
        r["p_dominated"] = float(np.mean(dom == "ciNPWT dominated"))
        r["p_more_costly_more_effective"] = float(
            np.mean(dom == "ciNPWT more costly & more effective"))
        r["p_less_costly_less_effective"] = float(
            np.mean(dom == "ciNPWT less costly & less effective"))
        r["p_cost_saving"] = float(np.mean(ic < 0.0))
        r["p_more_effective"] = float(np.mean(iq > 0.0))
        r["n_icer_defined"] = int(np.sum(
            (dom != "ciNPWT dominant") & (dom != "ciNPWT dominated") & (dom != "equal")))
        rows.append(r)
    return pd.DataFrame(rows)


def build_acceptability(sweep: dict, p_values: np.ndarray, n_iter: int) -> pd.DataFrame:
    """acceptability_by_risk.csv -- THE NOVEL OUTPUT (Fig 4).

    P(ciNPWT cost-effective) as a function of the baseline 90-day SSI risk `p`, at each
    reported WTP. Because ONE sample set is reused at every risk point and INMB is
    strictly increasing in `p` for a fixed draw, the curve must be monotone
    non-decreasing -- asserted live, not assumed.
    """
    rows = []
    for i, p in enumerate(p_values):
        ic = sweep["inc_cost"][i]
        iq = sweep["inc_qaly"][i]
        row = {"p": float(p)}
        for w in WTP_REPORTED:
            row[f"p_ce_{int(w / 1000)}k"] = float(np.mean(iq * w - ic > 0.0))
        row["n_iter"] = int(n_iter)
        rows.append(row)
    df = pd.DataFrame(rows)[["p", "p_ce_50k", "p_ce_100k", "p_ce_150k", "n_iter"]]
    for c in ("p_ce_50k", "p_ce_100k", "p_ce_150k"):
        d = np.diff(df[c].to_numpy())
        assert d.min() >= -1e-12, (
            f"acceptability_by_risk is NOT monotone increasing in p for {c}: worst step "
            f"{d.min():.3e} at p = {df['p'].to_numpy()[int(np.argmin(d)) + 1]:.4f}. With "
            "one reused sample set this can only happen if some draw has RR >= 1; "
            "investigate before shipping the figure."
        )
    return df


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (numpy only; ties averaged)."""
    def _rank(a):
        order = np.argsort(a, kind="mergesort")
        r = np.empty(len(a), dtype=float)
        r[order] = np.arange(1, len(a) + 1, dtype=float)
        # average ties
        s = a[order]
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and s[j + 1] == s[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = (i + j + 2) / 2.0
            i = j + 1
        return r
    rx, ry = _rank(np.asarray(x, float)), _rank(np.asarray(y, float))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else 0.0


def inmb_drivers(draws: pd.DataFrame, sweep: dict, strata_pos: dict,
                 p_values: np.ndarray) -> pd.DataFrame:
    """Spearman rank correlation of every sampled input with INMB at $100k, per stratum.

    Costs nothing (it reads the stored draws and incrementals) and answers the question
    the cost-CV diagnostic raises: given that the PSA's cost spread is modest and
    `m_ssi_cost` is absent, WHAT is actually driving the width of the CEAC? Also tells
    `run_dsa.py` and the manuscript whether the probabilistic and deterministic rankings
    agree.
    """
    rows = []
    for name in draws.columns:
        r = {"parameter": name}
        for sname, pos in sorted(strata_pos.items(), key=lambda kv: p_values[kv[1]]):
            inmb = sweep["inc_qaly"][pos] * WTP - sweep["inc_cost"][pos]
            r[f"rho_{sname}"] = _spearman(draws[name].to_numpy(), inmb)
        rows.append(r)
    df = pd.DataFrame(rows)
    return df.reindex(df["rho_low"].abs().sort_values(ascending=False).index)


def device_fixed_counterfactual(samples: list[dict], p_value: float,
                                stratum: str) -> dict:
    """Re-run ONE stratum with `c_device_cinpwt` HELD AT ITS BASE VALUE in every draw.

    Why this exists. The ciNPWT unit cost is represented TWICE in this build: it is
    sampled here, and it is also analysis axis 2 (`SETTINGS['price_grid']`, $100-$900),
    reported per stratum as a threshold price by `run_threshold.py`. Sampling it therefore
    propagates the same uncertainty a second time -- and at the low stratum it is the
    dominant driver of CEAC width (Spearman rho = -0.63), so P(cost-effective) there is
    substantially a statement about a `derived_assumption` with no source. Truncating the
    draws to the price axis does not fix that; DISCLOSING it does.

    This counterfactual is that disclosure made quantitative: every other parameter keeps
    its draw, only the device price is pinned at the base value, so the difference between
    the two P(CE) numbers is exactly the device-price assumption's contribution.

    ~10,000 extra `run_pair` calls (about 3 s at one stratum).
    """
    dev_base = float(I.cost.c_device_cinpwt.value)
    n = len(samples)
    ic = np.empty(n, dtype=float)
    iq = np.empty(n, dtype=float)
    for j, s in enumerate(samples):
        s2 = dict(s)
        s2["c_device_cinpwt"] = dev_base
        res_a, res_b = model.run_pair(p_base=float(p_value), horizon_cycles=HORIZON,
                                      disc_cost=DISC_COST, disc_qaly=DISC_QALY,
                                      half_cycle=HALF_CYCLE, sample=s2)
        ic[j] = res_b["cost"] - res_a["cost"]
        iq[j] = res_b["qaly"] - res_a["qaly"]
    out = {"device_fixed_at": dev_base, "p": float(p_value), "n_iter": n,
           "stratum": stratum}
    for w in WTP_REPORTED:
        tag = f"{int(w / 1000)}k"
        inmb = iq * w - ic
        out[f"p_ce_{tag}"] = float(np.mean(inmb > 0.0))
        out[f"inmb_{tag}_mean"] = float(np.mean(inmb))
        out[f"inmb_{tag}_p2_5"] = float(np.percentile(inmb, 2.5))
        out[f"inmb_{tag}_p97_5"] = float(np.percentile(inmb, 97.5))
    return out


def psa_vs_deterministic(sm: dict, sweep: dict, strata_pos: dict,
                         p_values: np.ndarray) -> dict:
    """Reconcile the PSA mean INMB with the deterministic base case, per stratum.

    THE PSA MEANS DO **NOT** REPRODUCE THE DETERMINISTIC BASE CASE, and it is not
    Monte-Carlo noise: the gap is the same sign and roughly the same relative size at all
    three strata (about -1%), and at the intermediate stratum it is ~3.3 Monte-Carlo
    standard errors. Two mechanisms, both properties of the mandated fit rather than of
    the model:

      1. **The dominant term.** BUILD-CONTRACT section 9 / the PSA convention require the
         relative effect to be sampled on the log scale preserving the MEDIAN, so
         `rr_cinpwt` draws have median 0.61 but MEAN 0.61 x exp(sdlog^2 / 2) = 0.6138.
         The deterministic run uses 0.61. A higher mean RR means fewer SSIs averted, so
         the mean INMB across draws sits below the INMB at the deterministic RR.
      2. **A small second term.** Rejection-truncating `c_device_cinpwt` to its $100-$900
         axis removes more mass below $100 than above $900, so the mean
         sampled device price sits slightly above the deterministic $250.

    Both are quantified here by re-running the engine deterministically at the MEAN of the
    draws, so the reconciliation is measured rather than argued.
    """
    rr_par = I.tp.rr_cinpwt
    _, sdlog = _lognormal_params(rr_par)
    rr = sm["draws"]["rr_cinpwt"].to_numpy()
    dev = sm["draws"]["c_device_cinpwt"].to_numpy()
    rr_mean, dev_mean = float(np.mean(rr)), float(np.mean(dev))

    def _inmb(p, sample=None):
        a, b = model.run_pair(p_base=float(p), horizon_cycles=HORIZON,
                              disc_cost=DISC_COST, disc_qaly=DISC_QALY,
                              half_cycle=HALF_CYCLE, sample=sample)
        return float(model.incremental(b, a, wtp=WTP)["inmb"])

    rows = []
    for name, pos in sorted(strata_pos.items(), key=lambda kv: p_values[kv[1]]):
        p = float(p_values[pos])
        inmb = sweep["inc_qaly"][pos] * WTP - sweep["inc_cost"][pos]
        mean = float(np.mean(inmb))
        mcse = float(np.std(inmb, ddof=1) / math.sqrt(len(inmb)))
        det = _inmb(p)
        pred_rr = _inmb(p, {"rr_cinpwt": rr_mean})
        pred_both = _inmb(p, {"rr_cinpwt": rr_mean, "c_device_cinpwt": dev_mean})
        rows.append({
            "stratum": name, "p": p, "deterministic": det, "psa_mean": mean,
            "gap": mean - det, "mcse": mcse,
            "gap_over_mcse": (mean - det) / mcse if mcse > 0 else float("nan"),
            "gap_pct": (mean - det) / det * 100.0 if det != 0 else float("nan"),
            "pred_rr_only": pred_rr, "pred_rr_and_device": pred_both,
            "residual": mean - pred_both,
        })
    return {
        "rows": rows,
        "rr_value": float(rr_par.value),
        "rr_sdlog": float(sdlog),
        "rr_mean_theory": float(rr_par.value) * math.exp(sdlog ** 2 / 2.0),
        "rr_mean_drawn": rr_mean,
        "rr_median_drawn": float(np.median(rr)),
        "device_value": float(I.cost.c_device_cinpwt.value),
        "device_mean_drawn": dev_mean,
    }


def _crossing(df: pd.DataFrame, col: str, level: float):
    """First `p` at which an acceptability curve reaches `level`, linearly interpolated
    between subgrid points. None if it never does."""
    p = df["p"].to_numpy()
    y = df[col].to_numpy()
    hit = np.where(y >= level)[0]
    if len(hit) == 0:
        return None
    k = int(hit[0])
    if k == 0 or y[k] == y[k - 1]:
        return float(p[k])
    return float(p[k - 1] + (level - y[k - 1]) * (p[k] - p[k - 1]) / (y[k] - y[k - 1]))


# --------------------------------------------------------------------------- #
# 8. I/O                                                                       #
# --------------------------------------------------------------------------- #
def write_results_csv(df: pd.DataFrame, filename: str) -> str:
    """Write a results CSV with `inputs.RESULTS_BANNER` as a '#'-prefixed line 1, so
    `pd.read_csv(path, skiprows=1)` works (BUILD-CONTRACT section 7)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", newline="") as fh:
        fh.write("# " + I.RESULTS_BANNER + "\n")
        df.to_csv(fh, index=False)
    return path


def _md(text) -> str:
    """Escape a free-text string for a markdown table cell. An unescaped '|' (e.g. in
    'P(cost-effective | p)') silently splits the row into an extra column."""
    return str(text).replace("|", r"\|")


def _money(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return ("-" if x < 0 else "") + f"${abs(x):,.0f}"


def _money2(x) -> str:
    """Money to the cent -- used where the reader has to match a difference of a few
    dollars against another table (the PSA-vs-deterministic reconciliation)."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return ("-" if x < 0 else "") + f"${abs(x):,.2f}"


def build_summary_md(sm, sweep, summary, ceac, accept, cv, drivers, strata_pos, p_values,
                     subgrid_idx, engine_read_probe, recon, dev_fixed) -> str:
    """psa_summary.md -- results banner first, then results, the three sampling-rule
    disclosures, the realized cost-CV diagnostic, and the exclusion log."""
    L: list[str] = []
    A = L.append

    A("# Probabilistic sensitivity analysis -- ciNPWT after instrumented posterior "
      "lumbar fusion")
    A("")
    A("> **" + I.RESULTS_BANNER + "**")
    A("")
    A("*Generated by `run_psa.py` from the read-only engine (`inputs.py` / `model.py`). "
      "Model-based result; not a substitute for clinical evidence.*")
    A("")

    # ---------------- what was run ---------------- #
    A("## What was run")
    A("")
    A(f"- **{sm['n_iter']:,} Monte-Carlo iterations**, `np.random.default_rng({sm['seed']})`.")
    A(f"- Horizon {HORIZON} cycles (2 years); {DISC_COST:.0%}/yr discounting on costs and "
      f"QALYs; trapezoidal half-cycle correction on dwell streams only; "
      f"{I.PRICE_YEAR} USD.")
    A(f"- **{len(sm['sampled_params'])} parameters sampled**; "
      f"{len(sm['excluded'])} excluded, every one of them logged below.")
    A(f"- Distribution fitting: `[low, high]` treated as an approximate 95% CI, "
      f"`SE = (high - low) / {Z95}`, **method of moments preserving the mean** -- except "
      "relative effects, sampled on the log scale preserving the **median** "
      "(`sdlog = (ln(high) - ln(low)) / 3.92`), so RR stays below 1 in every draw and "
      "arm ordering is preserved.")
    A("- **Correlation.** The engine applies `p_ciNPWT = p_control x RR` internally, so "
      "the two arms are correlated by construction; the RR is never pre-multiplied here.")
    A(f"- **One sample set of {sm['n_iter']:,} draws is reused at every risk point** "
      "(common random numbers). This is deliberate: for a fixed draw INMB is strictly "
      "increasing in `p`, so reusing the draws is what makes the acceptability curve "
      "monotone rather than jittery, and it keeps the sweep to a single pass.")
    A(f"- Acceptability-by-risk subgrid: **every {RISK_SUBGRID_STEP}th point** of the "
      f"291-point `SETTINGS['risk_grid']` **plus the three reported strata** (0.066 and "
      f"0.114 are not multiples of {RISK_SUBGRID_STEP} on that grid) = "
      f"**{len(subgrid_idx)} risk points**, p = {p_values[0]:.4f} to {p_values[-1]:.4f}, "
      f"i.e. {len(subgrid_idx) * sm['n_iter']:,} `run_pair` calls. (Wall-clock runtime is "
      "printed to stdout and recorded in `run_log.d/run_psa.md`, deliberately **not** "
      "written here: it is machine-dependent, and every file in `v2/results/` must be "
      "byte-reproducible from the seed.)")
    A("- Cost draws are expressed in each parameter's **own stored price year** (the six "
      "commercial SSI tranches in 2019 USD, `c_device_cinpwt` in 2025 USD); the engine "
      "inflates exactly once. No cost is read with `val()`.")
    A("")

    # ---------------- headline ---------------- #
    A("## Headline results, by risk stratum")
    A("")
    A("ciNPWT minus standard dressing, 2-year horizon. **`icer` is NaN wherever ciNPWT is "
      "dominant**, which is most of the range at and above the intermediate stratum, so "
      "nothing here averages an ICER: the summary is built on INMB and on the "
      "cost-effectiveness-plane frequencies.")
    A("")
    A("| stratum | p | inc. cost, mean (95% CrI) | inc. QALY, mean (95% CrI) | "
      "INMB @ $100k, mean (95% CrI) |")
    A("|---|---|---|---|---|")
    for _, r in summary.iterrows():
        A(f"| {r['stratum']} | {r['p']:.3f} | "
          f"{_money(r['inc_cost_mean'])} ({_money(r['inc_cost_p2_5'])} to "
          f"{_money(r['inc_cost_p97_5'])}) | "
          f"{r['inc_qaly_mean']:+.5f} ({r['inc_qaly_p2_5']:+.5f} to "
          f"{r['inc_qaly_p97_5']:+.5f}) | "
          f"{_money(r['inmb_100k_mean'])} ({_money(r['inmb_100k_p2_5'])} to "
          f"{_money(r['inmb_100k_p97_5'])}) |")
    A("")
    A("Probabilities are printed to **4 decimal places**. That is not fussiness: at the "
      "intermediate stratum P(CE) is 0.9999, and rounding it to `1.000` would assert a "
      "certainty the PSA does not have in the same row that reports a draw which is *not* "
      "dominant. No probability in this table is exactly 0 or exactly 1.")
    A("")
    A("| stratum | P(CE) @ $50k | P(CE) @ $100k | P(CE) @ $150k | P(dominant) | "
      "P(dominated) | P(cost-saving) | P(more effective) | draws with a defined ICER |")
    A("|---|---|---|---|---|---|---|---|---|")
    for _, r in summary.iterrows():
        A(f"| {r['stratum']} | {r['p_ce_50k']:.4f} | {r['p_ce_100k']:.4f} | "
          f"{r['p_ce_150k']:.4f} | {r['p_dominant']:.4f} | {r['p_dominated']:.4f} | "
          f"{r['p_cost_saving']:.4f} | {r['p_more_effective']:.4f} | "
          f"{int(r['n_icer_defined']):,} / {sm['n_iter']:,} |")
    A("")

    # ---------------- PSA means vs the deterministic base case ---------------- #
    A("### The PSA means do NOT reproduce the deterministic base case -- and that is a "
      "property of the fit, not Monte-Carlo noise")
    A("")
    _g = {r["stratum"]: r for r in recon["rows"]}
    _mid = _g.get("intermediate", recon["rows"][0])
    _lo_r = _g.get("low", recon["rows"][0])
    A(f"A reviewer holding Table 2 beside this table will find the two INMB columns "
      f"disagree, in the **same direction at every stratum**: the PSA mean sits "
      f"{abs(_mid['gap_pct']):.2f}% below the deterministic value at the intermediate "
      f"stratum ({_money2(_mid['psa_mean'])} against "
      f"{_money2(_mid['deterministic'])}), and "
      f"{abs(_lo_r['gap_pct']):.2f}% below it at the low stratum, where the same handful "
      f"of dollars is a much larger share of a much smaller INMB. The gap is "
      f"{abs(_mid['gap_over_mcse']):.1f} Monte-Carlo standard errors at the intermediate "
      f"stratum, so it is **not** noise. The two are different estimands, and the "
      f"difference is explained here rather than left for a reader to find:")
    A("")
    A("| stratum | p | deterministic INMB @ $100k | PSA mean INMB | gap | MCSE | "
      "gap / MCSE | gap % |")
    A("|---|---|---|---|---|---|---|---|")
    for r in recon["rows"]:
        A(f"| {r['stratum']} | {r['p']:.3f} | {_money2(r['deterministic'])} | "
          f"{_money2(r['psa_mean'])} | {_money2(r['gap'])} | {_money2(r['mcse'])} | "
          f"{r['gap_over_mcse']:+.2f} | {r['gap_pct']:+.3f}% |")
    A("")
    A(f"**The dominant mechanism is the mandated median-preserving lognormal on "
      f"`rr_cinpwt`.** BUILD-CONTRACT section 9 and the PSA convention require relative "
      f"effects to be sampled on the log scale preserving the **median**, so the drawn RR "
      f"has median {recon['rr_median_drawn']:.6f} (theoretical median exactly "
      f"{recon['rr_value']:.2f}) but **mean "
      f"{recon['rr_mean_drawn']:.6f}** -- theory "
      f"{recon['rr_value']:.2f} x exp({recon['rr_sdlog']:.5f}^2 / 2) = "
      f"{recon['rr_mean_theory']:.6f}. The deterministic base case uses "
      f"{recon['rr_value']:.2f}. A **higher mean RR means fewer SSIs averted**, so the "
      "mean INMB across draws must sit below the INMB evaluated at the deterministic RR. "
      "The two numbers answer different questions: the deterministic value is INMB at the "
      "median effect, the PSA value is the mean INMB over the effect distribution. "
      "A mean-preserving lognormal would close the gap and was deliberately **not** used "
      "-- it would put the median draw off the base case and let RR draws exceed 1.")
    A("")
    A(f"**A second, smaller term** is the device-price truncation (below): rejection-"
      f"truncating `c_device_cinpwt` to its $100-$900 axis removes more mass below $100 "
      f"than above $900, so the mean sampled price is "
      f"{_money2(recon['device_mean_drawn'])} against the deterministic "
      f"{_money2(recon['device_value'])} -- a fixed shift of about "
      f"{_money2(recon['device_mean_drawn'] - recon['device_value'])} per patient at "
      f"every stratum, which is why it dominates the *relative* gap at the low stratum "
      f"and barely moves it at the high one.")
    A("")
    A("Both are measured, not asserted -- the engine was re-run deterministically at the "
      "**mean of the draws** and reproduces the PSA means to within a few dollars:")
    A("")
    A("| stratum | deterministic | + drawn mean RR | + drawn mean device price | "
      "PSA mean | residual |")
    A("|---|---|---|---|---|---|")
    for r in recon["rows"]:
        A(f"| {r['stratum']} | {_money2(r['deterministic'])} | "
          f"{_money2(r['pred_rr_only'])} | {_money2(r['pred_rr_and_device'])} | "
          f"{_money2(r['psa_mean'])} | {_money2(r['residual'])} |")
    A("")
    A("The residual -- what is left after the two mean shifts are accounted for -- is a "
      "few dollars at every stratum, which is Monte-Carlo noise on the remaining "
      "parameters and nonlinearity. **Do not \"fix\" the gap by re-centring the sampler**: "
      "the median-preserving fit is contract-mandated and is what keeps every RR draw "
      "below 1. Report the deterministic value as the base case and the PSA mean as the "
      "mean over the parameter distribution, and say that they differ for this reason.")
    A("")

    # ---------------- acceptability by risk ---------------- #
    A("## Acceptability by risk -- the novel output (Fig 4)")
    A("")
    A("`acceptability_by_risk.csv` reports P(ciNPWT cost-effective) as a function of the "
      "baseline 90-day all-SSI risk `p`, conditioning on `p` rather than integrating "
      "over it. The curve is **monotone non-decreasing in `p`** (asserted live at every "
      "WTP, not assumed).")
    A("")
    A("| WTP | p at P(CE) = 0.50 | p at P(CE) = 0.80 | p at P(CE) = 0.95 | "
      "P(CE) at p = 0.005 | P(CE) at p = 0.150 |")
    A("|---|---|---|---|---|---|")
    for w in WTP_REPORTED:
        c = f"p_ce_{int(w / 1000)}k"
        def _f(level):
            x = _crossing(accept, c, level)
            return "never" if x is None else f"{x * 100:.2f}%"
        A(f"| ${w:,.0f}/QALY | {_f(0.50)} | {_f(0.80)} | {_f(0.95)} | "
          f"{accept[c].iloc[0]:.3f} | {accept[c].iloc[-1]:.3f} |")
    A("")
    A("Read against the deterministic threshold (`run_threshold.py`), this is the "
      "probabilistic statement of the same result: the deterministic `p*` is where the "
      "*mean* INMB crosses zero, while the value above is where *half the draws* cross "
      "it. The gap between the 0.50 and 0.95 crossings is the width of the decision's "
      "uncertainty in the units the clinical question is actually asked in -- a "
      "patient's predicted risk.")
    A("")

    # ---------------- disclosures ---------------- #
    A("## The three sampling-rule disclosures (BUILD-CONTRACT section 9)")
    A("")
    A("### 1. `p_base_ssi` was never sampled -- it is the x-axis, not an uncertain input")
    A("")
    A("`p_base_ssi` carries `dist='beta'` and a (0.005, 0.15) range, so a generic "
      "\"sample every non-`fixed` Param\" loop would have drawn it. It is excluded **by "
      "name**. Every acceptability number above is a probability **conditional on** `p`; "
      "integrating over `p` would have collapsed Fig 4 into a single number and silently "
      "answered a different question.")
    A("")
    A("### 2. `du_sup` / `du_deep` were truncated at 0 by RESAMPLING (not clipping)")
    A("")
    A("Both are `normal` on a quantity that must be non-negative -- a negative draw is a "
      "utility *gain* from surgical-site infection. A normal fitted to "
      "(0.08, 0.05, 0.11) has SE 0.0153, so 0 is 5.2 SD away and about 1e-7 of draws go "
      "negative. Negative draws were **rejected and redrawn**, not clipped to 0: "
      "clipping would pile them onto exactly 0 and create a spurious point mass at "
      "\"SSI costs no utility at all\".")
    A("")
    for name in TRUNCATE_AT_ZERO:
        p = I.param_by_name(name)
        se = _se(p)
        z = float(p.value) / se
        p_neg = 0.5 * math.erfc(z / math.sqrt(2.0))
        rej = sm["n_rejected"].get(name, 0)
        A(f"- `{name}` (mean {float(p.value):g}, SE {se:.5g}, so 0 is {z:.2f} SD away, "
          f"P(negative draw) = {p_neg:.2e}, expected {p_neg * sm['n_iter']:.4f} negative "
          f"draws in {sm['n_iter']:,}): **{rej} rejected and redrawn** at this seed. "
          + ("The guard is active but did not fire -- which is the expected outcome at "
             "this SE, not evidence that it is unnecessary: it is what keeps a future "
             "A2-revised (wider) decrement from silently producing a utility gain."
             if rej == 0 else ""))
    A("")
    A("**The same idiom is applied to `c_device_cinpwt`**, truncated to its declared "
      "$100-$900 range rather than at a single bound -- there the guard fires hard "
      f"({sm['n_rejected'].get('c_device_cinpwt', 0):,} draws rejected and redrawn). See "
      "the fourth disclosure below; it is a decision about the price *axis*, not about "
      "non-negativity, so it is reported there rather than here.")
    A("")
    A("### 3. `m_ssi_cost` stayed `fixed` -- the CEAC does NOT carry attribution uncertainty")
    A("")
    A("**State this plainly wherever the CEAC is shown.** SSI-attributable-cost "
      "*attribution* is structural between-source disagreement -- Pennington's matched "
      "increment against Edmiston's regression-adjusted all-cause increment, a "
      f"{cv['m_ssi_range'][2] / cv['m_ssi_range'][0]:.1f}-fold span "
      f"({cv['m_ssi_range'][0]:.3f} to {cv['m_ssi_range'][2]:.3f} on `m_ssi_cost`) -- and "
      "the contract handles it **deterministically**, through the all-cause x1.000 "
      "scenario and through the DSA, where it must be the top tornado bar. It is "
      "therefore **not sampled here**. The consequence is that **the CEAC, the "
      "acceptability-by-risk curve and Fig 4 are more confident than the evidence "
      "supports**: they propagate sampling noise on the six cost tranches while the "
      "single largest cost uncertainty in the model is absent. A reader who sees the "
      "tornado first will notice the mismatch; do not leave them to infer that the PSA "
      "covers it.")
    A("")

    # ---------------- cost CV diagnostic ---------------- #
    A("## Realized cost-uncertainty diagnostic")
    A("")
    A("The six SSI cost tranches are sampled as **independent** gammas, so their weighted "
      "sum has a smaller coefficient of variation than any single tranche -- "
      "independence shrinks aggregate variance. This is how much cost uncertainty the "
      "PSA actually carries:")
    A("")
    A(f"- **Realized CV of the total (weighted 24-month) cost per SSI: "
      f"{cv['cv_weighted']:.4f}** "
      f"(mean {_money(cv['mean_weighted'])}, SD {_money(cv['sd_weighted'])}, "
      f"95% CrI {_money(cv['p2_5_weighted'])} to {_money(cv['p97_5_weighted'])}, "
      f"{I.PRICE_YEAR} USD).")
    A(f"- Mean CV of a **single** tranche: {cv['mean_tranche_cv']:.4f}. Summed within "
      f"type: deep total {cv['cv_deep_total']:.4f}, superficial total "
      f"{cv['cv_sup_total']:.4f} -- the shrinkage from independence.")
    A(f"- The Dirichlet split adds some of it back: CV of `frac_deep` = "
      f"{cv['cv_frac_deep']:.4f}, and deep and superficial SSI differ by roughly "
      f"$61k per case, so the split is a material second source of cost spread.")
    A(f"- **For comparison, the uncertainty the PSA does NOT carry:** read on the same "
      f"method-of-moments footing, `m_ssi_cost`'s DSA range implies a CV of "
      f"**{cv['implied_cv_m_ssi_cost']:.4f}** on the *whole* cost block -- comparable to "
      f"or larger than the {cv['cv_weighted']:.4f} the PSA does carry. Roughly half the "
      f"defensible cost variance is outside the CEAC by design.")
    A("")
    A("### What actually drives the spread")
    A("")
    A("Given that the PSA's cost uncertainty is modest and `m_ssi_cost` is absent, it "
      "matters which input the CEAC width is coming from. Spearman rank correlation of "
      "each sampled input with INMB at $100,000/QALY:")
    A("")
    A("| parameter | rho (low, p=0.015) | rho (intermediate, p=0.066) | rho (high, p=0.114) |")
    A("|---|---|---|---|")
    for _, r in drivers.iterrows():
        A(f"| `{r['parameter']}` | {r['rho_low']:+.3f} | {r['rho_intermediate']:+.3f} | "
          f"{r['rho_high']:+.3f} |")
    A("")

    # ---------------- fourth disclosure: the device price ---------------- #
    A("### A fourth disclosure, not mandated but material: the ciNPWT device price IS sampled")
    A("")
    dev = I.cost.c_device_cinpwt
    dv = sm["draws"]["c_device_cinpwt"]
    A(f"`c_device_cinpwt` is an **axis** elsewhere in this build (swept $100-$900 across "
      f"the 161-point `SETTINGS['price_grid']` in `run_threshold.py` and Fig 3), and it "
      f"is also `dist='gamma'` with a `derived_assumption` range of "
      f"${float(dev.low):,.0f}-${float(dev.high):,.0f}. The contract's binding exclusions "
      f"name only `p_base_ssi` and `m_ssi_cost`, and `run_log.d/inputs.md` objection 2 "
      f"asks that the device price be excluded *\"when [it is] being swept\"* -- which it "
      f"is not here, since the PSA sweeps `p` only. **It is therefore sampled -- "
      f"truncated to the price axis -- and double-reported**, and the reader must know "
      f"three things about that choice:")
    A("")
    td = sm["trunc_diag"]["c_device_cinpwt"]
    A(f"1. The $100-$900 span is an **assumed** spread on a parameter no source reports, "
      f"not a measured CI. Read as a 95% CI it gives SE ${_se(dev):,.0f}, i.e. a CV of "
      f"{_se(dev) / float(dev.value):.2f} -- by far the widest relative spread in the "
      f"sampled set, and wider than any evidence-derived parameter in the model.")
    A(f"2. **The fitted gamma ignored the price axis, so the draws are TRUNCATED to "
      f"[${float(dev.low):,.0f}, ${float(dev.high):,.0f}] by rejection resampling** "
      f"(orchestrator decision 2026-08-09; the same idiom used for `du_sup` / `du_deep`, "
      f"resampled rather than clipped so no point mass is created at either bound). "
      f"Untruncated, the method-of-moments gamma is strongly right-skewed and does not "
      f"respect the axis at all: median ${td['median_untruncated']:,.0f} against a mean "
      f"of ${td['mean_untruncated']:,.0f}, realized range "
      f"${td['min_untruncated']:,.2f} to ${td['max_untruncated']:,.2f}, with "
      f"**{td['n_below_untruncated'] / td['n']:.2%} of draws below the $100 floor and "
      f"{td['n_above_untruncated'] / td['n']:.2%} above the $900 ceiling** of "
      f"`SETTINGS['price_grid']`. A ${td['min_untruncated']:,.0f} and a "
      f"${td['max_untruncated']:,.0f} ciNPWT unit are outside the range this paper itself "
      f"records as plausible -- wrong in both directions -- so they are not shipped. "
      f"{sm['n_rejected'].get('c_device_cinpwt', 0):,} draws were rejected and redrawn; "
      f"the shipped column is asserted inside [${float(dev.low):,.0f}, "
      f"${float(dev.high):,.0f}] before anything is written.")
    A(f"3. **The device price is represented TWICE in this build, and truncation does not "
      f"fix that -- disclosure does.** It is sampled here *and* swept as analysis axis 2, "
      f"where `run_threshold.py` reports a threshold price per stratum. So the CEAC "
      f"propagates the same uncertainty a second time. At the **low stratum it is the "
      f"dominant driver of CEAC width** (Spearman rho = "
      f"{float(drivers[drivers['parameter'] == 'c_device_cinpwt'].iloc[0]['rho_low']):+.3f}, "
      f"against "
      f"{float(drivers[drivers['parameter'] == 'c_device_cinpwt'].iloc[0]['rho_high']):+.3f} "
      f"at the high stratum), which means P(cost-effective) at low risk is substantially a "
      f"statement about a `derived_assumption` with no source rather than about clinical "
      f"evidence. **Both numbers, so a reader can see exactly how much of the "
      f"low-stratum width is device-price assumption:**")
    A("")
    dlow = summary[summary["stratum"] == dev_fixed["stratum"]].iloc[0]
    A(f"| P(CE) @ $100k, {dev_fixed['stratum']} stratum (p = {dev_fixed['p']:.3f}) | "
      "value | mean INMB @ $100k (95% CrI) |")
    A("|---|---|---|")
    A(f"| **device price SAMPLED** (truncated $100-$900) | "
      f"**{float(dlow['p_ce_100k']):.4f}** | {_money(dlow['inmb_100k_mean'])} "
      f"({_money(dlow['inmb_100k_p2_5'])} to {_money(dlow['inmb_100k_p97_5'])}) |")
    A(f"| **device price HELD FIXED at ${dev_fixed['device_fixed_at']:,.0f}** "
      f"(every other parameter still sampled) | "
      f"**{float(dev_fixed['p_ce_100k']):.4f}** | "
      f"{_money(dev_fixed['inmb_100k_mean'])} "
      f"({_money(dev_fixed['inmb_100k_p2_5'])} to "
      f"{_money(dev_fixed['inmb_100k_p97_5'])}) |")
    A("")
    A(f"   The difference -- {float(dlow['p_ce_100k']):.4f} against "
      f"{float(dev_fixed['p_ce_100k']):.4f} -- **is** the device-price assumption. "
      f"Nearly all of the low stratum's residual doubt about ciNPWT comes from not "
      f"knowing what the device costs, not from the clinical evidence. The "
      f"**deterministic price axis (Fig 3, and the threshold price per stratum) is the "
      f"honest presentation of that uncertainty**; the CEAC is the summary. Quote the "
      f"low-stratum P(CE) with the device-price caveat attached, or quote the fixed-price "
      f"figure and say the price is fixed.")
    A("")

    A("**Where the six SSI cost tranches' dispersion now comes from.** Until the A2 audit "
      "these ranges were an ASSUMED `value x 0.5` to `value x 1.5` span, on the stated "
      "premise that 'Edmiston reports no SDs or CIs at all'. That premise was false. "
      "Edmiston's Table 3 is titled **'Mean SSI Cost (95% CI)'** and publishes a 95% "
      "confidence interval for **all twelve** payments, so the assumed span is retired. "
      "BUILD-CONTRACT section 9 still binds this module to read `[low, high]` as an "
      "approximate 95% CI with `SE = (high - low)/3.92`; what changed is that the bounds "
      "are now Edmiston's own. For the four 0-6-month tranches the cumulative payment IS "
      "the marginal one, so the published CI transfers verbatim. For the other eight the "
      "marginal SE is recovered as `sqrt(SE_t^2 - SE_(t-1)^2)` under an "
      "independent-increments assumption -- **declared here as an approximation**, "
      "because the two nested cumulative estimates are positively correlated and the "
      "covariance is not recoverable from the published table; the conservative "
      "alternative `SE_1 + SE_2` is reported as a check. The realized per-tranche CVs "
      "above are far below the retired assumption's "
      f"{1 / Z95:.4f}, ranging from about 0.02 (commercial deep, n = 210,019) to about "
      "0.21 (Medicare superficial, n = 585), so the PSA is correspondingly tighter. Note "
      "what these intervals are and are not: they are Wald intervals on least-squares "
      "means from a log-link gamma GLM, i.e. SAMPLING precision on an adjusted mean, not "
      "the dispersion of cost across patients and not attribution uncertainty. "
      "Attribution uncertainty lives in `m_ssi_cost` and is handled deterministically.")
    A("")

    # ---------------- verification ---------------- #
    A("## Verification performed at run time")
    A("")
    A(f"- **Arm ordering: {sweep['n_inversions']} inversions** in "
      f"{len(subgrid_idx) * sm['n_iter']:,} draw x risk-point pairs "
      f"(`n_ssi_B < n_ssi_A` in every one; smallest gap "
      f"{sweep['min_ssi_gap']:.3e}). Guaranteed by construction: the engine applies "
      "`p_ciNPWT = p_control x RR` internally and the median-preserving lognormal keeps "
      f"every RR draw below 1 (max drawn RR = "
      f"{sm['draws']['rr_cinpwt'].max():.4f}).")
    A("- **CEAC**: both probabilities asserted in [0,1] and summing to 1 at every one of "
      f"the {len(CEAC_GRID)} WTP grid points, in each of the {len(strata_pos)} strata.")
    A("- **Acceptability by risk**: monotonicity in `p` asserted at all three WTPs.")
    for name in TRUNCATE_TO_RANGE:
        p = I.param_by_name(name)
        x = sm["draws"][name]
        A(f"- **Truncation of `{name}`**: every shipped draw asserted inside "
          f"[${float(p.low):,.0f}, ${float(p.high):,.0f}] before writing -- realized "
          f"${x.min():,.2f} to ${x.max():,.2f}, "
          f"{int((x < float(p.low)).sum())} below and "
          f"{int((x > float(p.high)).sum())} above.")
    A("- **Dominance labelling**: the vectorized classifier asserted identical to "
      f"`model.incremental`'s own strings on all {sm['n_iter']:,} intermediate-stratum "
      "draws.")
    A("- **Engine-read audit**: each candidate parameter was perturbed to its `high` "
      "bound one at a time and the engine's outputs watched, confirming that the "
      f"{len(engine_read_probe)} parameters declared engine-read are exactly the ones "
      "that move a result. Nothing inert was sampled into `psa_iterations.csv`.")
    A("")

    # ---------------- distributions ---------------- #
    A("## Distributions sampled")
    A("")
    A("| parameter | dist | value | low | high | SE | fitted | price year |")
    A("|---|---|---|---|---|---|---|---|")
    for _, r in sm["fitted"].iterrows():
        py = "--" if r["price_year"] is None or (isinstance(r["price_year"], float)
                                                 and not np.isfinite(r["price_year"])) \
            else f"{int(r['price_year'])} USD"
        A(f"| `{r['parameter']}` | {r['dist']} | {r['value']:g} | {r['low']:g} | "
          f"{r['high']:g} | {r['se']:.5g} | {_md(r['fitted'])} | {py} |")
    A("")

    # ---------------- exclusion log ---------------- #
    A("## Exclusion log -- every parameter NOT sampled, and why")
    A("")
    A("Required by BUILD-CONTRACT section 9 so the exclusions are **visible rather than "
      "hidden** inside a loop condition.")
    A("")
    A("| parameter | dist | category | reason |")
    A("|---|---|---|---|")
    for p, cat, reason in sm["excluded"]:
        A(f"| `{p.name}` | {p.dist} | {_md(cat)} | {_md(reason)} |")
    A("")
    A(f"**{len(sm['excluded'])} parameters excluded, "
      f"{len(sm['sampled_params'])} sampled, "
      f"{len(I.all_params())} in the store.**")
    A("")
    A("*Files written: `psa_iterations.csv`, `psa_summary.csv`, `ceac.csv`, "
      "`acceptability_by_risk.csv`, `psa_summary.md`. Read every CSV with "
      "`pd.read_csv(path, skiprows=1)` -- line 1 is the results banner.*")
    A("")

    text = "\n".join(L)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "psa_summary.md")
    with open(path, "w") as fh:
        fh.write(text)
    return path


# --------------------------------------------------------------------------- #
# 9. main                                                                      #
# --------------------------------------------------------------------------- #
def main():
    print("=" * 78)
    print(f"PROBABILISTIC SENSITIVITY ANALYSIS -- {N_ITER:,} iterations (seed {SEED})")
    print(I.RESULTS_BANNER)
    print(f"Horizon {HORIZON} cycles | discounting {DISC_COST:.0%}/yr | "
          f"WTP ${WTP:,.0f}/QALY primary")
    print("=" * 78)

    # ---- 1. sample ---------------------------------------------------------------- #
    sm = make_samples(N_ITER, SEED)
    print(f"\n[1/5] Sampled {len(sm['sampled_params'])} parameters x {N_ITER:,} draws.")
    print(f"      EXCLUDED BY NAME (contract section 9 rule 1): "
          f"{', '.join(AXIS_EXCLUSIONS)} -- the study's x-axis, never sampled.")
    for name in TRUNCATE_AT_ZERO:
        print(f"      TRUNCATED AT 0 BY RESAMPLING (rule 2): {name} -- "
              f"{sm['n_rejected'].get(name, 0)} negative draws rejected and redrawn.")
    for name in TRUNCATE_TO_RANGE:
        td = sm["trunc_diag"][name]
        print(f"      TRUNCATED TO [${td['lo']:,.0f}, ${td['hi']:,.0f}] BY RESAMPLING: "
              f"{name} -- the untruncated fit put "
              f"{td['n_below_untruncated'] / td['n']:.2%} of draws below the floor and "
              f"{td['n_above_untruncated'] / td['n']:.2%} above the ceiling "
              f"(range ${td['min_untruncated']:,.2f}-${td['max_untruncated']:,.2f}, "
              f"CV {td['cv_untruncated']:.4f}); {td['n_rejected']:,} rejected and redrawn.")
    print(f"      HELD FIXED (rule 3): m_ssi_cost -- the CEAC carries NO attribution "
          f"uncertainty.")
    n_fixed = sum(1 for _, c, _ in sm["excluded"] if c == "fixed")
    n_inert = sum(1 for _, c, _ in sm["excluded"] if c == "not read by the engine")
    print(f"      {len(sm['excluded'])} excluded in total "
          f"({len(AXIS_EXCLUSIONS)} axis, {n_fixed} fixed, {n_inert} not read by the "
          f"engine); full log in psa_summary.md.")
    assert "p_base_ssi" not in sm["draws"].columns, \
        "CONTRACT section 9 rule 1 VIOLATION: p_base_ssi was sampled"
    rr = sm["draws"]["rr_cinpwt"].to_numpy()
    assert rr.max() < 1.0, (
        f"a drawn RR reached {rr.max():.4f} >= 1: arm ordering would invert in that draw"
    )

    # ---- 2. engine-read audit ------------------------------------------------------- #
    probe = _probe_engine_read(float(STRATA[BASE_STRATUM]))
    declared = {p.name for p in sm["sampled_params"]}
    if probe != declared:
        raise AssertionError(
            "the declared sampled set does not match what the engine actually reads.\n"
            f"  sampled but inert : {sorted(declared - probe)}\n"
            f"  read but unsampled: {sorted(probe - declared)}"
        )
    print(f"[2/5] Engine-read audit: {len(probe)} parameters move a result when "
          f"perturbed; the sampled set matches exactly.")

    # ---- 3. sweep ------------------------------------------------------------------- #
    subgrid_idx, p_values, strata_pos = risk_subgrid()
    print(f"[3/5] Sweeping {len(p_values)} risk points x {N_ITER:,} draws "
          f"(every {RISK_SUBGRID_STEP}th grid point + the 3 strata; one reused sample "
          f"set)...")
    sweep = run_sweep(sm["samples"], p_values, strata_pos)
    print(f"      done in {sweep['seconds']:.0f} s. "
          f"Arm-order inversions: {sweep['n_inversions']} "
          f"(min n_ssi gap {sweep['min_ssi_gap']:.3e}).")
    assert sweep["n_inversions"] == 0, (
        f"{sweep['n_inversions']} arm-order inversions across "
        f"{len(p_values) * N_ITER:,} draws -- the engine's built-in correlation failed"
    )

    # ---- 4. deliverables ------------------------------------------------------------ #
    iters, p_base_iter = build_iterations(sm, sweep, strata_pos, p_values)
    summary = build_summary_csv(sweep, strata_pos, p_values, N_ITER)
    ceac = build_ceac(sweep, strata_pos, p_values)
    accept = build_acceptability(sweep, p_values, N_ITER)
    cv = cost_cv_diagnostic(sm["draws"])
    drivers = inmb_drivers(sm["draws"], sweep, strata_pos, p_values)

    # Reconcile the PSA means with the deterministic base case (9 extra run_pair calls),
    # and quantify how much of the LOW stratum's CEAC width is the device-price assumption
    # (one extra 10,000-draw pass at that stratum).
    recon = psa_vs_deterministic(sm, sweep, strata_pos, p_values)
    low_stratum = min(STRATA, key=lambda k: STRATA[k])
    t_dev = time.perf_counter()
    dev_fixed = device_fixed_counterfactual(
        sm["samples"], float(p_values[strata_pos[low_stratum]]), low_stratum)
    t_dev = time.perf_counter() - t_dev

    paths = [
        write_results_csv(iters, "psa_iterations.csv"),
        write_results_csv(summary, "psa_summary.csv"),
        write_results_csv(ceac, "ceac.csv"),
        write_results_csv(accept, "acceptability_by_risk.csv"),
    ]
    paths.append(build_summary_md(sm, sweep, summary, ceac, accept, cv, drivers,
                                  strata_pos, p_values, subgrid_idx, probe,
                                  recon, dev_fixed))
    print(f"[4/5] Deliverables built (iterations at the {BASE_STRATUM} stratum, "
          f"p = {p_base_iter:.3f}).")

    # ---- 5. console summary --------------------------------------------------------- #
    print("[5/5] Results (ciNPWT minus standard dressing, 2-year horizon):\n")
    for _, r in summary.iterrows():
        print(f"  [{r['stratum']}  p = {r['p']:.3f}]")
        print(f"    inc cost  mean {_money(r['inc_cost_mean']):>10s}   "
              f"95% CrI [{_money(r['inc_cost_p2_5'])}, {_money(r['inc_cost_p97_5'])}]")
        print(f"    inc QALY  mean {r['inc_qaly_mean']:>+10.5f}   "
              f"95% CrI [{r['inc_qaly_p2_5']:+.5f}, {r['inc_qaly_p97_5']:+.5f}]")
        print(f"    INMB@100k mean {_money(r['inmb_100k_mean']):>10s}   "
              f"95% CrI [{_money(r['inmb_100k_p2_5'])}, {_money(r['inmb_100k_p97_5'])}]")
        print(f"    P(CE) 50k/100k/150k = {r['p_ce_50k']:.4f} / {r['p_ce_100k']:.4f} / "
              f"{r['p_ce_150k']:.4f}   P(dominant) {r['p_dominant']:.4f}   "
              f"P(dominated) {r['p_dominated']:.4f}")
        print(f"    draws with a defined ICER: {int(r['n_icer_defined']):,} / {N_ITER:,} "
              f"(icer is NaN under dominance -- never averaged)\n")

    print("  ACCEPTABILITY BY RISK (Fig 4) -- P(cost-effective) vs baseline risk:")
    for w in WTP_REPORTED:
        c = f"p_ce_{int(w / 1000)}k"
        x50, x95 = _crossing(accept, c, 0.50), _crossing(accept, c, 0.95)
        print(f"    @${w:>7,.0f}/QALY: P(CE) rises {accept[c].iloc[0]:.3f} -> "
              f"{accept[c].iloc[-1]:.3f} over p = {p_values[0]:.3f}-{p_values[-1]:.3f}; "
              f"reaches 50% at p = "
              f"{'never' if x50 is None else f'{x50 * 100:.2f}%'}, "
              f"95% at p = {'never' if x95 is None else f'{x95 * 100:.2f}%'}")
    print(f"    monotone non-decreasing in p: asserted at all {len(WTP_REPORTED)} WTPs.")

    print(f"\n  REALIZED CV OF TOTAL PER-SSI COST ACROSS DRAWS: {cv['cv_weighted']:.4f}")
    print(f"    mean {_money(cv['mean_weighted'])} (95% CrI "
          f"{_money(cv['p2_5_weighted'])}-{_money(cv['p97_5_weighted'])}, "
          f"{I.PRICE_YEAR} USD); single-tranche CV {cv['mean_tranche_cv']:.4f}; "
          f"deep total {cv['cv_deep_total']:.4f}; sup total {cv['cv_sup_total']:.4f}")
    print(f"    m_ssi_cost is FIXED -- the attribution uncertainty the CEAC omits would "
          f"add a CV of {cv['implied_cv_m_ssi_cost']:.4f} on its own.")

    print("\n  WHAT DRIVES INMB SPREAD (Spearman rho with INMB@$100k), top 5 at the low "
          "stratum:")
    for _, r in drivers.head(5).iterrows():
        print(f"    {r['parameter']:<22s} low {r['rho_low']:+.3f}   "
              f"intermediate {r['rho_intermediate']:+.3f}   high {r['rho_high']:+.3f}")

    print("\n  PSA MEAN vs THE DETERMINISTIC BASE CASE (INMB @ $100k) -- a property of the "
          "fit, NOT Monte-Carlo noise:")
    for r in recon["rows"]:
        print(f"    {r['stratum']:<14s} det {r['deterministic']:>9,.2f}   PSA mean "
              f"{r['psa_mean']:>9,.2f}   gap {r['gap']:>8,.2f} "
              f"({r['gap_pct']:+.3f}%, {r['gap_over_mcse']:+.2f} MCSE)   "
              f"predicted {r['pred_rr_and_device']:>9,.2f}   "
              f"residual {r['residual']:>7,.2f}")
    print(f"    cause: the contract-mandated MEDIAN-preserving lognormal on rr_cinpwt has "
          f"mean {recon['rr_mean_drawn']:.6f} (theory "
          f"{recon['rr_value']:.2f} x exp({recon['rr_sdlog']:.5f}^2/2) = "
          f"{recon['rr_mean_theory']:.6f}) against the deterministic "
          f"{recon['rr_value']:.2f}; the truncated device price contributes a mean of "
          f"${recon['device_mean_drawn']:,.2f} against ${recon['device_value']:,.0f}.")

    low_row = summary[summary["stratum"] == dev_fixed["stratum"]].iloc[0]
    print(f"\n  DEVICE-PRICE DOUBLE REPRESENTATION -- P(CE) @ $100k at the "
          f"{dev_fixed['stratum']} stratum (p = {dev_fixed['p']:.3f}):")
    print(f"    device SAMPLED (truncated $100-$900) : "
          f"{float(low_row['p_ce_100k']):.4f}   mean INMB "
          f"{_money(low_row['inmb_100k_mean'])}")
    print(f"    device HELD FIXED at "
          f"${dev_fixed['device_fixed_at']:,.0f}          : "
          f"{dev_fixed['p_ce_100k']:.4f}   mean INMB "
          f"{_money(dev_fixed['inmb_100k_mean'])}")
    print("    the difference IS the device-price assumption; the price is also analysis "
          "axis 2, so sampling it represents the same uncertainty twice.")

    print(f"\n  RUNTIME (machine-dependent -- printed here and logged in "
          f"run_log.d/run_psa.md, deliberately NOT written into psa_summary.md, which "
          f"must be byte-reproducible): {len(p_values) * N_ITER:,} run_pair calls in "
          f"{sweep['seconds']:.0f} s "
          f"(~{sweep['seconds'] / (len(p_values) * N_ITER) * 1e6:.0f} us/pair); "
          f"+{N_ITER:,} calls in {t_dev:.0f} s for the device-fixed counterfactual.")

    print("\n  [OK] CEAC probabilities in [0,1] and summing to 1 at every WTP.")
    print(f"  [OK] p_base_ssi excluded by name -- {len(accept)} conditional risk points.")
    for name in TRUNCATE_TO_RANGE:
        x = sm["draws"][name].to_numpy()
        p = I.param_by_name(name)
        print(f"  [OK] {name} truncated to its declared range: realized "
              f"${x.min():,.2f}-${x.max():,.2f} inside "
              f"[${float(p.low):,.0f}, ${float(p.high):,.0f}]; "
              f"{int((x < float(p.low)).sum())} below / "
              f"{int((x > float(p.high)).sum())} above.")
    print(f"  [OK] 0 arm-order inversions in {len(p_values) * N_ITER:,} draws.")
    print("\nOutputs written:")
    for p in paths:
        print(f"  {p}")
    print("-" * 78)
    return 0


if __name__ == "__main__":
    main()
