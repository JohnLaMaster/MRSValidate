#!/usr/bin/env python3
"""Passing-Bablok CUSUM linearity test: classical Kolmogorov critical value vs wild bootstrap.

Supports Supplementary Section ``supp:sim:pb_cusum_calibration`` (manuscript
claim, Passing-Bablok subsection: the classical critical value 1.36 is
strongly conservative because the line is fitted to the same data; the
wild bootstrap with refitting is close to nominal and more powerful).

Design (reuses ``tests/unit/test_passing_bablok_validation.py``):

* Truth T ~ U(1, 10) mmol/L, reference y = T + e_y, DL output
  x = 0.5 + 1.1 T + e_x (null, linear) or x = T + c (T - 5.5)^2 + e_x
  (alternative, curvature c), with errors in both methods.
* Error models (null): ``normal`` e_y ~ N(0, 0.3^2), e_x ~ N(0, 0.33^2)
  (error SD ratio equal to the slope: the Passing-Bablok consistency
  condition); ``heteroscedastic``: SDs multiplied by T / 5.5 (constant CV,
  same average SD); ``skewed``: centred lognormal errors (log-SD 0.5)
  rescaled to the same SDs. Alternative: normal errors, both SD 0.3.
* For every dataset, ``method_comparison.passing_bablok(y, x, ids,
  cusum_method="bootstrap", n_cusum_bootstrap=B)`` gives the statistic H
  and the bootstrap decision; the classical decision is H > 1.358
  (Kolmogorov 95 % quantile, ``scipy.stats.kstwobign``), i.e. exactly
  ``cusum_method="passing_bablok_1983"`` applied to the same H.
  ``cusum_bootstrap_variance_ratio`` is left at None (perpendicular flips),
  appropriate here because the error variances are of similar size and n
  is below 500 (see sim_cusum_lambda.py for unequal error variances).
* Output: rejection rates (size under the null, power under curvature)
  with Monte Carlo SEs and all parameters.

Usage::

    PYTHONPATH=src python assets/publication/simulations/sim_pb_cusum_calibration.py [--quick] [--workers K]
"""

from __future__ import annotations

import time
import warnings

import numpy as np
from scipy import stats as scipy_stats

from _simcommon import ensure_src_on_path, mc_se, parse_args, run_cells, write_results

ensure_src_on_path()
from mrsvalidate.analysis import method_comparison as mc  # noqa: E402

CLAIM = "pb_cusum_calibration"
SEED = 20261010
N_REPS = 5_000  # MC SE about 0.31 percentage points at 5 %
N_CUSUM_BOOT = 199  # bootstrap replicates per dataset (library default 1000)
ALPHA = 0.05
TRUE_INTERCEPT, TRUE_SLOPE = 0.5, 1.1
REF_SD = 0.3  # mmol/L
TRUTH_RANGE = (1.0, 10.0)  # mmol/L
LOGNORMAL_SIGMA = 0.5

NULL_SIZES = (60, 200, 500)
NULL_ERRORS = ("normal", "heteroscedastic", "skewed")
POWER_SIZES = (60, 200)
CURVATURES = (0.01, 0.02, 0.04, 0.06)

QUICK_REPS = 40
QUICK_NULL_SIZES = (60, 200)

KOLMOGOROV_CRIT = float(scipy_stats.kstwobign.isf(ALPHA))  # 1.3581


def _errors(rng, sd, kind, truth):
    n = truth.shape[0]
    if kind == "normal":
        return rng.normal(scale=sd, size=n)
    if kind == "heteroscedastic":
        return rng.normal(size=n) * sd * truth / np.mean(TRUTH_RANGE)
    if kind == "skewed":
        s2 = LOGNORMAL_SIGMA**2
        z = rng.lognormal(sigma=LOGNORMAL_SIGMA, size=n)
        return (z - np.exp(s2 / 2)) / np.sqrt((np.exp(s2) - 1) * np.exp(s2)) * sd
    raise ValueError(kind)


def simulate(rng, n, error, curvature):
    truth = rng.uniform(*TRUTH_RANGE, n)
    ref = truth + _errors(rng, REF_SD, error, truth)
    if curvature == 0.0:
        cand = TRUE_INTERCEPT + TRUE_SLOPE * truth + _errors(rng, REF_SD * TRUE_SLOPE, error, truth)
    else:
        cand = truth + curvature * (truth - np.mean(TRUTH_RANGE)) ** 2 + _errors(rng, REF_SD, error, truth)
    return ref, cand


def run_cell(cell: dict, ss: np.random.SeedSequence) -> dict:
    rng = np.random.default_rng(ss)
    n, reps = cell["n"], cell["n_reps"]
    ids = np.array([f"s{i}" for i in range(n)])
    rej_boot = rej_classical = n_valid = 0
    for _ in range(reps):
        ref, cand = simulate(rng, n, cell["error"], cell["curvature"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mc.passing_bablok(
                ref, cand, ids, cusum_method="bootstrap", cusum_alpha=ALPHA,
                n_cusum_bootstrap=N_CUSUM_BOOT, seed=int(rng.integers(2**31 - 1)),
            )
        if not np.isfinite(r.cusum_p_value):
            continue
        n_valid += 1
        rej_boot += bool(r.cusum_reject_linearity)
        rej_classical += bool(r.cusum_statistic > KOLMOGOROV_CRIT)
    rb, rc = rej_boot / max(n_valid, 1), rej_classical / max(n_valid, 1)
    return {
        "hypothesis": "null (linear)" if cell["curvature"] == 0.0 else "alternative (quadratic)",
        "n": n,
        "error": cell["error"],
        "curvature": cell["curvature"],
        "true_intercept": TRUE_INTERCEPT if cell["curvature"] == 0.0 else 0.0,
        "true_slope": TRUE_SLOPE if cell["curvature"] == 0.0 else 1.0,
        "ref_error_sd": REF_SD,
        "dl_error_sd": REF_SD * TRUE_SLOPE if cell["curvature"] == 0.0 else REF_SD,
        "truth_range": f"U{TRUTH_RANGE}",
        "alpha": ALPHA,
        "n_cusum_bootstrap": N_CUSUM_BOOT,
        "n_reps": reps,
        "n_valid": n_valid,
        "rate_classical": rc,
        "mc_se_classical": mc_se(rc, n_valid),
        "rate_bootstrap": rb,
        "mc_se_bootstrap": mc_se(rb, n_valid),
        "kolmogorov_critical_value": KOLMOGOROV_CRIT,
        "seed": SEED,
    }


def main(argv=None) -> int:
    args = parse_args(__doc__, argv)
    reps = args.reps or (QUICK_REPS if args.quick else N_REPS)
    sizes = QUICK_NULL_SIZES if args.quick else NULL_SIZES
    cells = [{"n": n, "error": e, "curvature": 0.0, "n_reps": reps} for n in sizes for e in NULL_ERRORS]
    cells += [{"n": n, "error": "normal", "curvature": c, "n_reps": reps} for n in POWER_SIZES for c in CURVATURES]
    t0 = time.perf_counter()
    rows = run_cells(run_cell, cells, SEED, args.workers)
    write_results(CLAIM, rows, args.quick, time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
