#!/usr/bin/env python3
"""ECWAS type-I error: chi^2_3 critical value vs joint permutation null.

Supports Supplementary Section ``supp:sim:ecwas_type1`` (manuscript claim:
the chi^2_3 reference is anti-conservative because the Wilcoxon, Ansari-
Bradley and AntiSavage components are correlated under H0; the joint
permutation null controls the level).

Design (reuses ``tests/unit/test_ecwas_critical_values.py::
test_permutation_null_controls_type_one_error_where_chi2_does_not``):

* Two independent samples of size ``n`` per group, both N(0, 1) (H0 true).
* Statistic: Kossler-Mukherjee ECWAS (Eqn. ecwas), computed with
  ``distribution.ecwas_kossler_mukherjee(..., critical_value_method=
  "chi2_asymptotic")`` (the statistic does not depend on the method).
* Critical values at alpha = 0.05:
  - chi^2_3 upper 5 % quantile (7.815);
  - joint permutation null: the library's ``critical_value_method=
    "monte_carlo"`` critical value from ``N_PERM_CRIT`` relabelings,
    computed once per ``n``. Without ties the permutation distribution of
    a rank statistic depends only on the group sizes, so one critical value
    applies to every null dataset of that size; this is equivalent to (and
    much cheaper than) running the per-dataset permutation test.
* Output: rejection rate of each critical value, its Monte Carlo SE
  sqrt(p(1-p)/R), the critical values and all parameters.

Usage::

    PYTHONPATH=src python assets/publication/simulations/sim_ecwas_type1.py [--quick] [--workers K]
"""

from __future__ import annotations

import time

import numpy as np
from scipy import stats as scipy_stats

from _simcommon import ensure_src_on_path, mc_se, parse_args, run_cells, write_results

ensure_src_on_path()
from mrsvalidate.analysis import distribution as dist  # noqa: E402

CLAIM = "ecwas_type1"
SEED = 7
N_REPS = 10_000  # MC SE about 0.22 percentage points at 5 %
N_PERM_CRIT = 20_000  # relabelings for the permutation critical value
ALPHA = 0.05
SAMPLE_SIZES = (30, 100, 500)  # per group; 100 + 100 is the value quoted in the manuscript notes

QUICK_REPS = 300
QUICK_PERM_CRIT = 2_000


def run_cell(cell: dict, ss: np.random.SeedSequence) -> dict:
    n, n_reps, n_perm = cell["n_per_group"], cell["n_reps"], cell["n_perm_crit"]
    rng = np.random.default_rng(ss)
    ids = np.array([f"r{i}" for i in range(2 * n)])
    perm_seed = int(rng.integers(2**31 - 1))
    perm_crit = dist.ecwas_kossler_mukherjee(
        rng.normal(size=n), rng.normal(size=n), ids, alpha=ALPHA,
        critical_value_method="monte_carlo", n_null_resamples=n_perm, seed=perm_seed,
    ).critical_value
    chi2_crit = float(scipy_stats.chi2.ppf(1.0 - ALPHA, df=3))
    stats = np.array([
        dist.ecwas_kossler_mukherjee(
            rng.normal(size=n), rng.normal(size=n), ids, alpha=ALPHA, critical_value_method="chi2_asymptotic"
        ).statistic
        for _ in range(n_reps)
    ])
    rate_chi2 = float(np.mean(stats > chi2_crit))
    rate_perm = float(np.mean(stats > perm_crit))
    return {
        "n_per_group": n,
        "distribution": "N(0,1) both groups",
        "alpha": ALPHA,
        "n_reps": n_reps,
        "n_perm_crit": n_perm,
        "chi2_critical_value": chi2_crit,
        "perm_critical_value": float(perm_crit),
        "rate_chi2": rate_chi2,
        "mc_se_chi2": mc_se(rate_chi2, n_reps),
        "rate_perm": rate_perm,
        "mc_se_perm": mc_se(rate_perm, n_reps),
        "seed": SEED,
    }


def main(argv=None) -> int:
    args = parse_args(__doc__, argv)
    n_reps = args.reps or (QUICK_REPS if args.quick else N_REPS)
    n_perm = QUICK_PERM_CRIT if args.quick else N_PERM_CRIT
    cells = [{"n_per_group": n, "n_reps": n_reps, "n_perm_crit": n_perm} for n in SAMPLE_SIZES]
    t0 = time.perf_counter()
    rows = run_cells(run_cell, cells, SEED, args.workers)
    write_results(CLAIM, rows, args.quick, time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
