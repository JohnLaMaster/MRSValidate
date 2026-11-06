#!/usr/bin/env python3
"""Cluster-robust inference with few clusters: CR0 + z vs CR1 + t(G-1) vs CR2 + Bell-McCaffrey.

Supports Supplementary Section ``supp:sim:cluster_robust`` (manuscript
claim, after Eqn. sandwich mean: with few clusters the sandwich variance is
biased downwards and normal references over-reject; CR2 with
Bell-McCaffrey degrees of freedom is used below 50 clusters).

Design (reuses ``tests/unit/test_mixed_effects.py::
test_cr2_bell_mccaffrey_size_closer_to_nominal_than_cr0_with_10_clusters``,
whose docstring records a reference run with R = 2000, seed 123):

* G clusters with m = 6 observations each (N = 6 G).
* Covariate z_gn = a_g + 0.5 b_gn, response y_gn = u_g + e_gn with
  a_g, b_gn, u_g, e_gn iid N(0, 1): true intercept and slope are 0, the
  covariate and the errors both have a cluster component (intraclass
  correlation 0.5 for y, 0.8 for z), the setting in which cluster-robust
  standard errors are needed and small-G bias is largest.
* OLS fit of y on (1, z); cluster-robust SEs and reference degrees of
  freedom from the library's ``mixed_effects._cluster_robust_inference``
  (private helper, used as in the unit test) for CR0 (normal reference),
  CR1 (t with G - 1 df) and CR2 (t with Bell-McCaffrey df); two-sided
  Wald test of each true-zero coefficient at nominal 5 %
  (``mixed_effects._t_test``).
* Output: rejection rate per method and coefficient with Monte Carlo SE.

Usage::

    PYTHONPATH=src python assets/publication/simulations/sim_cluster_robust.py [--quick] [--workers K]
"""

from __future__ import annotations

import time

import numpy as np

from _simcommon import ensure_src_on_path, mc_se, parse_args, run_cells, write_results

ensure_src_on_path()
from mrsvalidate.analysis import mixed_effects as me  # noqa: E402

CLAIM = "cluster_robust"
SEED = 123
N_REPS = 10_000  # MC SE about 0.22 percentage points at 5 %
ALPHA = 0.05
N_PER_CLUSTER = 6
CLUSTER_COUNTS = (10, 20, 50)  # 10 is the value quoted in the manuscript notes
COVARIATE_WITHIN_SD = 0.5
METHODS = ("CR0", "CR1", "CR2")

QUICK_REPS = 200


def run_cell(cell: dict, ss: np.random.SeedSequence) -> list[dict]:
    rng = np.random.default_rng(ss)
    g, reps = cell["n_clusters"], cell["n_reps"]
    codes = np.repeat(np.arange(g), N_PER_CLUSTER)
    labels = codes.astype(str)
    n_obs = codes.shape[0]
    rejections = {m: np.zeros(2) for m in METHODS}
    df_sum = {m: np.zeros(2) for m in METHODS}
    for _ in range(reps):
        cov = rng.normal(size=g)[codes] + COVARIATE_WITHIN_SD * rng.normal(size=n_obs)
        x = np.column_stack([np.ones(n_obs), cov])
        y = rng.normal(size=g)[codes] + rng.normal(size=n_obs)  # true coefficients 0
        b = np.linalg.lstsq(x, y, rcond=None)[0]
        resid = y - x @ b
        for m in METHODS:
            se, df, _ = me._cluster_robust_inference(x, resid, labels, m)
            df_sum[m] += np.where(np.isfinite(df), df, np.nan)
            for j in range(2):
                _stat, p = me._t_test(b[j], se[j], df[j])
                rejections[m][j] += p < ALPHA
    rows = []
    for m in METHODS:
        for j, coef in enumerate(("intercept", "slope")):
            rate = float(rejections[m][j] / reps)
            rows.append({
                "n_clusters": g,
                "obs_per_cluster": N_PER_CLUSTER,
                "method": m,
                "reference": {"CR0": "normal", "CR1": "t(G-1)", "CR2": "t(Bell-McCaffrey df)"}[m],
                "coefficient": coef,
                "mean_df": float(df_sum[m][j] / reps) if m != "CR0" else float("inf"),
                "alpha": ALPHA,
                "n_reps": reps,
                "rate": rate,
                "mc_se": mc_se(rate, reps),
                "seed": SEED,
            })
    return rows


def main(argv=None) -> int:
    args = parse_args(__doc__, argv)
    reps = args.reps or (QUICK_REPS if args.quick else N_REPS)
    cells = [{"n_clusters": g, "n_reps": reps} for g in CLUSTER_COUNTS]
    t0 = time.perf_counter()
    rows = [r for block in run_cells(run_cell, cells, SEED, args.workers) for r in block]
    write_results(CLAIM, rows, args.quick, time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
