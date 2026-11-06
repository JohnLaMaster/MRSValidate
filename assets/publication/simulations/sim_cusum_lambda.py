#!/usr/bin/env python3
"""CUSUM wild bootstrap and the error-variance ratio lambda; Theil-Sen with an exact reference.

Supports Supplementary Section ``supp:sim:cusum_lambda`` (manuscript claims,
Passing-Bablok and Theil-Sen subsections: the bootstrap is calibrated only
when the flip direction matches the error structure; in silico the
reference is exact (lambda = 0) and Theil-Sen is used; in vivo lambda is
supplied as for Deming; over-/under-estimating lambda; Theil-Sen is
unbiased with an exact reference where Passing-Bablok is not, and Sen's
interval covers with and without tied reference values).

Notation (manuscript): x = DL output, y = reference,
lambda = sigma_y^2 / sigma_x^2 (Eqn. deming_lambda), i.e. the library's
``cusum_bootstrap_variance_ratio`` = var(reference error) / var(candidate
error). ``None`` (library default) flips residuals perpendicular to the
line and is identical to lambda = 1.

Part ``lambda`` (reuses the development scripts behind the 2026-11-04 D2
table in assets/dev/MANUSCRIPT_UPDATES.md):

* Truth T ~ U(1, 10) mmol/L, y = T + e_y, x = 0.3 + T + e_x,
  e_y ~ N(0, sd_y^2), e_x ~ N(0, sd_x^2).
* Settings (sd_y, sd_x): in silico (0, 0.5), lambda = 0;
  (0.25, 0.5), lambda = 0.25; (0.5, 0.5), lambda = 1; (0.5, 0.25), lambda = 4.
* Estimator / supplied lambda: in silico: Passing-Bablok with None,
  Passing-Bablok with 0.01 (the library requires lambda > 0 for
  Passing-Bablok), Theil-Sen (lambda = 0 built in). Errors in both:
  Passing-Bablok with None and with the true lambda at every n; at
  n = 1000 also with lambda/4, lambda/2, 2 lambda, 4 lambda.
* Wild bootstrap with B = 199 replicates, nominal 5 %; output is the
  rejection rate of a true linear null.

Part ``theil_sen`` (reuses ``tests/unit/test_theil_sen_validation.py``):

* Slope bias, n = 150: y = T exact (continuous U(1, 10), or 5 equally
  spaced levels repeated 30 times), x = y + N(0, 0.5^2), true slope 1.
  Theil-Sen vs Passing-Bablok mean slope minus 1, with MC SE.
* Slope CI coverage, n = 60: y exact (continuous, or 10 levels x 6),
  x = 0.5 + 1.1 y + N(0, 0.5^2); coverage of the analytic 95 % intervals
  (Sen 1968 tie-corrected for Theil-Sen; Passing-Bablok 1983 for PB).
* Theil-Sen bootstrap CUSUM, n = 60: size under the linear model above
  and power against x = y + 0.15 (y - 5.5)^2 + N(0, 0.5^2), B = 199.

Usage::

    PYTHONPATH=src python assets/publication/simulations/sim_cusum_lambda.py [--quick] [--workers K] [--part lambda|theil_sen|all]
"""

from __future__ import annotations

import time
import warnings

import numpy as np

from _simcommon import ensure_src_on_path, mc_se, parse_args, run_cells, write_results

ensure_src_on_path()
from mrsvalidate.analysis import method_comparison as mc  # noqa: E402

SEED = 20261104
ALPHA = 0.05
N_CUSUM_BOOT = 199

# --- part "lambda" -----------------------------------------------------------
N_REPS_LAMBDA = 2_000  # MC SE about 0.49 percentage points at 5 %; ~5 h on 4 cores (see README)
LAMBDA_SIZES = (200, 500, 1000)
MISSPEC_SIZE = 1000
MISSPEC_FACTORS = (0.25, 0.5, 2.0, 4.0)
TRUE_INTERCEPT_LAMBDA = 0.3
SETTINGS = {  # name: (sd_y reference, sd_x DL), mmol/L
    "in_silico": (0.0, 0.5),
    "lambda_0.25": (0.25, 0.5),
    "lambda_1": (0.5, 0.5),
    "lambda_4": (0.5, 0.25),
}
IN_SILICO_PB_SMALL_LAMBDA = 0.01

# --- part "theil_sen" -----------------------------------------------------------
N_REPS_TS = 5_000
TS_BIAS_N, TS_BIAS_SD = 150, 0.5
TS_BIAS_LEVELS = (None, 5)
TS_CI_N, TS_CI_SD, TS_CI_SLOPE, TS_CI_INTERCEPT = 60, 0.5, 1.1, 0.5
TS_CI_LEVELS = (None, 10)
TS_CUSUM_CURVATURE = 0.15

QUICK_REPS_LAMBDA = 20
QUICK_REPS_TS = 100
QUICK_LAMBDA_SIZES = (200,)
QUICK_MISSPEC_SIZE = 200


def _lambda_cells(sizes, misspec_size, reps):
    cells = []
    for n in sizes:
        cells += [
            {"setting": "in_silico", "n": n, "estimator": "passing_bablok", "supplied": None},
            {"setting": "in_silico", "n": n, "estimator": "passing_bablok", "supplied": IN_SILICO_PB_SMALL_LAMBDA},
            {"setting": "in_silico", "n": n, "estimator": "theil_sen", "supplied": 0.0},
        ]
        for name in ("lambda_0.25", "lambda_4"):
            cells += [
                {"setting": name, "n": n, "estimator": "passing_bablok", "supplied": None},
                {"setting": name, "n": n, "estimator": "passing_bablok", "supplied": "true"},
            ]
        # lambda = 1: None is identical to supplying 1.0, so only one cell.
        cells.append({"setting": "lambda_1", "n": n, "estimator": "passing_bablok", "supplied": "true"})
    for name in ("lambda_0.25", "lambda_1", "lambda_4"):
        for f in MISSPEC_FACTORS:
            cells.append({"setting": name, "n": misspec_size, "estimator": "passing_bablok", "supplied": f})
    for c in cells:
        c["n_reps"] = reps
    return cells


def run_lambda_cell(cell: dict, ss: np.random.SeedSequence) -> dict:
    rng = np.random.default_rng(ss)
    n, reps = cell["n"], cell["n_reps"]
    sd_y, sd_x = SETTINGS[cell["setting"]]
    lam_true = sd_y**2 / sd_x**2
    sup = cell["supplied"]
    if sup is None:
        lam_sup, label, factor = None, "none (perpendicular)", float("nan")
    elif sup == "true":
        lam_sup, label, factor = lam_true, "true", 1.0
    elif cell["estimator"] == "theil_sen":
        lam_sup, label, factor = 0.0, "0 (built in)", 1.0
    elif cell["setting"] == "in_silico":
        lam_sup, label, factor = float(sup), f"{sup:g}", float("nan")
    else:
        lam_sup, label, factor = lam_true * sup, f"{sup:g} x true", float(sup)
    ids = np.arange(n).astype(str)
    rej = n_valid = 0
    for _ in range(reps):
        truth = rng.uniform(1.0, 10.0, n)
        ref = truth + rng.normal(0.0, sd_y, n) if sd_y > 0 else truth
        cand = TRUE_INTERCEPT_LAMBDA + truth + rng.normal(0.0, sd_x, n)
        boot_seed = int(rng.integers(2**31 - 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if cell["estimator"] == "theil_sen":
                r = mc.theil_sen(ref, cand, ids, cusum_alpha=ALPHA, n_cusum_bootstrap=N_CUSUM_BOOT, seed=boot_seed)
            else:
                r = mc.passing_bablok(
                    ref, cand, ids, cusum_alpha=ALPHA, n_cusum_bootstrap=N_CUSUM_BOOT, seed=boot_seed,
                    cusum_bootstrap_variance_ratio=lam_sup,
                )
        if not np.isfinite(r.cusum_p_value):
            continue
        n_valid += 1
        rej += bool(r.cusum_reject_linearity)
    rate = rej / max(n_valid, 1)
    return {
        "part": "lambda",
        "setting": cell["setting"],
        "n": n,
        "ref_error_sd": sd_y,
        "dl_error_sd": sd_x,
        "lambda_true": lam_true,
        "estimator": cell["estimator"],
        "lambda_supplied": "None" if lam_sup is None else lam_sup,
        "lambda_supplied_label": label,
        "misspecification_factor": factor,
        "true_intercept": TRUE_INTERCEPT_LAMBDA,
        "true_slope": 1.0,
        "truth_range": "U(1, 10)",
        "alpha": ALPHA,
        "n_cusum_bootstrap": N_CUSUM_BOOT,
        "n_reps": reps,
        "n_valid": n_valid,
        "rate": rate,
        "mc_se": mc_se(rate, n_valid),
        "seed": SEED,
    }


def _exact_reference(rng, n, levels):
    if levels is None:
        return rng.uniform(1.0, 10.0, n)
    return np.repeat(np.linspace(1.0, 10.0, levels), n // levels)


def run_ts_cell(cell: dict, ss: np.random.SeedSequence) -> dict:
    rng = np.random.default_rng(ss)
    reps, kind, levels = cell["n_reps"], cell["kind"], cell["levels"]
    base = {"part": "theil_sen", "quantity": kind, "reference": "continuous U(1, 10)" if levels is None else
            f"{levels} levels in [1, 10]", "n_reps": reps, "seed": SEED}
    if kind == "slope_bias":
        n = TS_BIAS_N
        ids = np.arange(n).astype(str)
        ts, pb = np.empty(reps), np.empty(reps)
        for k in range(reps):
            ref = _exact_reference(rng, n, levels)
            cand = ref + rng.normal(scale=TS_BIAS_SD, size=n)
            ts[k] = mc.theil_sen(ref, cand, ids, cusum_method="passing_bablok_1983").slope
            pb[k] = mc.passing_bablok(ref, cand, ids, cusum_method="passing_bablok_1983").slope
        return {**base, "n": n, "dl_error_sd": TS_BIAS_SD, "true_slope": 1.0, "true_intercept": 0.0,
                "ts_value": float(np.mean(ts) - 1.0), "ts_mc_se": float(np.std(ts, ddof=1) / np.sqrt(reps)),
                "pb_value": float(np.mean(pb) - 1.0), "pb_mc_se": float(np.std(pb, ddof=1) / np.sqrt(reps))}
    if kind == "slope_ci_coverage":
        n = TS_CI_N
        ids = np.arange(n).astype(str)
        cov_ts = cov_pb = 0
        for _ in range(reps):
            ref = _exact_reference(rng, n, levels)
            cand = TS_CI_INTERCEPT + TS_CI_SLOPE * ref + rng.normal(scale=TS_CI_SD, size=n)
            lo, hi = mc.theil_sen(ref, cand, ids, cusum_method="passing_bablok_1983").slope_ci
            cov_ts += lo <= TS_CI_SLOPE <= hi
            lo, hi = mc.passing_bablok(ref, cand, ids, cusum_method="passing_bablok_1983").slope_ci
            cov_pb += lo <= TS_CI_SLOPE <= hi
        p_ts, p_pb = cov_ts / reps, cov_pb / reps
        return {**base, "n": n, "dl_error_sd": TS_CI_SD, "true_slope": TS_CI_SLOPE, "true_intercept": TS_CI_INTERCEPT,
                "confidence_level": 0.95, "ts_value": p_ts, "ts_mc_se": mc_se(p_ts, reps),
                "pb_value": p_pb, "pb_mc_se": mc_se(p_pb, reps)}
    if kind == "ts_cusum_size_power":
        n = TS_CI_N
        ids = np.arange(n).astype(str)
        rej_lin = rej_q = 0
        for _ in range(reps):
            ref = _exact_reference(rng, n, None)
            noise = rng.normal(scale=TS_CI_SD, size=n)
            cand = TS_CI_INTERCEPT + TS_CI_SLOPE * ref + noise
            cand_q = ref + TS_CUSUM_CURVATURE * (ref - 5.5) ** 2 + noise
            s1, s2 = (int(v) for v in rng.integers(2**31 - 1, size=2))
            rej_lin += bool(mc.theil_sen(ref, cand, ids, n_cusum_bootstrap=N_CUSUM_BOOT, seed=s1).cusum_reject_linearity)
            rej_q += bool(mc.theil_sen(ref, cand_q, ids, n_cusum_bootstrap=N_CUSUM_BOOT, seed=s2).cusum_reject_linearity)
        p_lin, p_q = rej_lin / reps, rej_q / reps
        return {**base, "n": n, "dl_error_sd": TS_CI_SD, "true_slope": TS_CI_SLOPE, "true_intercept": TS_CI_INTERCEPT,
                "curvature": TS_CUSUM_CURVATURE, "alpha": ALPHA, "n_cusum_bootstrap": N_CUSUM_BOOT,
                "ts_value": p_lin, "ts_mc_se": mc_se(p_lin, reps), "ts_power": p_q, "ts_power_mc_se": mc_se(p_q, reps)}
    raise ValueError(kind)


def main(argv=None) -> int:
    args = parse_args(__doc__, argv, parts=("lambda", "theil_sen"))
    part = args.part
    if part in ("theil_sen", "all"):
        reps = args.reps or (QUICK_REPS_TS if args.quick else N_REPS_TS)
        cells = [{"kind": "slope_bias", "levels": lv, "n_reps": reps} for lv in TS_BIAS_LEVELS]
        cells += [{"kind": "slope_ci_coverage", "levels": lv, "n_reps": reps} for lv in TS_CI_LEVELS]
        cells += [{"kind": "ts_cusum_size_power", "levels": None, "n_reps": reps}]
        t0 = time.perf_counter()
        rows = run_cells(run_ts_cell, cells, SEED + 1, args.workers)
        write_results("theil_sen_validation", rows, args.quick, time.perf_counter() - t0)
    if part in ("lambda", "all"):
        reps = args.reps or (QUICK_REPS_LAMBDA if args.quick else N_REPS_LAMBDA)
        cells = _lambda_cells(
            QUICK_LAMBDA_SIZES if args.quick else LAMBDA_SIZES,
            QUICK_MISSPEC_SIZE if args.quick else MISSPEC_SIZE,
            reps,
        )
        # Longest cells first so that a parallel pool finishes evenly.
        order = sorted(range(len(cells)), key=lambda i: -cells[i]["n"])
        t0 = time.perf_counter()
        rows = run_cells(run_lambda_cell, [cells[i] for i in order], SEED, args.workers)
        rows = [r for _, r in sorted(zip(order, rows))]
        write_results("cusum_lambda", rows, args.quick, time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
