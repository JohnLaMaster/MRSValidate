"""Simulation-based validation of Passing-Bablok regression.

* Classical (asymptotic Kolmogorov) CUSUM linearity test: rejection rate
  under a truly linear model with error in both methods should not exceed
  the nominal 5% by much (it is in fact very conservative, because the
  line is fitted), and it should reject a clearly quadratic relation most
  of the time.
* Wild-bootstrap CUSUM test (``cusum_method="bootstrap"``): rejection rate under linearity
  close to the nominal 5% (slightly below, since the statistic is
  lattice-valued), and higher power than the classical test against mild
  curvature.
* Analytic (rank-based) slope CI: coverage of the true slope should be
  close to the nominal 95%.
"""

import numpy as np

from mrsvalidate.analysis import method_comparison as mc

N_OBS = 60
N_REPS = 300
TRUE_SLOPE = 1.1
TRUE_INTERCEPT = 0.5


def _simulate(rng, quadratic=False, curvature=0.3):
    truth = rng.uniform(1.0, 10.0, N_OBS)
    ref = truth + rng.normal(scale=0.3, size=N_OBS)
    if quadratic:
        cand = truth + curvature * (truth - 5.5) ** 2 + rng.normal(scale=0.3, size=N_OBS)
    else:
        # error SD ratio equal to the slope: the PB consistency condition.
        cand = TRUE_INTERCEPT + TRUE_SLOPE * truth + rng.normal(scale=0.3 * TRUE_SLOPE, size=N_OBS)
    return ref, cand


def test_pb1983_cusum_size_and_power_and_slope_ci_coverage():
    rng = np.random.default_rng(20261010)
    ids = np.array([f"s{i}" for i in range(N_OBS)])
    reject_linear = reject_quadratic = covered = 0
    for _ in range(N_REPS):
        ref, cand = _simulate(rng)
        r = mc.passing_bablok(ref, cand, ids, cusum_method="passing_bablok_1983")
        reject_linear += r.cusum_reject_linearity
        covered += r.slope_ci[0] <= TRUE_SLOPE <= r.slope_ci[1]
        ref_q, cand_q = _simulate(rng, quadratic=True)
        reject_quadratic += mc.passing_bablok(
            ref_q, cand_q, ids, cusum_method="passing_bablok_1983"
        ).cusum_reject_linearity

    size = reject_linear / N_REPS
    power = reject_quadratic / N_REPS
    coverage = covered / N_REPS
    assert size <= 0.07, size
    assert power >= 0.8, power
    assert 0.90 <= coverage <= 0.99, coverage


N_BOOT_REPS = 400
N_CUSUM_BOOT = 99  # small B to keep the test fast; the default is 1000


def test_bootstrap_cusum_size_is_near_nominal_unlike_asymptotic():
    rng = np.random.default_rng(20261011)
    ids = np.array([f"s{i}" for i in range(N_OBS)])
    reject_boot = reject_asym = 0
    for k in range(N_BOOT_REPS):
        ref, cand = _simulate(rng)
        r = mc.passing_bablok(ref, cand, ids, cusum_method="bootstrap", n_cusum_bootstrap=N_CUSUM_BOOT, seed=k)
        reject_boot += r.cusum_reject_linearity
        reject_asym += r.cusum_statistic > 1.3581
    size_boot = reject_boot / N_BOOT_REPS
    size_asym = reject_asym / N_BOOT_REPS
    assert 0.015 <= size_boot <= 0.08, size_boot
    assert size_asym <= 0.015, size_asym


def test_bootstrap_cusum_has_more_power_than_asymptotic_against_mild_curvature():
    rng = np.random.default_rng(20261012)
    ids = np.array([f"s{i}" for i in range(N_OBS)])
    reject_boot = reject_asym = 0
    n_reps = 200
    for k in range(n_reps):
        ref, cand = _simulate(rng, quadratic=True, curvature=0.06)
        r = mc.passing_bablok(ref, cand, ids, cusum_method="bootstrap", n_cusum_bootstrap=N_CUSUM_BOOT, seed=k)
        reject_boot += r.cusum_reject_linearity
        reject_asym += r.cusum_statistic > 1.3581
    power_boot = reject_boot / n_reps
    power_asym = reject_asym / n_reps
    assert power_boot >= 0.5, power_boot
    assert power_boot >= power_asym + 0.2, (power_boot, power_asym)
