import numpy as np
import pytest
from scipy import stats as scipy_stats

from mrsvalidate.analysis import distribution as dist


def test_kossler_mukherjee_chi2_critical_value_matches_equation():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = rng.normal(loc=1.0, size=200)
    ids = np.array([f"s{i}" for i in range(200)])
    r = dist.ecwas_kossler_mukherjee(x, y, ids, alpha=0.05, critical_value_method="chi2_asymptotic")
    assert r.critical_value == pytest.approx(scipy_stats.chi2.ppf(0.95, df=3))
    assert r.p_value == pytest.approx(scipy_stats.chi2.sf(r.statistic, df=3))


def test_kossler_mukherjee_monte_carlo_critical_value_reproducible():
    rng = np.random.default_rng(1)
    x = rng.normal(size=100)
    y = rng.normal(loc=0.5, size=100)
    ids = np.array([f"s{i}" for i in range(100)])
    r1 = dist.ecwas_kossler_mukherjee(
        x, y, ids, critical_value_method="monte_carlo", n_null_resamples=300, seed=7
    )
    r2 = dist.ecwas_kossler_mukherjee(
        x, y, ids, critical_value_method="monte_carlo", n_null_resamples=300, seed=7
    )
    assert r1.critical_value == pytest.approx(r2.critical_value)
    assert r1.p_value == pytest.approx(r2.p_value)
    assert 0.0 <= r1.p_value <= 1.0


def test_kossler_mukherjee_two_methods_roughly_agree_for_large_n():
    rng = np.random.default_rng(2)
    x = rng.normal(size=500)
    y = rng.normal(size=500)
    ids = np.array([f"s{i}" for i in range(500)])
    r_chi2 = dist.ecwas_kossler_mukherjee(x, y, ids, critical_value_method="chi2_asymptotic")
    r_mc = dist.ecwas_kossler_mukherjee(
        x, y, ids, critical_value_method="monte_carlo", n_null_resamples=2000, seed=0
    )
    assert r_chi2.critical_value == pytest.approx(r_mc.critical_value, rel=0.25)


def test_max_shape_default_monte_carlo_critical_value_present():
    rng = np.random.default_rng(3)
    x = rng.normal(size=150)
    y = rng.normal(loc=0.3, size=150)
    ids = np.array([f"s{i}" for i in range(150)])
    r = dist.ecwas_manuscript_max_shape(x, y, ids, n_null_resamples=300, seed=0)
    assert r.critical_value_method == "monte_carlo"
    assert np.isfinite(r.critical_value)
    assert r.p_value is not None


def test_max_shape_chi2_asymptotic_option():
    rng = np.random.default_rng(4)
    x = rng.normal(size=150)
    y = rng.normal(loc=0.3, size=150)
    ids = np.array([f"s{i}" for i in range(150)])
    r = dist.ecwas_manuscript_max_shape(
        x, y, ids, critical_value_method="chi2_asymptotic", n_null_resamples=300, seed=0
    )
    assert r.critical_value == pytest.approx(scipy_stats.chi2.ppf(0.95, df=3))


def test_reject_null_flag_consistent_with_statistic_vs_critical_value():
    rng = np.random.default_rng(5)
    x = rng.normal(size=300)
    y = rng.normal(loc=3.0, scale=2.0, size=300)  # clearly different distribution
    ids = np.array([f"s{i}" for i in range(300)])
    r = dist.ecwas_kossler_mukherjee(x, y, ids)
    assert r.reject_null == (r.statistic > r.critical_value)
    assert r.reject_null is True


def test_kossler_mukherjee_defaults_to_permutation_null():
    rng = np.random.default_rng(5)
    x, y = rng.normal(size=60), rng.normal(size=60)
    ids = np.array([f"r{i}" for i in range(120)])
    r = dist.ecwas_kossler_mukherjee(x, y, ids, seed=0)
    assert r.critical_value_method == "monte_carlo"
    assert r.meta.configuration["critical_value_source"] == "joint_permutation_null"
    assert r.meta.configuration["n_null_resamples"] == 10000


def test_permutation_null_accounts_for_correlated_components():
    # Wilcoxon and AntiSavage scores are strongly correlated, so the
    # composite's null upper quantile lies above the chi2(3) value that
    # would apply to three independent components.
    rng = np.random.default_rng(6)
    x, y = rng.normal(size=200), rng.normal(size=200)
    ids = np.array([f"r{i}" for i in range(400)])
    r = dist.ecwas_kossler_mukherjee(x, y, ids, seed=0)
    assert r.critical_value > scipy_stats.chi2.ppf(0.95, df=3)


def test_permutation_null_controls_type_one_error_where_chi2_does_not():
    # Without ties the permutation null depends only on the group sizes, so
    # one permutation critical value applies to every null dataset of that
    # size. Compare its type-I error with the chi2(3) critical value.
    rng = np.random.default_rng(7)
    n, reps = 100, 4000
    ids = np.array([f"r{i}" for i in range(2 * n)])
    perm_crit = dist.ecwas_kossler_mukherjee(
        rng.normal(size=n), rng.normal(size=n), ids, n_null_resamples=20000, seed=0
    ).critical_value
    chi2_crit = scipy_stats.chi2.ppf(0.95, df=3)
    stats = np.array([
        dist.ecwas_kossler_mukherjee(
            rng.normal(size=n), rng.normal(size=n), ids, critical_value_method="chi2_asymptotic"
        ).statistic
        for _ in range(reps)
    ])
    # Binomial SD of a 5 % rate over 4000 replicates is about 0.35 percentage points.
    assert abs(np.mean(stats > perm_crit) - 0.05) < 0.012
    assert np.mean(stats > chi2_crit) > 0.06
