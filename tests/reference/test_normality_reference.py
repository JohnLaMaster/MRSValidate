import numpy as np
import pytest
from scipy import stats as scipy_stats

from mrsvalidate.analysis import normality as norm
from mrsvalidate.results import ResultStatus


def test_kolmogorov_smirnov_against_true_null_matches_scipy_reference():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=5.0, scale=2.0, size=100)
    ids = np.array([f"s{i}" for i in range(100)])
    # Standardized rather than passed via args=(5.0, 2.0): some SciPy
    # versions raise a TypeError from kstest's internal fast path for
    # that call form (see mrsvalidate.analysis.normality.kolmogorov_smirnov),
    # and the KS statistic is exactly invariant under this rescaling.
    expected = scipy_stats.kstest((x - 5.0) / 2.0, "norm")
    r = norm.kolmogorov_smirnov(x, ids, mean=5.0, std=2.0)
    assert r.statistic == pytest.approx(expected.statistic)
    assert r.p_value == pytest.approx(expected.pvalue)
    assert r.critical_value == pytest.approx(scipy_stats.kstwobign.ppf(0.95) / np.sqrt(100))


def test_kolmogorov_smirnov_rejects_nonpositive_std():
    x = np.array([1.0, 2.0, 3.0])
    ids = np.array(["a", "b", "c"])
    r = norm.kolmogorov_smirnov(x, ids, mean=0.0, std=-1.0)
    assert r.meta.status is ResultStatus.INVALID_INPUT


def test_lilliefors_matches_statsmodels_reference():
    from statsmodels.stats.diagnostic import lilliefors as sm_lilliefors

    rng = np.random.default_rng(0)
    x = rng.normal(size=100)
    ids = np.array([f"s{i}" for i in range(100)])
    expected_stat, expected_p = sm_lilliefors(x, dist="norm", pvalmethod="table")
    r = norm.lilliefors(x, ids)
    assert r.statistic == pytest.approx(expected_stat)
    assert r.p_value == pytest.approx(expected_p)
    assert np.isfinite(r.critical_value)


def test_anderson_darling_matches_scipy_reference():
    rng = np.random.default_rng(0)
    x = rng.normal(size=150)
    ids = np.array([f"s{i}" for i in range(150)])
    expected = scipy_stats.anderson(x, dist="norm")
    r = norm.anderson_darling(x, ids, alpha=0.05)
    assert r.statistic == pytest.approx(expected.statistic)
    assert r.critical_value == pytest.approx(expected.critical_values[2])  # index for 5%


def test_anderson_darling_rejects_unsupported_alpha():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ids = np.array([f"s{i}" for i in range(5)])
    with pytest.raises(ValueError):
        norm.anderson_darling(x, ids, alpha=0.033)


def test_dagostino_pearson_matches_scipy_and_chi2_critical_value():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    ids = np.array([f"s{i}" for i in range(200)])
    expected = scipy_stats.normaltest(x)
    r = norm.dagostino_pearson(x, ids)
    assert r.statistic == pytest.approx(expected.statistic)
    assert r.p_value == pytest.approx(expected.pvalue)
    assert r.critical_value == pytest.approx(scipy_stats.chi2.ppf(0.95, df=2))


