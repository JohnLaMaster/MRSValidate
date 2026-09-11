import numpy as np
import pytest
from scipy import stats as scipy_stats

from mrsvalidate.analysis import normality as norm
from mrsvalidate.results import ResultStatus


def test_shapiro_wilk_does_not_reject_normal_data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    ids = np.array([f"s{i}" for i in range(200)])
    r = norm.shapiro_wilk(x, ids, seed=1, n_null_resamples=500)
    assert r.meta.status is ResultStatus.VALID
    assert not r.reject_null
    assert r.p_value > 0.05


def test_shapiro_wilk_rejects_clearly_nonnormal_data():
    rng = np.random.default_rng(0)
    x = rng.exponential(size=300)
    ids = np.array([f"s{i}" for i in range(300)])
    r = norm.shapiro_wilk(x, ids, seed=1, n_null_resamples=500)
    assert r.reject_null
    assert r.p_value < 0.01


def test_shapiro_wilk_critical_value_reproducible_with_seed():
    rng = np.random.default_rng(0)
    x = rng.normal(size=50)
    ids = np.array([f"s{i}" for i in range(50)])
    r1 = norm.shapiro_wilk(x, ids, seed=42, n_null_resamples=300)
    r2 = norm.shapiro_wilk(x, ids, seed=42, n_null_resamples=300)
    assert r1.critical_value == pytest.approx(r2.critical_value)


def test_normality_test_suite_runs_all_five():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    ids = np.array([f"s{i}" for i in range(200)])
    results = norm.normality_test_suite(x, ids, seed=0)
    assert set(results.keys()) == {
        "shapiro_wilk", "kolmogorov_smirnov", "lilliefors", "anderson_darling", "dagostino_pearson",
    }
    assert all(r.meta.status is ResultStatus.VALID for r in results.values())
