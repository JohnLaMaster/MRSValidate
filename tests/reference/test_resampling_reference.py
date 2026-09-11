import numpy as np
import pytest
from scipy import stats as scipy_stats

from mrsvalidate.analysis import resampling as rs


def test_permutation_test_independent_matches_scipy_reference():
    rng = np.random.default_rng(2)
    a = rng.normal(size=30)
    b = rng.normal(loc=0.5, size=30)
    ids = np.array([f"s{i}" for i in range(30)])

    def stat(x, y):
        return np.mean(x) - np.mean(y)

    expected = scipy_stats.permutation_test(
        (a, b), stat, permutation_type="independent", n_resamples=500, random_state=42
    )
    result = rs.permutation_test(
        stat, a, b, sample_ids=ids, design="independent", n_resamples=500, seed=42
    )
    assert result.observed_statistic == pytest.approx(expected.statistic)
    assert result.p_value == pytest.approx(expected.pvalue)
    assert result.meta.configuration["scipy_permutation_type"] == "independent"


