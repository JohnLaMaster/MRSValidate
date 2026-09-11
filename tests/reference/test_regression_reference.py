import numpy as np
import pytest
from scipy import stats as scipy_stats

from mrsvalidate.analysis import regression as reg


def test_theil_sen_matches_scipy_reference():
    rng = np.random.default_rng(0)
    prediction = np.linspace(1, 10, 20)
    reference = 2.0 * prediction + 1.0 + rng.normal(scale=0.01, size=20)
    ids = np.array([f"s{i}" for i in range(20)])

    expected_slope, expected_intercept, expected_low, expected_high = (
        scipy_stats.theilslopes(reference, prediction, method="joint")
    )
    result = reg.theil_sen(reference, prediction, ids)
    assert result.slope == pytest.approx(expected_slope)
    assert result.intercept == pytest.approx(expected_intercept)
    assert result.slope_low == pytest.approx(expected_low)
    assert result.slope_high == pytest.approx(expected_high)
    assert result.meta.configuration["intercept_method"] == "joint"


def test_pearson_r_matches_scipy_reference():
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.5])
    pred = np.array([1.2, 1.9, 3.3, 3.8, 5.1])
    ids = np.array(["a", "b", "c", "d", "e"])

    expected = scipy_stats.pearsonr(ref, pred)
    result = reg.pearson_r(ref, pred, ids)
    assert result.r == pytest.approx(expected.statistic)
    assert result.p_value == pytest.approx(expected.pvalue)

