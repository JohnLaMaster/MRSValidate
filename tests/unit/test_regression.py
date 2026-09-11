import numpy as np
import pytest

from mrsvalidate.analysis import regression as reg
from mrsvalidate.results import ResultStatus

IDS = np.array(["a", "b", "c", "d", "e"])


def test_perfect_identity():
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    r = reg.regression_metrics(ref, pred, IDS)
    assert r.r == pytest.approx(1.0)
    assert r.r_squared_direct == pytest.approx(1.0)
    assert r.slope == pytest.approx(1.0)
    assert r.intercept == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(r.residuals, 0.0, atol=1e-9)


def test_constant_additive_bias_still_perfectly_correlated():
    ref = np.array([1.0, 2.0, 3.0])
    pred = np.array([2.0, 3.0, 4.0])
    ids = np.array(["a", "b", "c"])
    r = reg.pearson_r(ref, pred, ids)
    assert r.r == pytest.approx(1.0)


def test_r_squared_direct_can_be_negative_and_is_not_clipped():
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([10.0, -10.0, 20.0, -20.0])  # terrible predictions
    ids = np.array(["a", "b", "c", "d"])
    r = reg.r_squared_direct(ref, pred, ids)
    assert r.value < 0.0


def test_r_squared_undefined_for_constant_reference():
    ref = np.array([5.0, 5.0, 5.0])
    pred = np.array([1.0, 2.0, 3.0])
    ids = np.array(["a", "b", "c"])
    r = reg.r_squared_direct(ref, pred, ids)
    assert r.meta.status is ResultStatus.UNDEFINED


def test_concise_score_manuscript_is_not_algebraic_r_cubed():
    ref = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    pred = np.array([1.5, 2.5, 2.5, 5.0, 90.0])
    ids = np.array(["a", "b", "c", "d", "e"])
    concise = reg.concise_score_manuscript(ref, pred, ids)
    r = reg.pearson_r(ref, pred, ids)
    assert concise.value != pytest.approx(r.r**3)
    r2 = reg.r_squared_direct(ref, pred, ids)
    assert concise.value == pytest.approx(r.r * max(0.0, r2.value))


def test_concise_score_manuscript_zero_when_r2_nonpositive():
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([10.0, -10.0, 20.0, -20.0])
    ids = np.array(["a", "b", "c", "d"])
    r2 = reg.r_squared_direct(ref, pred, ids)
    assert r2.value <= 0.0
    concise = reg.concise_score_manuscript(ref, pred, ids)
    assert concise.value == pytest.approx(0.0)


def test_median_centered_correlation_has_no_pvalue():
    ref = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    pred = np.array([1.1, 2.2, 2.9, 4.1, 95.0])
    ids = np.array(["a", "b", "c", "d", "e"])
    r = reg.median_centered_correlation(ref, pred, ids)
    assert r.p_value is None


def test_adjusted_r_squared_le_r_squared():
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    pred = np.array([1.1, 2.3, 2.8, 4.2, 4.9, 6.4])
    ids = np.array([f"s{i}" for i in range(6)])
    r2 = reg.r_squared_direct(ref, pred, ids)
    adj = reg.adjusted_r_squared(ref, pred, ids, n_predictors=1)
    assert adj.value <= r2.value + 1e-12
