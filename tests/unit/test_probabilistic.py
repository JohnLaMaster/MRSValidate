import numpy as np
import pytest

from mrsvalidate.analysis import probabilistic as prob
from mrsvalidate.results import ResultStatus


def test_nlpd_hand_computed():
    y_true = np.array([0.0, 1.0])
    y_pred = np.array([0.0, 0.0])
    var = np.array([1.0, 1.0])
    ids = np.array(["a", "b"])
    r = prob.nlpd(y_true, y_pred, var, ids)
    expected_term = [
        0.5 * (np.log(1.0) + 0.0 + np.log(2 * np.pi)),
        0.5 * (np.log(1.0) + 1.0 + np.log(2 * np.pi)),
    ]
    assert r.value == pytest.approx(np.mean(expected_term))


def test_nlpd_rejects_nonpositive_variance():
    y_true = np.array([0.0, 1.0])
    y_pred = np.array([0.0, 0.0])
    var = np.array([1.0, -1.0])
    ids = np.array(["a", "b"])
    r = prob.nlpd(y_true, y_pred, var, ids)
    assert r.meta.n_used == 1
    assert r.meta.exclusion_reasons == {"nonpositive_or_nonfinite_variance": 1}


def test_nlpd_all_invalid_is_invalid_input():
    y_true = np.array([0.0])
    y_pred = np.array([0.0])
    var = np.array([0.0])
    ids = np.array(["a"])
    r = prob.nlpd(y_true, y_pred, var, ids)
    assert r.meta.status is ResultStatus.INVALID_INPUT


def test_nlpd_variance_floor_applied_and_recorded():
    y_true = np.array([0.0])
    y_pred = np.array([0.0])
    var = np.array([1e-10])
    ids = np.array(["a"])
    r = prob.nlpd(y_true, y_pred, var, ids, variance_floor=0.01)
    assert r.meta.status is ResultStatus.VALID
    assert r.meta.configuration["n_variance_floor_applied"] == 1


def test_nlpd_macro_equal_weight():
    from mrsvalidate.results import ScalarMetricResult, build_meta

    m1 = prob.nlpd(np.array([0.0]), np.array([0.0]), np.array([1.0]), np.array(["a"]))
    m2 = prob.nlpd(np.array([1.0]), np.array([0.0]), np.array([4.0]), np.array(["a"]))
    macro = prob.nlpd_macro([m1, m2])
    assert macro.value == pytest.approx((m1.value + m2.value) / 2)


def test_joint_nlpd_full_covariance_diagonal_matches_sum_of_marginals():
    e = np.array([[1.0, 2.0]])
    cov = np.diag([1.0, 4.0])
    ids = np.array(["a"])
    r = prob.joint_nlpd_full_covariance(e, cov, ids)
    expected = 0.5 * (np.log(1.0 * 4.0) + (1.0**2 / 1.0 + 2.0**2 / 4.0) + 2 * np.log(2 * np.pi))
    assert r.values[0] == pytest.approx(expected)


def test_joint_nlpd_full_covariance_singular_excluded():
    e = np.array([[1.0, 2.0]])
    cov = np.array([[1.0, 1.0], [1.0, 1.0]])  # singular
    ids = np.array(["a"])
    r = prob.joint_nlpd_full_covariance(e, cov, ids)
    assert r.meta.status is ResultStatus.INVALID_INPUT
