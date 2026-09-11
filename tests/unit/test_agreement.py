import numpy as np
import pytest

from mrsvalidate.analysis import agreement as ag
from mrsvalidate.results import ResultStatus


def test_lin_ccc_perfect_identity_equals_one():
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.0, 2.0, 3.0, 4.0])
    ids = np.array(["a", "b", "c", "d"])
    r = ag.lin_ccc(ref, pred, ids)
    assert r.rho_c == pytest.approx(1.0)
    assert r.experimental is False


def test_lin_ccc_less_than_one_for_biased_but_correlated_data():
    # CCC is an agreement measure, not merely correlation: a perfectly
    # correlated but constant-bias-shifted method must disagree (rho_c < 1)
    # even though Pearson r == 1.
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([2.0, 3.0, 4.0, 5.0])
    ids = np.array(["a", "b", "c", "d"])
    r = ag.lin_ccc(ref, pred, ids)
    assert r.r == pytest.approx(1.0)
    assert r.rho_c < 1.0


def test_lin_ccc_symmetric_under_swap():
    ref = np.array([1.0, 5.0, 3.0, 8.0, 2.0])
    pred = np.array([2.0, 4.0, 3.5, 7.0, 3.0])
    ids = np.array(["a", "b", "c", "d", "e"])
    r_fwd = ag.lin_ccc(ref, pred, ids)
    r_swap = ag.lin_ccc(pred, ref, ids)
    assert r_fwd.rho_c == pytest.approx(r_swap.rho_c)


def test_robust_ccc_is_marked_experimental():
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.0, 2.0, 3.0, 4.0])
    ids = np.array(["a", "b", "c", "d"])
    r = ag.robust_ccc(ref, pred, ids)
    assert r.experimental is True
    assert r.method == "manuscript_robust_ccc"
    assert r.rho_c == pytest.approx(1.0)


def test_lin_ccc_undefined_for_constant_equal_vectors_with_zero_variance():
    ref = np.array([5.0, 5.0, 5.0])
    pred = np.array([5.0, 5.0, 5.0])
    ids = np.array(["a", "b", "c"])
    r = ag.lin_ccc(ref, pred, ids)
    assert r.meta.status is ResultStatus.UNDEFINED
