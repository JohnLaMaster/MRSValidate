import numpy as np
import pytest

from mrsvalidate.analysis import error_consistency as ec
from mrsvalidate.results import ResultStatus


def test_error_consistency_identical_residuals_gives_zero_contribution():
    r_i = np.array([1.0, -2.0, 3.0])
    r_j = np.array([1.0, -2.0, 3.0])
    ids = np.array(["a", "b", "c"])
    r = ec.error_consistency(r_i, r_j, ids)
    np.testing.assert_allclose(r.contributions, [0.0, 0.0, 0.0])
    assert r.rho_ec == pytest.approx(0.0)


def test_error_consistency_opposite_sign_same_magnitude_is_negative_but_magnitude_zero():
    # sign differs (negative contribution direction) but magnitudes equal -> |diff|=0
    r_i = np.array([2.0])
    r_j = np.array([-2.0])
    ids = np.array(["a"])
    r = ec.error_consistency(r_i, r_j, ids)
    assert r.contributions[0] == pytest.approx(0.0)


def test_error_consistency_different_magnitudes_same_sign():
    r_i = np.array([4.0])
    r_j = np.array([1.0])
    ids = np.array(["a"])
    r = ec.error_consistency(r_i, r_j, ids)
    expected = 1.0 * (4.0 - 1.0) / (4.0 + 1.0)
    assert r.contributions[0] == pytest.approx(expected)


def test_error_consistency_zero_zero_pair_contributes_zero_and_is_counted():
    r_i = np.array([0.0, 1.0])
    r_j = np.array([0.0, 1.0])
    ids = np.array(["a", "b"])
    r = ec.error_consistency(r_i, r_j, ids)
    assert r.zero_zero_pairs == 1
    assert r.contributions[0] == pytest.approx(0.0)


def test_error_consistency_excludes_nan_rows():
    r_i = np.array([1.0, np.nan])
    r_j = np.array([1.0, 2.0])
    ids = np.array(["a", "b"])
    r = ec.error_consistency(r_i, r_j, ids)
    assert r.meta.n_used == 1
    assert r.meta.excluded_ids == ("b",)
