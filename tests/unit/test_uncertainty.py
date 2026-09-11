import numpy as np
import pytest

from mrsvalidate.analysis import uncertainty as unc
from mrsvalidate.results import ResultStatus


def test_cobeau_positive_when_uncertainty_tracks_error():
    error = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    sigma = np.array([0.2, 0.4, 1.1, 1.9, 5.2])
    ids = np.array(["a", "b", "c", "d", "e"])
    r = unc.cobeau(error, sigma, ids)
    assert r.value > 0.9
    assert r.method == "pearson"


def test_cobeau_median_centered_has_no_pvalue():
    error = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    sigma = np.array([0.2, 0.4, 1.1, 1.9, 5.2])
    ids = np.array(["a", "b", "c", "d", "e"])
    r = unc.cobeau(error, sigma, ids, method="median_centered")
    assert r.p_value is None


def test_cobeau_fisher_aggregate_reduces_to_mean_when_equal_n():
    values = np.array([0.5, 0.7, 0.3])
    n = np.array([100, 100, 100])
    ids = np.array(["m1", "m2", "m3"])
    r = unc.cobeau_fisher_aggregate(values, n, ids)
    expected = np.tanh(np.mean(np.arctanh(values)))
    assert r.value == pytest.approx(expected)


def test_cobeau_fisher_aggregate_excludes_small_n_and_perfect_correlation():
    values = np.array([0.5, 1.0, 0.6])
    n = np.array([100, 100, 2])  # second excluded for |r|>=1, third for n<=3
    ids = np.array(["m1", "m2", "m3"])
    r = unc.cobeau_fisher_aggregate(values, n, ids)
    assert r.meta.n_used == 1
    assert r.value == pytest.approx(0.5)
