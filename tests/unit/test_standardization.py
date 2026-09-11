import numpy as np
import pytest

from mrsvalidate.analysis.standardization import standardize
from mrsvalidate.results import ResultStatus

IDS = np.array(["a", "b", "c", "d"])


def test_standardize_normal_uses_reference_derived_params_for_both():
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([2.0, 2.0, 4.0, 4.0])
    r = standardize(ref, pred, IDS, method="normal")
    mu = np.mean(ref)
    sigma = np.std(ref, ddof=1)
    assert r.location == pytest.approx(mu)
    assert r.scale == pytest.approx(sigma)
    np.testing.assert_allclose(r.z_reference, (ref - mu) / sigma)
    np.testing.assert_allclose(r.z_prediction, (pred - mu) / sigma)


def test_standardize_robust_uses_median_and_iqr_of_reference():
    ref = np.array([1.0, 2.0, 3.0, 100.0])
    pred = np.array([1.0, 2.0, 3.0, 4.0])
    r = standardize(ref, pred, IDS, method="robust")
    med = np.median(ref)
    q1, q3 = np.quantile(ref, [0.25, 0.75])
    assert r.location == pytest.approx(med)
    assert r.scale == pytest.approx(q3 - q1)


def test_standardize_undefined_for_constant_reference():
    ref = np.array([5.0, 5.0, 5.0, 5.0])
    pred = np.array([1.0, 2.0, 3.0, 4.0])
    r = standardize(ref, pred, IDS, method="normal")
    assert r.meta.status is ResultStatus.UNDEFINED
    assert r.z_reference.size == 0
