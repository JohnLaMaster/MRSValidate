import numpy as np
import pytest

from mrsvalidate.analysis import agreement
from mrsvalidate.analysis import multivariate as mv
from mrsvalidate.results import ResultStatus


def test_multivariate_ccc_perfect_agreement_equals_one():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(50, 1, 3))
    y = np.concatenate([base, base], axis=1)  # 2 identical methods, 3 variables
    ids = np.array([f"s{i}" for i in range(50)])
    r = mv.multivariate_ccc(y, ids)
    assert r.rho_c_multivariate == pytest.approx(1.0, abs=1e-9)


def test_multivariate_ccc_two_methods_one_variable_matches_lin_ccc():
    rng = np.random.default_rng(1)
    x1 = rng.normal(size=200)
    x2 = 0.5 * x1 + rng.normal(scale=2.0, size=200) + 3.0  # biased + noisy
    y = np.stack([x1, x2], axis=1)[:, :, np.newaxis]  # (n, d=2, p=1)
    ids = np.array([f"s{i}" for i in range(200)])

    mv_result = mv.multivariate_ccc(y, ids)
    lin_result = agreement.lin_ccc(x1, x2, ids)
    assert mv_result.rho_c_multivariate == pytest.approx(lin_result.rho_c, abs=1e-9)


def test_multivariate_ccc_insufficient_for_single_method():
    y = np.zeros((10, 1, 2))
    ids = np.array([f"s{i}" for i in range(10)])
    r = mv.multivariate_ccc(y, ids)
    assert r.meta.status is ResultStatus.INSUFFICIENT_DATA


def test_multivariate_ccc_excludes_nan_subjects():
    rng = np.random.default_rng(2)
    y = rng.normal(size=(20, 2, 2))
    y[0, 0, 0] = np.nan
    ids = np.array([f"s{i}" for i in range(20)])
    r = mv.multivariate_ccc(y, ids)
    assert r.meta.n_omitted == 1
