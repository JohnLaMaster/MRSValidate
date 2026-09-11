import numpy as np
import pytest

from mrsvalidate.analysis import mixed_effects as me
from mrsvalidate.results import ResultStatus


def _make_dataset(rng, n_metabolites=6, n_clusters=40, bias=0.0, slope_dev=0.0):
    rows = []
    for c in range(n_clusters):
        cluster_id = f"c{c}"
        for m in range(n_metabolites):
            metabolite_id = f"m{m}"
            y = rng.uniform(1.0, 10.0)
            x = y * (1.0 + slope_dev) + bias + rng.normal(scale=0.2)
            rows.append((y, x, x - y, metabolite_id, cluster_id, f"{cluster_id}_{metabolite_id}"))
    y = np.array([r[0] for r in rows])
    x = np.array([r[1] for r in rows])
    d = np.array([r[2] for r in rows])
    metab = np.array([r[3] for r in rows])
    cluster = np.array([r[4] for r in rows])
    ids = np.array([r[5] for r in rows])
    return y, x, d, metab, cluster, ids


def test_mixed_test1_near_zero_alpha_for_unbiased_data():
    rng = np.random.default_rng(0)
    y, x, d, metab, cluster, ids = _make_dataset(rng, bias=0.0)
    r = me.mixed_equivalence_test_1(d, y, metab, cluster, ids)
    assert r.meta.status is ResultStatus.VALID
    assert abs(r.alpha) < 0.3


def test_mixed_test1_detects_model_level_bias():
    rng = np.random.default_rng(1)
    y, x, d, metab, cluster, ids = _make_dataset(rng, bias=3.0)
    r = me.mixed_equivalence_test_1(d, y, metab, cluster, ids)
    assert r.alpha == pytest.approx(3.0, abs=0.3)
    assert r.alpha_p_value < 0.01


def test_mixed_test1_insufficient_data_single_metabolite():
    y = np.array([1.0, 2.0, 3.0])
    x = y.copy()
    d = x - y
    metab = np.array(["m0", "m0", "m0"])
    cluster = np.array(["c0", "c1", "c2"])
    ids = np.array(["a", "b", "c"])
    r = me.mixed_equivalence_test_1(d, y, metab, cluster, ids)
    assert r.meta.status is ResultStatus.INSUFFICIENT_DATA


def test_mixed_test2_beta_near_zero_for_equal_variance_no_slope_dev():
    rng = np.random.default_rng(2)
    y, x, d, metab, cluster, ids = _make_dataset(rng, bias=0.0, slope_dev=0.0)
    r = me.mixed_equivalence_test_2(d, y, x, metab, cluster, ids)
    assert r.meta.status is ResultStatus.VALID
    assert abs(r.beta) < 0.1


def test_mixed_test3_does_not_reject_for_near_identity():
    rng = np.random.default_rng(3)
    y, x, d, metab, cluster, ids = _make_dataset(rng, bias=0.0, slope_dev=0.0)
    r = me.mixed_equivalence_test_3(
        y, x, cluster, ids,
        reference_error_variance=1.0, test_error_variance=1.0,
        n_bootstrap=200, seed=0,
    )
    assert r.meta.status is ResultStatus.VALID
    assert not r.reject_joint_null


def test_mixed_test3_rejects_for_clear_disagreement():
    rng = np.random.default_rng(4)
    y, x, d, metab, cluster, ids = _make_dataset(rng, bias=3.0, slope_dev=0.5)
    r = me.mixed_equivalence_test_3(
        y, x, cluster, ids,
        reference_error_variance=1.0, test_error_variance=1.0,
        n_bootstrap=200, seed=0,
    )
    assert r.reject_joint_null
