import numpy as np
import pytest

from mrsvalidate.analysis import method_comparison as mc


def test_equivalence_test_1_zero_bias_for_identity():
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    cand = ref.copy()
    ids = np.array([f"s{i}" for i in range(6)])
    r = mc.equivalence_test_1(ref, cand, ids)
    assert r.alpha == pytest.approx(0.0, abs=1e-9)


def test_equivalence_test_1_detects_constant_bias():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=300)
    cand = ref + 2.0 + rng.normal(scale=0.01, size=300)
    ids = np.array([f"s{i}" for i in range(300)])
    r = mc.equivalence_test_1(ref, cand, ids)
    assert r.alpha == pytest.approx(2.0, abs=0.05)
    assert r.alpha_p_value < 0.001


def test_equivalence_test_2_near_zero_for_equal_variance():
    rng = np.random.default_rng(1)
    n = 500
    truth = rng.normal(scale=5.0, size=n)
    ref = truth + rng.normal(scale=1.0, size=n)
    cand = truth + rng.normal(scale=1.0, size=n)  # same error variance as reference
    ids = np.array([f"s{i}" for i in range(n)])
    r = mc.equivalence_test_2(ref, cand, ids)
    assert abs(r.beta) < 0.1


def test_equivalence_test_2_applies_bias_correction():
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    cand = ref + 5.0
    ids = np.array([f"s{i}" for i in range(4)])
    r = mc.equivalence_test_2(ref, cand, ids, bias_correction=5.0)
    assert r.bias_correction_applied == pytest.approx(5.0)


def test_equivalence_test_3_does_not_reject_for_identity():
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    cand = ref.copy()
    ids = np.array([f"s{i}" for i in range(8)])
    r = mc.equivalence_test_3(
        ref, cand, ids, reference_error_variance=1.0, test_error_variance=1.0,
        n_bootstrap=200, seed=0,
    )
    assert r.alpha == pytest.approx(0.0, abs=1e-9)
    assert r.beta == pytest.approx(1.0, abs=1e-9)
    assert not r.reject_joint_null


def test_equivalence_test_3_rejects_for_clear_disagreement():
    rng = np.random.default_rng(2)
    ref = rng.uniform(1, 10, size=300)
    cand = 2.0 * ref + 5.0 + rng.normal(scale=0.1, size=300)
    ids = np.array([f"s{i}" for i in range(300)])
    r = mc.equivalence_test_3(
        ref, cand, ids, reference_error_variance=1.0, test_error_variance=1.0,
        n_bootstrap=200, seed=0,
    )
    assert r.reject_joint_null
