import numpy as np
import pytest

from mrsvalidate.analysis import method_comparison as mc
from mrsvalidate.results import ResultStatus


def test_bland_altman_perfect_agreement():
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    cand = ref.copy()
    ids = np.array([f"s{i}" for i in range(5)])
    r = mc.bland_altman(ref, cand, ids)
    assert r.bias == pytest.approx(0.0)
    assert r.scale == pytest.approx(0.0)
    assert r.loa_low == pytest.approx(0.0)
    assert r.loa_high == pytest.approx(0.0)


def test_bland_altman_constant_bias():
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    cand = ref + 2.0
    ids = np.array([f"s{i}" for i in range(5)])
    r = mc.bland_altman(ref, cand, ids)
    assert r.bias == pytest.approx(2.0)
    assert r.scale == pytest.approx(0.0, abs=1e-9)


def test_bland_altman_x_axis_choice_changes_x_values_not_bias():
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    cand = ref + 1.0
    ids = np.array([f"s{i}" for i in range(4)])
    r_ref_axis = mc.bland_altman(ref, cand, ids, x_axis="reference")
    r_mean_axis = mc.bland_altman(ref, cand, ids, x_axis="mean")
    assert r_ref_axis.bias == pytest.approx(r_mean_axis.bias)
    np.testing.assert_allclose(r_ref_axis.x_values, ref)
    np.testing.assert_allclose(r_mean_axis.x_values, (ref + cand) / 2)


def test_bland_altman_conditional_reports_coefficients():
    rng = np.random.default_rng(0)
    ref = np.linspace(1, 100, 200)
    cand = ref * 1.05 + rng.normal(scale=1.0 + 0.05 * ref)
    ids = np.array([f"s{i}" for i in range(200)])
    r = mc.bland_altman(ref, cand, ids, conditional=True)
    assert r.conditional_bias_coefficients is not None
    assert r.conditional_scale_coefficients is not None
    _intercept, slope = r.conditional_bias_coefficients
    assert slope == pytest.approx(0.05, abs=0.02)


def test_passing_bablok_recovers_exact_noiseless_slope_and_intercept():
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    cand = 2.0 * ref + 1.0
    ids = np.array([f"s{i}" for i in range(5)])
    r = mc.passing_bablok(ref, cand, ids, n_bootstrap=50, n_cusum_permutations=50, seed=0)
    assert r.slope == pytest.approx(2.0)
    assert r.intercept == pytest.approx(1.0)


def test_passing_bablok_recovers_negative_slope_noiseless():
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    cand = -2.0 * ref + 3.0
    ids = np.array([f"s{i}" for i in range(6)])
    r = mc.passing_bablok(ref, cand, ids, n_bootstrap=50, n_cusum_permutations=50, seed=0)
    assert r.slope == pytest.approx(-2.0)
    assert r.intercept == pytest.approx(3.0)


def test_passing_bablok_identity_gives_unit_slope_zero_intercept():
    ref = np.array([1.0, 5.0, 2.0, 8.0, 3.0, 9.0, 4.0])
    cand = ref.copy()
    ids = np.array([f"s{i}" for i in range(7)])
    r = mc.passing_bablok(ref, cand, ids, n_bootstrap=50, n_cusum_permutations=50, seed=0)
    assert r.slope == pytest.approx(1.0)
    assert r.intercept == pytest.approx(0.0, abs=1e-9)


def test_passing_bablok_excludes_equal_reference_pairs():
    ref = np.array([1.0, 1.0, 2.0, 3.0])  # first pair (0,1) has equal reference
    cand = np.array([1.0, 1.5, 2.0, 3.0])
    ids = np.array(["a", "b", "c", "d"])
    r = mc.passing_bablok(ref, cand, ids, n_bootstrap=20, n_cusum_permutations=20, seed=0)
    assert r.n_excluded_equal_reference_pairs == 1


def test_deming_noiseless_recovers_exact_slope_regardless_of_lambda():
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    test_vals = 2.0 * ref + 1.0
    ids = np.array([f"s{i}" for i in range(5)])
    for rv, tv in [(1.0, 1.0), (4.0, 1.0), (1.0, 4.0)]:
        r = mc.deming(ref, test_vals, ids, reference_error_variance=rv, test_error_variance=tv)
        assert r.slope == pytest.approx(2.0, abs=1e-6)
        assert r.intercept == pytest.approx(1.0, abs=1e-6)
        assert r.variance_ratio_value == pytest.approx(rv / tv)


def test_deming_requires_positive_variances():
    ref = np.array([1.0, 2.0, 3.0])
    test_vals = np.array([1.0, 2.0, 3.0])
    ids = np.array(["a", "b", "c"])
    r = mc.deming(ref, test_vals, ids, reference_error_variance=-1.0, test_error_variance=1.0)
    assert r.meta.status is ResultStatus.INVALID_INPUT


def test_deming_variance_ratio_convention_recorded():
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    test_vals = np.array([1.1, 2.1, 2.9, 4.2])
    ids = np.array(["a", "b", "c", "d"])
    r = mc.deming(ref, test_vals, ids, reference_error_variance=2.0, test_error_variance=1.0)
    assert r.variance_ratio_convention == "lambda = var(reference) / var(test)"
    assert r.variance_ratio_value == pytest.approx(2.0)
