import numpy as np
import pytest
from scipy import stats

from mrsvalidate.analysis import method_comparison as mc
from mrsvalidate.results import ResultStatus


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


# ---------------------------------------------------------------------------
# Analytic + bootstrap reporting
# ---------------------------------------------------------------------------

def _ids(n):
    return np.array([f"s{i}" for i in range(n)])


def test_equivalence_test_1_matches_linregress_and_reports_both_cis():
    rng = np.random.default_rng(3)
    n = 40
    ref = rng.uniform(1, 10, n)
    cand = ref + 0.2 + rng.normal(scale=0.5, size=n)
    r = mc.equivalence_test_1(ref, cand, _ids(n), n_bootstrap=500, seed=0)
    fit = stats.linregress(ref - ref.mean(), cand - ref)
    assert r.alpha == pytest.approx(fit.intercept)
    assert r.beta == pytest.approx(fit.slope)
    assert r.alpha_se == pytest.approx(fit.intercept_stderr)
    t = stats.t.ppf(0.975, n - 2)
    assert r.alpha_ci_analytic == pytest.approx((r.alpha - t * r.alpha_se, r.alpha + t * r.alpha_se))
    assert r.alpha_ci_bootstrap[0] < r.alpha < r.alpha_ci_bootstrap[1]
    assert r.reject_null_analytic == (r.alpha_p_value < 0.05)
    assert r.reject_null_bootstrap == (not r.alpha_ci_bootstrap[0] <= 0 <= r.alpha_ci_bootstrap[1])
    assert r.analytic_bootstrap_concordant == (r.reject_null_analytic == r.reject_null_bootstrap)
    assert r.meta.random_seed == 0
    again = mc.equivalence_test_1(ref, cand, _ids(n), n_bootstrap=500, seed=0)
    assert again.alpha_ci_bootstrap == r.alpha_ci_bootstrap


def test_equivalence_test_1_bootstrap_and_analytic_agree_for_clear_bias():
    rng = np.random.default_rng(4)
    ref = rng.uniform(1, 10, 100)
    cand = ref + 1.0 + rng.normal(scale=0.3, size=100)
    r = mc.equivalence_test_1(ref, cand, _ids(100), n_bootstrap=500, seed=1)
    assert r.reject_null_analytic and r.reject_null_bootstrap
    assert r.analytic_bootstrap_concordant


def test_equivalence_test_1_constant_reference_undefined():
    ref = np.full(5, 3.0)
    r = mc.equivalence_test_1(ref, ref + np.arange(5.0), _ids(5))
    assert r.meta.status is ResultStatus.UNDEFINED


def test_equivalence_test_2_reports_both_cis():
    rng = np.random.default_rng(5)
    n = 200
    truth = rng.normal(scale=2.0, size=n)
    ref = truth + rng.normal(scale=1.0, size=n)
    cand = truth + rng.normal(scale=3.0, size=n)  # clearly unequal precision
    r = mc.equivalence_test_2(ref, cand, _ids(n), n_bootstrap=500, seed=0)
    fit = stats.linregress(cand + ref, cand - ref)
    assert r.beta == pytest.approx(fit.slope)
    assert r.beta_se == pytest.approx(fit.stderr)
    t = stats.t.ppf(0.975, n - 2)
    assert r.beta_ci_analytic == pytest.approx((r.beta - t * r.beta_se, r.beta + t * r.beta_se))
    assert r.reject_null_analytic and r.reject_null_bootstrap
    assert r.analytic_bootstrap_concordant


def test_equivalence_test_2_bias_correction_does_not_change_beta_or_p_value():
    rng = np.random.default_rng(6)
    n = 50
    ref = rng.uniform(1, 10, n)
    cand = ref + 2.0 + rng.normal(scale=0.4, size=n)
    plain = mc.equivalence_test_2(ref, cand, _ids(n), n_bootstrap=300, seed=2)
    corrected = mc.equivalence_test_2(ref, cand, _ids(n), bias_correction=2.0, n_bootstrap=300, seed=2)
    assert corrected.beta == pytest.approx(plain.beta, rel=1e-10, abs=1e-12)
    assert corrected.beta_se == pytest.approx(plain.beta_se, rel=1e-8)
    assert corrected.beta_p_value == pytest.approx(plain.beta_p_value, rel=1e-8)
    np.testing.assert_allclose(corrected.beta_ci_bootstrap, plain.beta_ci_bootstrap, rtol=1e-8, atol=1e-12)
    # only the intercept moves, by -c (1 - beta)
    assert corrected.alpha == pytest.approx(plain.alpha - 2.0 * (1 - plain.beta))


def test_equivalence_test_3_reports_jackknife_analytic_cis():
    rng = np.random.default_rng(7)
    n = 80
    truth = rng.uniform(1, 10, n)
    ref = truth + rng.normal(scale=0.3, size=n)
    cand = truth + rng.normal(scale=0.3, size=n)
    r = mc.equivalence_test_3(
        ref, cand, _ids(n), reference_error_variance=1.0, test_error_variance=1.0,
        n_bootstrap=500, seed=0,
    )
    # manual jackknife via the public deming()
    jk = np.array([
        [
            (fit := mc.deming(
                np.delete(ref, i), np.delete(cand, i), _ids(n - 1),
                reference_error_variance=1.0, test_error_variance=1.0,
            )).intercept,
            fit.slope,
        ]
        for i in range(n)
    ])
    se = np.sqrt((n - 1) / n * np.sum((jk - jk.mean(axis=0)) ** 2, axis=0))
    assert r.alpha_se_jackknife == pytest.approx(se[0])
    assert r.beta_se_jackknife == pytest.approx(se[1])
    t = stats.t.ppf(1 - 0.05 / 4, n - 2)
    assert r.beta_ci_analytic == pytest.approx((r.beta - t * se[1], r.beta + t * se[1]))
    assert not r.reject_joint_null
    assert not r.reject_joint_null_analytic
    assert r.analytic_bootstrap_concordant
    assert r.meta.configuration["decision_basis"] == "bootstrap"
    assert 0 < r.beta_p_value_analytic <= 1


def test_equivalence_test_3_analytic_rejects_for_clear_disagreement():
    rng = np.random.default_rng(2)
    ref = rng.uniform(1, 10, size=300)
    cand = 2.0 * ref + 5.0 + rng.normal(scale=0.1, size=300)
    r = mc.equivalence_test_3(
        ref, cand, _ids(300), reference_error_variance=1.0, test_error_variance=1.0,
        n_bootstrap=200, seed=0,
    )
    assert r.reject_joint_null and r.reject_joint_null_analytic
    assert r.analytic_bootstrap_concordant


def test_equivalence_test_3_bias_correction_shifts_only_intercept():
    rng = np.random.default_rng(8)
    n = 40
    ref = rng.uniform(1, 10, n)
    cand = ref + 1.5 + rng.normal(scale=0.3, size=n)
    kw = {"reference_error_variance": 1.0, "test_error_variance": 1.0, "n_bootstrap": 200, "seed": 4}
    plain = mc.equivalence_test_3(ref, cand, _ids(n), **kw)
    corrected = mc.equivalence_test_3(ref, cand, _ids(n), bias_correction=1.5, **kw)
    assert corrected.beta == pytest.approx(plain.beta)
    assert corrected.beta_se_jackknife == pytest.approx(plain.beta_se_jackknife)
    assert corrected.alpha == pytest.approx(plain.alpha - 1.5)
    assert corrected.alpha_se_jackknife == pytest.approx(plain.alpha_se_jackknife)
