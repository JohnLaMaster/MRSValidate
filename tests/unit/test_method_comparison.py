import numpy as np
import pytest
from scipy import stats

from mrsvalidate.analysis import method_comparison as mc
from mrsvalidate.results import ResultStatus


def _ids(n):
    return np.array([f"s{i}" for i in range(n)])


# ---------------------------------------------------------------------------
# Bland-Altman
# ---------------------------------------------------------------------------


def test_bland_altman_perfect_agreement():
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    cand = ref.copy()
    ids = _ids(5)
    r = mc.bland_altman(ref, cand, ids, n_bootstrap=50, seed=0)
    assert r.bias == pytest.approx(0.0)
    assert r.scale == pytest.approx(0.0)
    assert r.loa_low == pytest.approx(0.0)
    assert r.loa_high == pytest.approx(0.0)


def test_bland_altman_constant_bias():
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    cand = ref + 2.0
    ids = _ids(5)
    r = mc.bland_altman(ref, cand, ids, n_bootstrap=50, seed=0)
    assert r.bias == pytest.approx(2.0)
    assert r.scale == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(r.conditional_loa_low, 2.0, atol=1e-9)
    np.testing.assert_allclose(r.conditional_loa_high, 2.0, atol=1e-9)


def test_bland_altman_x_axis_choice_changes_x_values_not_bias():
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    cand = ref + 1.0
    ids = _ids(4)
    r_ref_axis = mc.bland_altman(ref, cand, ids, x_axis="reference", n_bootstrap=50, seed=0)
    r_mean_axis = mc.bland_altman(ref, cand, ids, x_axis="mean", n_bootstrap=50, seed=0)
    assert r_ref_axis.bias == pytest.approx(r_mean_axis.bias)
    np.testing.assert_allclose(r_ref_axis.x_values, ref)
    np.testing.assert_allclose(r_mean_axis.x_values, (ref + cand) / 2)


def test_bland_altman_new_defaults_are_manuscript_choices():
    rng = np.random.default_rng(3)
    ref = rng.uniform(1, 10, 40)
    cand = ref + rng.normal(scale=0.3, size=40)
    r = mc.bland_altman(ref, cand, _ids(40), n_bootstrap=100, seed=0)
    assert r.location == "median"
    assert r.scale_method == "robust_sd"
    assert r.loa_method == "normal"
    assert r.conditional is True
    assert r.ci_method == "bootstrap"
    cfg = r.meta.configuration
    assert cfg["robust_sd_consistency_constant"] == pytest.approx(1.0 / stats.norm.ppf(0.75))
    assert cfg["conditional_bias_model"] == "theil_sen"
    assert r.meta.random_seed == 0


def test_bland_altman_conditional_reports_coefficients():
    rng = np.random.default_rng(0)
    ref = np.linspace(1, 100, 200)
    cand = ref * 1.05 + rng.normal(scale=1.0 + 0.05 * ref)
    ids = _ids(200)
    for location, scale in [("median", "robust_sd"), ("mean", "sd")]:
        r = mc.bland_altman(
            ref, cand, ids, location=location, scale=scale, conditional=True, ci_method="none"
        )
        assert r.conditional_bias_coefficients is not None
        assert r.conditional_scale_coefficients is not None
        _intercept, slope = r.conditional_bias_coefficients
        assert slope == pytest.approx(0.05, abs=0.02)
        # SD(c) = 1 + 0.05 c is recovered on the SD scale by both families.
        s0, s1 = r.conditional_scale_coefficients
        assert s1 == pytest.approx(0.05, abs=0.02)
        assert s0 == pytest.approx(1.0, abs=0.6)


def test_bland_altman_parametric_conditional_matches_ols():
    rng = np.random.default_rng(4)
    ref = rng.uniform(1, 10, 50)
    cand = ref + 0.1 * ref + rng.normal(scale=0.2, size=50)
    r = mc.bland_altman(
        ref, cand, _ids(50), location="mean", scale="sd", conditional=True, ci_method="none"
    )
    d = cand - ref
    b1, b0 = np.polyfit(ref, d, 1)
    s1, s0 = np.polyfit(ref, np.abs(d - (b0 + b1 * ref)), 1)
    k = np.sqrt(np.pi / 2)
    np.testing.assert_allclose(r.conditional_bias_coefficients, (b0, b1))
    np.testing.assert_allclose(r.conditional_scale_coefficients, (s0 * k, s1 * k))
    np.testing.assert_allclose(r.conditional_loa_low, b0 + b1 * ref - 1.96 * k * (s0 + s1 * ref))
    np.testing.assert_allclose(r.conditional_loa_high, b0 + b1 * ref + 1.96 * k * (s0 + s1 * ref))


def test_bland_altman_robust_conditional_matches_theil_sen():
    rng = np.random.default_rng(5)
    ref = rng.uniform(1, 10, 50)
    cand = ref + rng.standard_t(3, size=50) * 0.2
    r = mc.bland_altman(ref, cand, _ids(50), ci_method="none")
    d = cand - ref
    b1, b0, _, _ = stats.theilslopes(d, ref)
    s1, s0, _, _ = stats.theilslopes(np.abs(d - (b0 + b1 * ref)), ref)
    k = 1.0 / stats.norm.ppf(0.75)
    np.testing.assert_allclose(r.conditional_bias_coefficients, (b0, b1))
    np.testing.assert_allclose(r.conditional_scale_coefficients, (s0 * k, s1 * k))


def test_bland_altman_robust_sd_is_normal_consistent():
    rng = np.random.default_rng(6)
    n = 20000
    ref = rng.uniform(1, 10, n)
    cand = ref + rng.normal(size=n)
    r = mc.bland_altman(ref, cand, _ids(n), conditional=False, ci_method="none")
    d = cand - ref
    mad = np.median(np.abs(d - np.median(d)))
    assert r.scale == pytest.approx(1.4826 * mad, rel=1e-4)
    assert r.scale == pytest.approx(1.0, abs=0.03)
    coverage = np.mean((d >= r.loa_low) & (d <= r.loa_high))
    assert coverage == pytest.approx(0.95, abs=0.01)


def test_bland_altman_quantile_loa():
    rng = np.random.default_rng(7)
    ref = rng.uniform(1, 10, 101)
    cand = ref + rng.exponential(size=101)
    r = mc.bland_altman(ref, cand, _ids(101), loa_method="quantile", conditional=False, ci_method="none")
    d = cand - ref
    assert r.loa_low == pytest.approx(np.quantile(d, 0.025))
    assert r.loa_high == pytest.approx(np.quantile(d, 0.975))
    assert r.meta.configuration["quantile_method"] == "linear"
    assert r.conditional_loa_low is None


def test_bland_altman_quantile_with_conditional_raises():
    ref = np.arange(1.0, 11.0)
    with pytest.raises(ValueError, match="quantile"):
        mc.bland_altman(ref, ref + 1, _ids(10), loa_method="quantile")


def test_bland_altman_analytic_ci_matches_bland_altman_1999():
    rng = np.random.default_rng(8)
    n = 30
    ref = rng.uniform(1, 10, n)
    cand = ref + rng.normal(scale=0.5, size=n)
    r = mc.bland_altman(
        ref, cand, _ids(n), location="mean", scale="sd", conditional=False, ci_method="analytic"
    )
    d = cand - ref
    s = np.std(d, ddof=1)
    t = stats.t.ppf(0.975, n - 1)
    np.testing.assert_allclose(r.bias_ci, (d.mean() - t * s / np.sqrt(n), d.mean() + t * s / np.sqrt(n)))
    half = t * s * np.sqrt(1 / n + 1.96**2 / (2 * (n - 1)))
    np.testing.assert_allclose(r.loa_low_ci, (r.loa_low - half, r.loa_low + half))
    np.testing.assert_allclose(r.loa_high_ci, (r.loa_high - half, r.loa_high + half))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"location": "median", "scale": "sd", "conditional": False},
        {"location": "mean", "scale": "robust_sd", "conditional": False},
        {"location": "mean", "scale": "sd", "loa_method": "quantile", "conditional": False},
    ],
)
def test_bland_altman_analytic_ci_rejects_non_parametric_settings(kwargs):
    ref = np.arange(1.0, 11.0)
    with pytest.raises(ValueError, match="bootstrap"):
        mc.bland_altman(ref, ref + 1, _ids(10), ci_method="analytic", **kwargs)


def test_bland_altman_ci_none_gives_nan():
    ref = np.arange(1.0, 11.0)
    r = mc.bland_altman(ref, ref + np.sin(ref), _ids(10), ci_method="none")
    assert all(np.isnan(r.bias_ci)) and all(np.isnan(r.loa_low_ci)) and all(np.isnan(r.loa_high_ci))
    assert r.ci_grid is None and r.conditional_bias_band is None


def test_bland_altman_bootstrap_ci_and_bands():
    rng = np.random.default_rng(9)
    n = 60
    ref = rng.uniform(1, 10, n)
    cand = 1.05 * ref + rng.normal(scale=0.1 + 0.05 * ref)
    r = mc.bland_altman(ref, cand, _ids(n), n_bootstrap=300, seed=1)
    again = mc.bland_altman(ref, cand, _ids(n), n_bootstrap=300, seed=1)
    assert r.bias_ci == again.bias_ci
    assert r.bias_ci[0] <= r.bias <= r.bias_ci[1]
    assert r.loa_low_ci[0] <= r.loa_low <= r.loa_low_ci[1]
    assert r.loa_high_ci[0] <= r.loa_high <= r.loa_high_ci[1]
    assert r.ci_grid.shape == (50,)
    assert r.ci_grid[0] == pytest.approx(ref.min()) and r.ci_grid[-1] == pytest.approx(ref.max())
    b0, b1 = r.conditional_bias_coefficients
    point_bias = b0 + b1 * r.ci_grid
    for band in (r.conditional_bias_band, r.conditional_loa_low_band, r.conditional_loa_high_band):
        low, high = band
        assert low.shape == high.shape == (50,)
        assert np.all(low <= high)
    low, high = r.conditional_bias_band
    assert np.mean((low <= point_bias) & (point_bias <= high)) > 0.9
    assert r.meta.configuration["n_bootstrap"] == 300


def test_bland_altman_bootstrap_bias_ci_coverage():
    # Paired percentile bootstrap of the median bias, n = 50, normal errors.
    rng = np.random.default_rng(10)
    n, reps, covered = 50, 150, 0
    for _ in range(reps):
        ref = rng.uniform(1, 10, n)
        cand = ref + 0.5 + rng.normal(size=n)
        r = mc.bland_altman(ref, cand, _ids(n), conditional=False, n_bootstrap=300, seed=rng.integers(1 << 31))
        covered += r.bias_ci[0] <= 0.5 <= r.bias_ci[1]
    assert 0.88 <= covered / reps <= 0.99


def test_bland_altman_conditional_constant_axis_is_undefined():
    ref = np.array([2.0, 2.0, 2.0])
    r = mc.bland_altman(ref, np.array([2.1, 1.9, 2.0]), _ids(3))
    assert r.meta.status is ResultStatus.UNDEFINED
    ok = mc.bland_altman(ref, np.array([2.1, 1.9, 2.0]), _ids(3), conditional=False, ci_method="none")
    assert ok.meta.status is ResultStatus.VALID


def test_bland_altman_insufficient_data():
    r = mc.bland_altman(np.array([1.0]), np.array([1.0]), _ids(1))
    assert r.meta.status is ResultStatus.INSUFFICIENT_DATA
    assert np.isnan(r.bias)


# ---------------------------------------------------------------------------
# Passing-Bablok
# ---------------------------------------------------------------------------


def test_passing_bablok_recovers_exact_noiseless_slope_and_intercept():
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    cand = 2.0 * ref + 1.0
    ids = _ids(5)
    for estimator in ("passing_bablok_1983", "manuscript_literal"):
        r = mc.passing_bablok(ref, cand, ids, estimator=estimator, n_bootstrap=50, n_cusum_permutations=50, seed=0)
        assert r.slope == pytest.approx(2.0)
        assert r.intercept == pytest.approx(1.0)


def test_passing_bablok_recovers_negative_slope_noiseless():
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    cand = -2.0 * ref + 3.0
    ids = _ids(6)
    r = mc.passing_bablok(
        ref, cand, ids, estimator="manuscript_literal", n_bootstrap=50, n_cusum_permutations=50, seed=0
    )
    assert r.slope == pytest.approx(-2.0)
    assert r.intercept == pytest.approx(3.0)


def test_passing_bablok_1983_negative_slope_clamps_with_warning():
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    cand = -2.0 * ref + 3.0
    r = mc.passing_bablok(ref, cand, _ids(6))
    assert r.slope == pytest.approx(-2.0)
    assert any("clamped" in w for w in r.meta.warnings)


def test_passing_bablok_identity_gives_unit_slope_zero_intercept():
    ref = np.array([1.0, 5.0, 2.0, 8.0, 3.0, 9.0, 4.0])
    cand = ref.copy()
    ids = _ids(7)
    r = mc.passing_bablok(ref, cand, ids, n_bootstrap=50, n_cusum_permutations=50, seed=0)
    assert r.slope == pytest.approx(1.0)
    assert r.intercept == pytest.approx(0.0, abs=1e-9)


def test_passing_bablok_excludes_equal_reference_pairs():
    ref = np.array([1.0, 1.0, 2.0, 3.0])  # first pair (0,1) has equal reference
    cand = np.array([1.0, 1.5, 2.0, 3.0])
    ids = np.array(["a", "b", "c", "d"])
    r = mc.passing_bablok(
        ref, cand, ids, estimator="manuscript_literal", n_bootstrap=20, n_cusum_permutations=20, seed=0
    )
    assert r.n_excluded_equal_reference_pairs == 1
    assert r.n_pairwise_slopes == 5


def test_passing_bablok_1983_keeps_infinite_slopes_and_drops_identical_points():
    ref = np.array([1.0, 1.0, 2.0, 3.0, 3.0])
    cand = np.array([1.0, 1.5, 2.0, 3.0, 3.0])  # (0,1): +inf; (3,4): identical
    r = mc.passing_bablok(ref, cand, _ids(5))
    assert r.estimator == "passing_bablok_1983"
    assert r.n_infinite_slopes == 1
    assert r.n_excluded_equal_reference_pairs == 1
    assert r.n_pairwise_slopes == 9
    assert r.meta.configuration["n_infinite_slopes"] == 1


def test_passing_bablok_1983_discards_minus_one_slopes_and_shifts():
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    cand = np.array([2.0, 1.0, 3.5, 4.0, 2.0])
    r = mc.passing_bablok(ref, cand, _ids(5))
    i, j = np.triu_indices(5, 1)
    s = (cand[j] - cand[i]) / (ref[j] - ref[i])
    s = np.sort(s[s != -1.0])
    k = int(np.sum(s < -1))
    n_s = len(s)
    assert r.n_excluded_minus_one_slopes == int(np.sum(((cand[j] - cand[i]) / (ref[j] - ref[i])) == -1.0)) > 0
    assert r.shift_k == k
    assert r.n_pairwise_slopes == n_s
    if n_s % 2:
        expected = s[(n_s + 1) // 2 + k - 1]
    else:
        expected = 0.5 * (s[n_s // 2 + k - 1] + s[n_s // 2 + k])
    assert r.slope == pytest.approx(expected)
    assert r.intercept == pytest.approx(np.median(cand - expected * ref))


def test_passing_bablok_1983_even_number_of_slopes_averages():
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    cand = np.array([1.2, 1.9, 3.3, 3.8, 5.4, 5.7, 7.5, 7.7, 9.6])
    # duplicate one point so that one pair is identical -> N = 36 - 1 + 9 = 44 (even)
    ref2 = np.append(ref, ref[0])
    cand2 = np.append(cand, cand[0])
    r = mc.passing_bablok(ref2, cand2, _ids(10))
    assert r.n_pairwise_slopes % 2 == 0
    i, j = np.triu_indices(10, 1)
    dr, dc = ref2[j] - ref2[i], cand2[j] - cand2[i]
    keep = ~((dr == 0) & (dc == 0))
    s = np.sort(dc[keep] / dr[keep])
    s = s[s != -1]
    k = int(np.sum(s < -1))
    n_s = len(s)
    assert r.slope == pytest.approx(0.5 * (s[n_s // 2 + k - 1] + s[n_s // 2 + k]))


def test_passing_bablok_analytic_ci_formula():
    rng = np.random.default_rng(11)
    n = 25
    truth = rng.uniform(1, 10, n)
    ref = truth + rng.normal(scale=0.3, size=n)
    cand = 1.2 * truth + rng.normal(scale=0.36, size=n)
    r = mc.passing_bablok(ref, cand, _ids(n))
    assert r.ci_method == "analytic"
    i, j = np.triu_indices(n, 1)
    s = np.sort((cand[j] - cand[i]) / (ref[j] - ref[i]))
    k = int(np.sum(s < -1))
    n_s = len(s)
    c = stats.norm.ppf(0.975) * np.sqrt(n * (n - 1) * (2 * n + 5) / 18)
    m1 = int(np.floor((n_s - c) / 2 + 0.5))
    m2 = n_s - m1 + 1
    b_lo, b_hi = s[m1 + k - 1], s[m2 + k - 1]
    assert r.slope_ci == pytest.approx((b_lo, b_hi))
    expected_int = sorted([np.median(cand - b_hi * ref), np.median(cand - b_lo * ref)])
    assert r.intercept_ci == pytest.approx(tuple(expected_int))
    assert r.slope_ci[0] <= r.slope <= r.slope_ci[1]


def test_passing_bablok_analytic_ci_clamps_for_tiny_samples():
    ref = np.array([1.0, 2.0, 3.0])
    cand = np.array([1.1, 2.3, 2.9])
    r = mc.passing_bablok(ref, cand, _ids(3))
    assert r.meta.status is ResultStatus.VALID
    assert any("clamped" in w for w in r.meta.warnings)


def test_passing_bablok_bootstrap_ci_is_seeded():
    rng = np.random.default_rng(12)
    ref = rng.uniform(1, 10, 30)
    cand = ref + rng.normal(scale=0.3, size=30)
    a = mc.passing_bablok(ref, cand, _ids(30), ci_method="bootstrap", n_bootstrap=200, seed=3)
    b = mc.passing_bablok(ref, cand, _ids(30), ci_method="bootstrap", n_bootstrap=200, seed=3)
    assert a.ci_method == "bootstrap"
    assert a.slope_ci == b.slope_ci
    assert a.slope_ci[0] <= a.slope <= a.slope_ci[1]
    assert a.meta.random_seed == 3


def test_passing_bablok_cusum_1983_statistic():
    rng = np.random.default_rng(13)
    n = 40
    ref = rng.uniform(1, 10, n)
    cand = ref + rng.normal(scale=0.3, size=n)
    r = mc.passing_bablok(ref, cand, _ids(n), cusum_method="passing_bablok_1983")
    assert r.cusum_method == "passing_bablok_1983"
    e = (cand - r.slope * ref) - r.intercept
    pos, neg = e > 1e-9, e < -1e-9
    n_pos, n_neg = pos.sum(), neg.sum()
    score = np.where(pos, np.sqrt(n_neg / n_pos), np.where(neg, -np.sqrt(n_pos / n_neg), 0.0))
    assert score.sum() == pytest.approx(0.0, abs=1e-9)
    order = np.argsort((ref + r.slope * (cand - r.intercept)) / np.sqrt(1 + r.slope**2), kind="stable")
    h = np.max(np.abs(np.cumsum(score[order]))) / np.sqrt(n_pos + n_neg + 1)
    assert r.cusum_statistic == pytest.approx(h)
    assert r.cusum_p_value == pytest.approx(stats.kstwobign.sf(h))
    assert r.cusum_critical_value == pytest.approx(1.3581, abs=1e-3)


@pytest.mark.parametrize("cusum_method", ["bootstrap", "passing_bablok_1983"])
def test_passing_bablok_cusum_degenerate_for_noiseless_data(cusum_method):
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    r = mc.passing_bablok(ref, 2.0 * ref + 1.0, _ids(5), cusum_method=cusum_method, seed=0)
    assert np.isnan(r.cusum_statistic)
    assert np.isnan(r.cusum_p_value)
    assert not r.cusum_reject_linearity
    assert any("CUSUM" in w for w in r.meta.warnings)


def test_passing_bablok_permutation_cusum_option():
    rng = np.random.default_rng(14)
    ref = rng.uniform(1, 10, 30)
    cand = ref + rng.normal(scale=0.3, size=30)
    r = mc.passing_bablok(ref, cand, _ids(30), cusum_method="permutation", n_cusum_permutations=200, seed=0)
    assert r.cusum_method == "permutation"
    assert r.cusum_statistic == r.cusum_max_abs
    assert 0 < r.cusum_p_value <= 1
    assert np.isnan(r.cusum_critical_value)


def test_passing_bablok_invalid_option_raises():
    ref = np.arange(1.0, 6.0)
    with pytest.raises(ValueError):
        mc.passing_bablok(ref, ref, _ids(5), estimator="bogus")
    with pytest.raises(ValueError):
        mc.passing_bablok(ref, ref, _ids(5), cusum_method="bogus")
    with pytest.raises(ValueError):
        mc.passing_bablok(ref, ref, _ids(5), cusum_method="bootstrap", n_cusum_bootstrap=0)
    with pytest.raises(ValueError):
        mc.passing_bablok(ref, ref, _ids(5), cusum_bootstrap_variance_ratio=0.0)


def _noisy_linear(seed, n=40):
    rng = np.random.default_rng(seed)
    truth = rng.uniform(1, 10, n)
    return truth + rng.normal(scale=0.3, size=n), 0.5 + 1.1 * truth + rng.normal(scale=0.33, size=n)


def test_passing_bablok_bootstrap_cusum_records_configuration():
    ref, cand = _noisy_linear(21)
    assert mc.passing_bablok(ref, cand, _ids(40), n_cusum_bootstrap=99, seed=0).cusum_method == "bootstrap"  # default
    r = mc.passing_bablok(ref, cand, _ids(40), cusum_method="bootstrap", n_cusum_bootstrap=199, seed=5)
    assert r.cusum_method == "bootstrap"
    cfg = r.meta.configuration
    assert cfg["cusum_method"] == "bootstrap"
    assert cfg["cusum_p_value_method"] == "wild_bootstrap_rademacher_refit"
    assert cfg["n_cusum_bootstrap"] == 199
    assert cfg["n_cusum_bootstrap_valid"] == 199
    assert cfg["cusum_bootstrap_line_reestimated_per_replicate"] is True
    assert cfg["cusum_bootstrap_variance_ratio"] is None
    assert cfg["cusum_bootstrap_residual_direction"] == "perpendicular"
    assert cfg["n_cusum_permutations"] is None
    # Same observed statistic as the classical test; only the calibration differs.
    classical = mc.passing_bablok(ref, cand, _ids(40), cusum_method="passing_bablok_1983")
    assert r.cusum_statistic == classical.cusum_statistic
    assert r.cusum_max_abs == classical.cusum_max_abs
    assert classical.meta.configuration["n_cusum_bootstrap"] is None
    assert 1 / 200 <= r.cusum_p_value <= 1
    assert np.isfinite(r.cusum_critical_value)
    # The fitted-line effect: bootstrap critical value far below Kolmogorov's 1.358.
    assert r.cusum_critical_value < classical.cusum_critical_value
    assert r.cusum_reject_linearity == (r.cusum_p_value <= 0.05)


def test_passing_bablok_bootstrap_cusum_p_value_formula_and_seeding():
    ref, cand = _noisy_linear(22)
    kw = {"cusum_method": "bootstrap", "n_cusum_bootstrap": 99, "seed": 11}
    a = mc.passing_bablok(ref, cand, _ids(40), **kw)
    b = mc.passing_bablok(ref, cand, _ids(40), **kw)
    assert a.cusum_p_value == b.cusum_p_value
    assert a.cusum_critical_value == b.cusum_critical_value
    # Analytic CI consumes no random numbers, so the CUSUM bootstrap sees a fresh rng.
    h_star = mc._pb_cusum_wild_bootstrap(
        ref, cand, a.intercept, a.slope, "passing_bablok_1983", 99, np.random.default_rng(11)
    )
    assert np.all(np.isfinite(h_star))
    h = a.cusum_statistic
    assert a.cusum_p_value == pytest.approx((1 + np.sum(h_star >= h * (1 - 1e-9))) / 100)
    assert a.cusum_critical_value == pytest.approx(np.quantile(h_star, 0.95))


class _AllMinusOne:
    """Stand-in rng whose Rademacher draws are all -1 (pure reflection)."""

    def integers(self, low, high, size):
        return np.zeros(size, dtype=np.int64)


def test_passing_bablok_bootstrap_replicate_reflects_across_fitted_line():
    ref, cand = _noisy_linear(23)
    fit = mc.passing_bablok(ref, cand, _ids(40), cusum_method="passing_bablok_1983")
    b0, b1 = fit.intercept, fit.slope
    h_star = mc._pb_cusum_wild_bootstrap(ref, cand, b0, b1, "passing_bablok_1983", 1, _AllMinusOne())
    # Mirror image of every point in the fitted line, then a fresh PB fit.
    e_perp = (cand - b0 - b1 * ref) / np.sqrt(1 + b1**2)
    nu = np.array([-b1, 1.0]) / np.sqrt(1 + b1**2)
    ref_m, cand_m = ref - 2 * e_perp * nu[0], cand - 2 * e_perp * nu[1]
    np.testing.assert_allclose((cand_m - b0 - b1 * ref_m) / np.sqrt(1 + b1**2), -e_perp, atol=1e-12)
    refit = mc.passing_bablok(ref_m, cand_m, _ids(40), cusum_method="passing_bablok_1983")
    assert h_star[0] == pytest.approx(refit.cusum_statistic)


def test_passing_bablok_bootstrap_cusum_skips_degenerate_replicates(monkeypatch):
    ref, cand = _noisy_linear(24)

    def few_valid(*args):
        out = np.full(args[5], np.nan)
        out[:30] = 0.5
        return out

    monkeypatch.setattr(mc, "_pb_cusum_wild_bootstrap", few_valid)
    r = mc.passing_bablok(ref, cand, _ids(40), cusum_method="bootstrap", n_cusum_bootstrap=50, seed=0)
    assert r.meta.configuration["n_cusum_bootstrap_valid"] == 30
    assert any("20 of 50 CUSUM bootstrap replicates" in w for w in r.meta.warnings)
    assert np.isfinite(r.cusum_p_value)
    r = mc.passing_bablok(ref, cand, _ids(40), cusum_method="bootstrap", n_cusum_bootstrap=100, seed=0)
    assert np.isnan(r.cusum_p_value)
    assert not r.cusum_reject_linearity
    assert any("only 30 of 100" in w for w in r.meta.warnings)


@pytest.mark.parametrize("max_pairs", [1 << 18, 0])  # triu-gather and row-by-row paths
def test_passing_bablok_workspace_fit_matches_sorted_estimate(monkeypatch, max_pairs):
    monkeypatch.setattr(mc, "_PB_TRIU_MAX_PAIRS", max_pairs)
    rng = np.random.default_rng(25)
    for trial in range(200):
        n = int(rng.integers(3, 30))
        ref = np.round(rng.uniform(0, 10, n), int(rng.integers(0, 2)))  # ties -> +/-inf slopes
        slope = rng.choice([1.0, 0.5, 2.0, -1.0, -0.3])
        cand = np.round(slope * ref + rng.normal(size=n), int(rng.integers(0, 2)))
        if trial % 10 == 0:
            cand = ref.copy()
        ws = mc._PBWorkspace(n)
        for estimator in ("passing_bablok_1983", "manuscript_literal"):
            expected = mc._pb_estimate(ref, cand, estimator)
            got = ws.fit(ref, cand, estimator)
            np.testing.assert_array_equal(got, (expected["slope"], expected["intercept"]))


# ---------------------------------------------------------------------------
# Deming
# ---------------------------------------------------------------------------


def test_deming_noiseless_recovers_exact_slope_regardless_of_lambda():
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    test_vals = 2.0 * ref + 1.0
    ids = _ids(5)
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


def test_passing_bablok_bootstrap_variance_ratio_sets_flip_direction():
    ref, cand = _noisy_linear(26)
    fit = mc.passing_bablok(ref, cand, _ids(40), cusum_method="passing_bablok_1983")
    b0, b1, lam = fit.intercept, fit.slope, 0.25
    h_star = mc._pb_cusum_wild_bootstrap(ref, cand, b0, b1, "passing_bablok_1983", 1, _AllMinusOne(), lam)
    # Oblique reflection along d = (-lam * b1, 1): the residual flips sign and
    # the moved point stays on the line through the original point along d.
    c = (cand - b0 - b1 * ref) / (1 + lam * b1**2)
    ref_m, cand_m = ref + 2 * c * lam * b1, cand - 2 * c
    np.testing.assert_allclose(cand_m - b0 - b1 * ref_m, -(cand - b0 - b1 * ref), atol=1e-12)
    refit = mc.passing_bablok(ref_m, cand_m, _ids(40), cusum_method="passing_bablok_1983")
    assert h_star[0] == pytest.approx(refit.cusum_statistic)
    # lambda = 1 is the perpendicular bootstrap.
    rng_a, rng_b = np.random.default_rng(1), np.random.default_rng(1)
    a = mc._pb_cusum_wild_bootstrap(ref, cand, b0, b1, "passing_bablok_1983", 20, rng_a)
    b = mc._pb_cusum_wild_bootstrap(ref, cand, b0, b1, "passing_bablok_1983", 20, rng_b, 1.0)
    np.testing.assert_allclose(a, b)
    r = mc.passing_bablok(
        ref, cand, _ids(40), cusum_method="bootstrap", n_cusum_bootstrap=49,
        cusum_bootstrap_variance_ratio=lam, seed=0,
    )
    assert r.meta.configuration["cusum_bootstrap_variance_ratio"] == lam
    assert r.meta.configuration["cusum_bootstrap_residual_direction"].startswith("error_covariance_conjugate")


def test_bootstrap_cusum_warns_for_large_n_without_variance_ratio():
    rng = np.random.default_rng(31)
    n = 500
    truth = rng.uniform(1.0, 10.0, n)
    ref = truth + rng.normal(0.0, 0.3, n)
    cand = 0.5 + 1.1 * truth + rng.normal(0.0, 0.33, n)
    ids = _ids(n)
    r = mc.passing_bablok(ref, cand, ids, n_cusum_bootstrap=20, seed=0)
    assert any("cusum_bootstrap_variance_ratio" in w for w in r.meta.warnings)
    r_lambda = mc.passing_bablok(
        ref, cand, ids, n_cusum_bootstrap=20, cusum_bootstrap_variance_ratio=0.83, seed=0
    )
    assert not any("cusum_bootstrap_variance_ratio" in w for w in r_lambda.meta.warnings)
    small = mc.passing_bablok(ref[:100], cand[:100], ids[:100], n_cusum_bootstrap=20, seed=0)
    assert not any("cusum_bootstrap_variance_ratio" in w for w in small.meta.warnings)
