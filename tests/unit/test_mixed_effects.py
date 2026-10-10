import numpy as np
import pytest

from mrsvalidate.analysis import mixed_effects as me
from mrsvalidate.analysis.method_comparison import deming
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


# ---------------------------------------------------------------------------
# Tests 1-2
# ---------------------------------------------------------------------------


def test_mixed_test1_near_zero_alpha_for_unbiased_data():
    rng = np.random.default_rng(0)
    y, _x, d, metab, cluster, ids = _make_dataset(rng, bias=0.0)
    r = me.mixed_equivalence_test_1(d, y, metab, cluster, ids)
    assert r.meta.status is ResultStatus.VALID
    assert abs(r.alpha) < 0.3


def test_mixed_test1_detects_model_level_bias():
    rng = np.random.default_rng(1)
    y, _x, d, metab, cluster, ids = _make_dataset(rng, bias=3.0)
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


def test_mixed_test1_insufficient_data_single_cluster():
    rng = np.random.default_rng(5)
    y, _x, d, metab, _cluster, ids = _make_dataset(rng, n_clusters=3)
    r = me.mixed_equivalence_test_1(d, y, metab, np.full(len(y), "c0"), ids)
    assert r.meta.status is ResultStatus.INSUFFICIENT_DATA


def test_mixed_test1_small_sample_resolution_and_records():
    rng = np.random.default_rng(6)
    y, _x, d, metab, cluster, ids = _make_dataset(rng, n_clusters=20)
    auto = me.mixed_equivalence_test_1(d, y, metab, cluster, ids)
    assert auto.se_method == "CR2"
    assert auto.meta.configuration["small_sample"] == "auto"
    assert auto.meta.configuration["se_method"] == "CR2"
    assert np.isfinite(auto.df) and 1.0 < auto.df < 20.0

    cr0 = me.mixed_equivalence_test_1(d, y, metab, cluster, ids, small_sample="CR0")
    cr1 = me.mixed_equivalence_test_1(d, y, metab, cluster, ids, small_sample="CR1")
    assert cr0.df == np.inf
    assert cr1.df == 19.0
    n, p, g = len(y), 2, 20
    factor = g / (g - 1) * (n - 1) / (n - p)
    assert cr1.alpha_se_cluster_robust == pytest.approx(cr0.alpha_se_cluster_robust * np.sqrt(factor), rel=1e-12)

    y50, _x, d50, metab50, cluster50, ids50 = _make_dataset(rng, n_metabolites=3, n_clusters=50)
    assert me.mixed_equivalence_test_1(d50, y50, metab50, cluster50, ids50).se_method == "CR1"

    with pytest.raises(ValueError):
        me.mixed_equivalence_test_1(d, y, metab, cluster, ids, small_sample="HC3")


def test_mixed_test2_beta_near_zero_for_equal_variance_no_slope_dev():
    rng = np.random.default_rng(2)
    y, x, d, metab, cluster, ids = _make_dataset(rng, bias=0.0, slope_dev=0.0)
    r = me.mixed_equivalence_test_2(d, y, x, metab, cluster, ids)
    assert r.meta.status is ResultStatus.VALID
    assert abs(r.beta) < 0.1
    assert np.isfinite(r.tau_alpha_sq) and r.tau_alpha_sq >= 0
    assert np.isfinite(r.tau_beta_sq) and r.tau_beta_sq >= 0
    assert r.se_method == "CR2"
    assert np.isfinite(r.beta_t) and np.isfinite(r.df)


def _cr2_brute_force(x, e, cluster_codes):
    """Hand computation of CR2 + Bell-McCaffrey df with the full N x N matrices."""
    n, p = x.shape
    bread = np.linalg.inv(x.T @ x)
    i_minus_h = np.eye(n) - x @ bread @ x.T
    groups = [np.flatnonzero(cluster_codes == g) for g in np.unique(cluster_codes)]
    a_mats = []
    meat = np.zeros((p, p))
    for idx in groups:
        evals, evecs = np.linalg.eigh(i_minus_h[np.ix_(idx, idx)])
        a_g = evecs @ np.diag(evals**-0.5) @ evecs.T
        a_mats.append(a_g)
        score = x[idx].T @ a_g @ e[idx]
        meat += np.outer(score, score)
    se = np.sqrt(np.diag(bread @ meat @ bread))
    dfs = []
    for j in range(p):
        c = np.eye(p)[j]
        g_mat = np.column_stack([i_minus_h[:, idx] @ a_g @ x[idx] @ bread @ c for idx, a_g in zip(groups, a_mats)])
        m = g_mat.T @ g_mat
        dfs.append(np.trace(m) ** 2 / np.trace(m @ m))
    return se, np.array(dfs)


def test_cr2_matches_hand_computation_balanced():
    rng = np.random.default_rng(10)
    n_clusters, n_per = 6, 4
    codes = np.repeat(np.arange(n_clusters), n_per)
    x = np.column_stack([np.ones(n_clusters * n_per), rng.normal(size=n_clusters * n_per)])
    e = rng.normal(size=n_clusters * n_per)
    se, df, warns = me._cluster_robust_inference(x, e, codes.astype(str), "CR2")
    se_ref, df_ref = _cr2_brute_force(x, e, codes)
    np.testing.assert_allclose(se, se_ref, rtol=1e-10)
    np.testing.assert_allclose(df, df_ref, rtol=1e-10)
    assert warns == ()
    # Intercept-only, balanced: CR2 df is exactly G - 1.
    _se1, df1, _ = me._cluster_robust_inference(x[:, :1], e, codes.astype(str), "CR2")
    assert df1[0] == pytest.approx(n_clusters - 1, rel=1e-10)


def test_cr2_matches_hand_computation_unbalanced():
    rng = np.random.default_rng(11)
    codes = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4])
    n = codes.shape[0]
    x = np.column_stack([np.ones(n), rng.normal(size=n), rng.uniform(size=n)])
    e = rng.normal(size=n)
    se, df, _ = me._cluster_robust_inference(x, e, codes.astype(str), "CR2")
    se_ref, df_ref = _cr2_brute_force(x, e, codes)
    np.testing.assert_allclose(se, se_ref, rtol=1e-10)
    np.testing.assert_allclose(df, df_ref, rtol=1e-10)


def test_cr2_warns_on_unit_leverage_cluster():
    # A cluster-specific dummy gives that cluster leverage 1 (I - H_gg singular).
    codes = np.repeat(np.arange(5), 3)
    rng = np.random.default_rng(12)
    x = np.column_stack([np.ones(15), (codes == 0).astype(float), rng.normal(size=15)])
    e = rng.normal(size=15)
    se, _df, warns = me._cluster_robust_inference(x, e, codes.astype(str), "CR2")
    assert any("pseudo-inverse" in w for w in warns)
    assert np.all(np.isfinite(se))


def test_cr2_bell_mccaffrey_size_closer_to_nominal_than_cr0_with_10_clusters():
    """G = 10 clusters under H0: CR2 + BM t-test is near 5 %, CR0 + z over-rejects.

    Reference run (R = 2000, seed 123): CR0+z 0.110 (intercept) / 0.175
    (slope), CR1+t(G-1) 0.061 / 0.120, CR2+BM 0.044 / 0.051.
    """
    rng = np.random.default_rng(123)
    n_clusters, n_per, n_rep = 10, 6, 1000
    codes = np.repeat(np.arange(n_clusters), n_per)
    labels = codes.astype(str)
    rejections = {"CR0": np.zeros(2), "CR2": np.zeros(2)}
    for _ in range(n_rep):
        cov = rng.normal(size=n_clusters)[codes] + 0.5 * rng.normal(size=codes.shape[0])
        x = np.column_stack([np.ones(codes.shape[0]), cov])
        y = rng.normal(size=n_clusters)[codes] + rng.normal(size=codes.shape[0])  # true coefficients 0
        b = np.linalg.lstsq(x, y, rcond=None)[0]
        resid = y - x @ b
        for method, counts in rejections.items():
            se, df, _ = me._cluster_robust_inference(x, resid, labels, method)
            for j in range(2):
                _stat, p = me._t_test(b[j], se[j], df[j])
                counts[j] += p < 0.05
    rate_cr0 = rejections["CR0"] / n_rep
    rate_cr2 = rejections["CR2"] / n_rep
    assert np.all(np.abs(rate_cr2 - 0.05) < 0.025)
    assert np.all(rate_cr0 > 0.08)
    assert np.all(np.abs(rate_cr2 - 0.05) < np.abs(rate_cr0 - 0.05) - 0.03)


# ---------------------------------------------------------------------------
# York regression (Test 3, stage 1)
# ---------------------------------------------------------------------------


def _york_single(x, y, vx, vy):
    return me._york_grouped(
        np.asarray(x, float), np.asarray(y, float), np.broadcast_to(vx, np.shape(x)).astype(float),
        np.broadcast_to(vy, np.shape(x)).astype(float), np.array([0]), np.ones((1, len(x))),
    )


@pytest.mark.parametrize(("ref_var", "test_var"), [(1.0, 1.0), (0.25, 0.64), (2.0, 0.1), (0.01, 3.0)])
def test_york_constant_variances_reproduces_deming(ref_var, test_var):
    rng = np.random.default_rng(20)
    eta = rng.uniform(1, 10, 40)
    ref = eta + rng.normal(0, np.sqrt(ref_var), 40)
    cand = 0.4 + 1.15 * eta + rng.normal(0, np.sqrt(test_var), 40)
    fit = _york_single(ref, cand, ref_var, test_var)
    dem = deming(ref, cand, np.arange(40).astype(str), reference_error_variance=ref_var, test_error_variance=test_var)
    assert fit.converged[0, 0]
    assert fit.slope[0, 0] == pytest.approx(dem.slope, abs=1e-10)
    assert fit.intercept[0, 0] == pytest.approx(dem.intercept, abs=1e-10)


def test_york_pearson_data_matches_york_2004():
    # Pearson's data with York's weights (York et al. 2004, Am. J. Phys. 72:367).
    x = np.array([0.0, 0.9, 1.8, 2.6, 3.3, 4.4, 5.2, 6.1, 6.5, 7.4])
    wx = np.array([1000, 1000, 500, 800, 200, 80, 60, 20, 1.8, 1.0])
    y = np.array([5.9, 5.4, 4.4, 4.6, 3.5, 3.7, 2.8, 2.8, 2.4, 1.5])
    wy = np.array([1, 1.8, 4, 8, 20, 20, 70, 70, 100, 500.0])
    fit = _york_single(x, y, 1 / wx, 1 / wy)
    assert fit.slope[0, 0] == pytest.approx(-0.4805, abs=5e-5)
    assert fit.intercept[0, 0] == pytest.approx(5.4799, abs=5e-5)
    assert np.sqrt(fit.var_slope[0, 0]) == pytest.approx(0.0580, abs=5e-5)
    assert np.sqrt(fit.var_intercept[0, 0]) == pytest.approx(0.2950, abs=5e-5)


def test_york_frequency_weights_equal_duplication_and_grouping():
    rng = np.random.default_rng(21)
    x = np.sort(rng.uniform(0, 5, 12))
    y = 1.0 + 0.8 * x + rng.normal(0, 0.2, 12)
    vx = rng.uniform(0.01, 0.05, 12)
    vy = rng.uniform(0.02, 0.08, 12)
    freq = rng.integers(0, 3, 12).astype(float)
    freq[:3] = 1.0
    weighted = me._york_grouped(x, y, vx, vy, np.array([0]), freq[None, :])
    rep = np.repeat(np.arange(12), freq.astype(int))
    dup = me._york_grouped(x[rep], y[rep], vx[rep], vy[rep], np.array([0]), np.ones((1, rep.shape[0])))
    for field in ("slope", "intercept", "var_slope", "var_intercept", "cov_intercept_slope"):
        assert getattr(weighted, field)[0, 0] == pytest.approx(getattr(dup, field)[0, 0], rel=1e-9)
    # Two groups fit jointly equal the two separate fits.
    joint = me._york_grouped(np.r_[x, x], np.r_[y, 2 * y], np.r_[vx, vx], np.r_[vy, vy], np.array([0, 12]),
                             np.ones((1, 24)))
    second = _york_single(x, 2 * y, vx, vy)
    assert joint.slope[0, 1] == pytest.approx(second.slope[0, 0], rel=1e-10)


# ---------------------------------------------------------------------------
# Test 3 (two-stage York + REML synthesis + cluster bootstrap)
# ---------------------------------------------------------------------------


def _simulate_me_test3(rng, *, n_metabolites=12, n_spectra=60, alpha=0.0, beta=1.0, tau_a_sq=0.0, tau_b_sq=0.0,
                       panel=None):
    """Data from the me_test3 model with heteroscedastic per-observation variances.

    ``panel = (u, v, centers)`` fixes the metabolite effects (panel-conditional
    simulation); otherwise they are drawn fresh (population simulation).
    """
    if panel is None:
        u = rng.normal(0.0, np.sqrt(tau_a_sq), n_metabolites)
        v = rng.normal(0.0, np.sqrt(tau_b_sq), n_metabolites)
        centers = rng.uniform(2.0, 10.0, n_metabolites)
    else:
        u, v, centers = panel
        n_metabolites = u.shape[0]
    k = np.repeat(np.arange(n_metabolites), n_spectra)
    c = np.tile(np.arange(n_spectra), n_metabolites)
    eta = centers[k] * (1.0 + rng.uniform(-0.4, 0.4, k.shape[0]))
    var_ref = (0.03 * eta) ** 2 * rng.uniform(0.5, 2.0, k.shape[0])
    var_test = (0.05 * eta) ** 2 * rng.uniform(0.5, 2.0, k.shape[0])
    ref = eta + rng.normal(0.0, np.sqrt(var_ref))
    cand = alpha + u[k] + (beta + v[k]) * eta + rng.normal(0.0, np.sqrt(var_test))
    metab = np.array([f"m{i:02d}" for i in k])
    clusters = np.array([f"s{i}" for i in c])
    ids = np.array([f"s{ci}_m{ki}" for ci, ki in zip(c, k)])
    return ref, cand, metab, clusters, ids, var_ref, var_test


def _run_test3(data, **kwargs):
    ref, cand, metab, clusters, ids, var_ref, var_test = data
    return me.mixed_equivalence_test_3(
        ref, cand, metab, clusters, ids, reference_error_variance=var_ref, test_error_variance=var_test, **kwargs
    )


def test_mixed_test3_does_not_reject_for_near_identity():
    rng = np.random.default_rng(3)
    y, x, _d, metab, cluster, ids = _make_dataset(rng, bias=0.0, slope_dev=0.0)
    r = me.mixed_equivalence_test_3(
        y, x, metab, cluster, ids,
        reference_error_variance=0.02, test_error_variance=0.02,
        n_bootstrap=200, seed=0,
    )
    assert r.meta.status is ResultStatus.VALID
    assert not r.reject_joint_null
    assert r.n_metabolites == 6 and r.n_clusters == 40
    assert r.metabolite_labels == tuple(f"m{m}" for m in range(6))
    assert r.metabolite_slopes.shape == (6,)
    assert r.tau_alpha_sq_ci[0] <= r.tau_alpha_sq_ci[1]


def test_mixed_test3_rejects_for_clear_disagreement():
    rng = np.random.default_rng(4)
    y, x, _d, metab, cluster, ids = _make_dataset(rng, bias=3.0, slope_dev=0.5)
    r = me.mixed_equivalence_test_3(
        y, x, metab, cluster, ids,
        reference_error_variance=1.0, test_error_variance=1.0,
        n_bootstrap=200, seed=0,
    )
    assert r.reject_joint_null


def test_mixed_test3_recovers_simulated_parameters():
    """Averaged over 25 simulated panels (K = 12, 60 spectra): estimates near truth.

    Reference run (100 panels): mean alpha 0.194 (truth 0.2), beta 1.0500
    (1.05), tau_alpha_sq 0.0399 (0.04), tau_beta_sq 0.00220 (0.0025).
    """
    rng = np.random.default_rng(30)
    est = []
    for rep in range(25):
        data = _simulate_me_test3(rng, alpha=0.2, beta=1.05, tau_a_sq=0.04, tau_b_sq=0.0025)
        r = _run_test3(data, n_bootstrap=1, seed=rep)
        assert r.meta.status is ResultStatus.VALID
        est.append([r.alpha, r.beta, r.tau_alpha_sq, r.tau_beta_sq])
    mean = np.mean(est, axis=0)
    assert mean[0] == pytest.approx(0.2, abs=0.05)
    assert mean[1] == pytest.approx(1.05, abs=0.01)
    assert mean[2] == pytest.approx(0.04, abs=0.015)
    assert mean[3] == pytest.approx(0.0025, abs=0.001)


def test_mixed_test3_per_metabolite_matches_deming_with_constant_variances():
    rng = np.random.default_rng(31)
    y, x, _d, metab, cluster, ids = _make_dataset(rng, n_metabolites=3, n_clusters=15)
    r = me.mixed_equivalence_test_3(
        y, x, metab, cluster, ids, reference_error_variance=0.05, test_error_variance=0.04, n_bootstrap=10, seed=0,
    )
    for label, slope, intercept in zip(r.metabolite_labels, r.metabolite_slopes, r.metabolite_intercepts):
        m = metab == label
        dem = deming(y[m], x[m], ids[m], reference_error_variance=0.05, test_error_variance=0.04)
        assert slope == pytest.approx(dem.slope, abs=1e-10)
        assert intercept == pytest.approx(dem.intercept, abs=1e-10)


def test_mixed_test3_size_and_power():
    """H0 (alpha=0, beta=1, small tau^2) rarely rejected; beta = 1.1 always rejected.

    Reference runs (B = 400): tau^2 = 0 -> 4/100 rejections; small tau^2
    (1e-4, 1e-5) -> 3/100; beta = 1.1 -> 30/30.
    """
    rng = np.random.default_rng(32)
    n_null = 30
    rejections = 0
    for rep in range(n_null):
        data = _simulate_me_test3(rng, tau_a_sq=1e-4, tau_b_sq=1e-5)
        rejections += _run_test3(data, n_bootstrap=200, seed=rep).reject_joint_null
    assert rejections / n_null <= 0.15
    for rep in range(8):
        data = _simulate_me_test3(rng, beta=1.1, tau_a_sq=1e-4, tau_b_sq=1e-5)
        assert _run_test3(data, n_bootstrap=200, seed=rep).reject_joint_null


def test_mixed_test3_seed_reproducible_and_config_recorded():
    rng = np.random.default_rng(33)
    data = _simulate_me_test3(rng, n_metabolites=4, n_spectra=20)
    a = _run_test3(data, n_bootstrap=100, seed=7)
    b = _run_test3(data, n_bootstrap=100, seed=7)
    assert a.alpha_ci == b.alpha_ci and a.beta_ci == b.beta_ci
    cfg = a.meta.configuration
    assert cfg["variance_input"] == {"reference_error_variance": "per_observation",
                                     "test_error_variance": "per_observation"}
    assert cfg["n_bootstrap_valid"] == a.n_bootstrap_valid == 100
    assert cfg["confidence_level_per_parameter"] == pytest.approx(0.975)
    assert a.meta.random_seed == 7


def test_mixed_test3_skips_sparse_metabolites_in_bootstrap():
    rng = np.random.default_rng(34)
    ref, cand, metab, clusters, ids, vr, vt = _simulate_me_test3(rng, n_metabolites=3, n_spectra=30)
    # Metabolite m02 observed in only 3 spectra: bootstrap replicates often have < 3 distinct points.
    keep = (metab != "m02") | np.isin(clusters, ["s0", "s1", "s2"])
    r = me.mixed_equivalence_test_3(
        ref[keep], cand[keep], metab[keep], clusters[keep], ids[keep],
        reference_error_variance=vr[keep], test_error_variance=vt[keep], n_bootstrap=200, seed=0,
    )
    assert r.meta.status is ResultStatus.VALID
    assert r.meta.configuration["n_bootstrap_metabolite_skips_too_few_observations"] > 0
    assert any("skipped" in w for w in r.meta.warnings)


def test_mixed_test3_excludes_metabolite_with_too_few_observations():
    rng = np.random.default_rng(35)
    ref, cand, metab, clusters, ids, vr, vt = _simulate_me_test3(rng, n_metabolites=3, n_spectra=10)
    keep = (metab != "m02") | np.isin(clusters, ["s0", "s1"])
    r = me.mixed_equivalence_test_3(
        ref[keep], cand[keep], metab[keep], clusters[keep], ids[keep],
        reference_error_variance=vr[keep], test_error_variance=vt[keep], n_bootstrap=20, seed=0,
    )
    assert r.meta.status is ResultStatus.VALID
    assert r.metabolite_labels == ("m00", "m01")
    assert r.meta.exclusion_reasons["metabolite_with_fewer_than_3_observations"] == 2


def test_mixed_test3_insufficient_data():
    rng = np.random.default_rng(36)
    ref, cand, metab, clusters, ids, vr, vt = _simulate_me_test3(rng, n_metabolites=2, n_spectra=10)
    one = metab == "m00"
    r = me.mixed_equivalence_test_3(
        ref[one], cand[one], metab[one], clusters[one], ids[one],
        reference_error_variance=vr[one], test_error_variance=vt[one], n_bootstrap=10, seed=0,
    )
    assert r.meta.status is ResultStatus.INSUFFICIENT_DATA
    r = me.mixed_equivalence_test_3(
        ref, cand, metab, np.full(ref.shape[0], "s0"), ids,
        reference_error_variance=vr, test_error_variance=vt, n_bootstrap=10, seed=0,
    )
    assert r.meta.status is ResultStatus.INSUFFICIENT_DATA


def test_mixed_test3_invalid_variances():
    rng = np.random.default_rng(37)
    ref, cand, metab, clusters, ids, vr, _vt = _simulate_me_test3(rng, n_metabolites=2, n_spectra=10)
    r = me.mixed_equivalence_test_3(
        ref, cand, metab, clusters, ids, reference_error_variance=vr, test_error_variance=0.0, n_bootstrap=10,
    )
    assert r.meta.status is ResultStatus.INVALID_INPUT
    with pytest.raises(ValueError):
        me.mixed_equivalence_test_3(
            ref, cand, metab, clusters, ids, reference_error_variance=vr[:3], test_error_variance=1.0,
        )


def test_reml_random_effects_known_cases():
    # Homogeneous estimates -> tau^2 = 0 and the common value.
    theta = np.array([[1.0, 1.0, 1.0, 1.0]])
    s2 = np.array([[0.1, 0.2, 0.3, 0.4]])
    mu, tau2, conv = me._reml_random_effects(theta, s2, np.ones_like(theta, dtype=bool))
    assert tau2[0] == 0.0 and mu[0] == pytest.approx(1.0) and conv[0]
    # Equal s^2: REML tau^2 = sample variance - s^2 (closed form), mean = simple mean.
    theta = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]])
    s2 = np.full((1, 5), 0.5)
    mu, tau2, conv = me._reml_random_effects(theta, s2, np.ones_like(theta, dtype=bool))
    assert tau2[0] == pytest.approx(np.var(theta, ddof=1) - 0.5, rel=1e-8)
    assert mu[0] == pytest.approx(2.0)
    # Masked entries are ignored.
    theta_m = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 100.0]])
    s2_m = np.full((1, 6), 0.5)
    mask = np.array([[True] * 5 + [False]])
    mu_m, tau2_m, _ = me._reml_random_effects(theta_m, s2_m, mask)
    assert mu_m[0] == pytest.approx(mu[0]) and tau2_m[0] == pytest.approx(tau2[0])


def test_reml_random_effects_pooled_estimate_is_metabolite_balanced():
    # Very unequal stage-1 precisions must not change the pooled estimate:
    # every metabolite carries equal weight (metabolite-balanced aggregation).
    theta = np.array([[0.0, 1.0, 2.0, 10.0]])
    s2 = np.array([[1e-4, 1e-4, 1e-4, 10.0]])
    mu, tau2, _ = me._reml_random_effects(theta, s2, np.ones_like(theta, dtype=bool))
    assert mu[0] == pytest.approx(np.mean(theta))
    assert tau2[0] > 0.0


# ---------------------------------------------------------------------------
# Test 3 inference scope: panel-conditional (default) vs population (pigeonhole)
# ---------------------------------------------------------------------------


def _fixed_panel(rng, n_metabolites, tau_a_sq, tau_b_sq):
    """Fixed metabolite effects whose panel average is exactly alpha = 0, beta = 1."""
    centers = rng.uniform(2.0, 10.0, n_metabolites)
    u = rng.normal(0.0, np.sqrt(tau_a_sq), n_metabolites)
    v = rng.normal(0.0, np.sqrt(tau_b_sq), n_metabolites)
    return u - u.mean(), v - v.mean(), centers


def test_mixed_test3_inference_option_recorded_and_validated():
    rng = np.random.default_rng(40)
    data = _simulate_me_test3(rng, n_metabolites=5, n_spectra=20, tau_a_sq=0.04, tau_b_sq=0.0025)
    default = _run_test3(data, n_bootstrap=80, seed=3)
    panel = _run_test3(data, n_bootstrap=80, seed=3, inference="panel")
    pop = _run_test3(data, n_bootstrap=80, seed=3, inference="population")
    # Default is panel mode, and panel mode is untouched by the new option.
    assert default.inference == panel.inference == "panel"
    assert default.alpha_ci == panel.alpha_ci and default.beta_ci == panel.beta_ci
    assert default.tau_beta_sq_ci == panel.tau_beta_sq_ci
    assert panel.meta.configuration["inference"] == "panel"
    assert panel.meta.configuration["ci_method"] == "cluster_bootstrap_percentile_two_stage"
    assert panel.meta.configuration["bootstrap_resampling_units"] == "clusters"
    # Population mode: same point estimates, different intervals, recorded choice.
    assert pop.inference == "population"
    assert pop.meta.configuration["inference"] == "population"
    assert pop.meta.configuration["ci_method"] == "two_way_pigeonhole_bootstrap_percentile_two_stage"
    assert pop.meta.configuration["bootstrap_resampling_units"] == "metabolites x clusters"
    assert (pop.alpha, pop.beta, pop.tau_alpha_sq) == (panel.alpha, panel.beta, panel.tau_alpha_sq)
    assert pop.alpha_ci != panel.alpha_ci
    again = _run_test3(data, n_bootstrap=80, seed=3, inference="population")
    assert again.alpha_ci == pop.alpha_ci and again.beta_ci == pop.beta_ci
    with pytest.raises(ValueError, match="inference"):
        _run_test3(data, n_bootstrap=10, seed=0, inference="metabolites")
    # Early-exit results also carry the choice.
    bad = me.mixed_equivalence_test_3(*data[:5], reference_error_variance=data[5], test_error_variance=0.0,
                                      n_bootstrap=10, inference="population")
    assert bad.meta.status is ResultStatus.INVALID_INPUT and bad.inference == "population"


def test_mixed_test3_population_resamples_metabolites_with_replacement():
    """With 2 metabolites, half the pigeonhole replicates draw one metabolite twice and are dropped."""
    rng = np.random.default_rng(41)
    data = _simulate_me_test3(rng, n_metabolites=2, n_spectra=25, tau_a_sq=0.04, tau_b_sq=0.0025)
    panel = _run_test3(data, n_bootstrap=400, seed=0)
    pop = _run_test3(data, n_bootstrap=400, seed=0, inference="population")
    assert panel.n_bootstrap_valid == 400
    assert 150 <= pop.n_bootstrap_valid <= 250  # Binomial(400, 1/2)
    assert pop.meta.configuration["n_bootstrap_dropped"] == 400 - pop.n_bootstrap_valid
    assert any("distinct" in w for w in pop.meta.warnings)


def test_mixed_test3_population_intervals_wider_with_heterogeneity():
    """Population intervals include between-metabolite variation; panel intervals do not."""
    rng = np.random.default_rng(42)
    data = _simulate_me_test3(rng, n_metabolites=12, n_spectra=30, tau_a_sq=0.64, tau_b_sq=0.04)
    panel = _run_test3(data, n_bootstrap=300, seed=1)
    pop = _run_test3(data, n_bootstrap=300, seed=1, inference="population")
    width = lambda ci: ci[1] - ci[0]
    assert width(pop.beta_ci) > 1.8 * width(panel.beta_ci)
    assert width(pop.alpha_ci) > 2.5 * width(panel.alpha_ci)
    # tau_beta ~ 0.2 across 12 metabolites: the population SE of beta is ~ 0.2 / sqrt(12) ~ 0.06.
    assert 0.1 < width(pop.beta_ci) < 0.6


def test_mixed_test3_panel_mode_size_under_panel_null():
    """Panel mode, fixed panel with exact panel-average alpha = 0, beta = 1, nonzero spread.

    Reference run (1000 datasets, K = 12, 30 spectra, B = 500,
    tau^2 = (0.04, 0.0025)): joint rejection 0.053 (alpha-only 0.040,
    beta-only 0.034); with tau^2 = 0: 0.044.
    """
    rng = np.random.default_rng(43)
    panel = _fixed_panel(rng, 12, 0.04, 0.0025)
    n_rep = 40
    rejections = sum(
        _run_test3(_simulate_me_test3(rng, n_spectra=30, panel=panel), n_bootstrap=200, seed=rep).reject_joint_null
        for rep in range(n_rep)
    )
    assert rejections / n_rep <= 0.15


def test_mixed_test3_population_mode_size_under_population_null():
    """Population mode, fresh random metabolite effects per dataset (population means 0 and 1).

    Reference runs (1000 datasets each, 30 spectra, B = 500): K = 12,
    tau^2 = (0.04, 0.0025): 0.011 (conservative: stage-1 error counted
    twice by the pigeonhole bootstrap); K = 25: 0.002; K = 12,
    tau^2 = (0.16, 0.01): 0.046; K = 12, tau^2 = (0.64, 0.04): 0.076;
    K = 25, tau^2 = (0.64, 0.04): 0.062.
    """
    rng = np.random.default_rng(44)
    n_rep = 40
    rejections = sum(
        _run_test3(_simulate_me_test3(rng, n_spectra=30, tau_a_sq=0.16, tau_b_sq=0.01), n_bootstrap=200, seed=rep,
                   inference="population").reject_joint_null
        for rep in range(n_rep)
    )
    assert rejections / n_rep <= 0.15


def test_mixed_test3_panel_mode_rejects_more_under_population_null():
    """Expected, not a bug: panel mode tests the panel summary, not the population mean.

    Reference runs (1000 datasets each, K = 12, 30 spectra, B = 500):
    tau^2 = (0.04, 0.0025): panel 0.238 vs population 0.011;
    tau^2 = (0.64, 0.04): panel 0.788 vs population 0.076.
    """
    rng = np.random.default_rng(45)
    n_rep = 30
    panel_rej = pop_rej = 0
    for rep in range(n_rep):
        data = _simulate_me_test3(rng, n_spectra=30, tau_a_sq=0.64, tau_b_sq=0.04)
        panel_rej += _run_test3(data, n_bootstrap=150, seed=rep).reject_joint_null
        pop_rej += _run_test3(data, n_bootstrap=150, seed=rep, inference="population").reject_joint_null
    assert panel_rej / n_rep >= 0.5
    assert pop_rej / n_rep <= 0.25
