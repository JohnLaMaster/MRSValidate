import numpy as np
import pytest

from mrsvalidate.analysis import distribution as dist
from mrsvalidate.results import ResultStatus


def test_wilcoxon_component_moments_match_textbook_formula_no_ties():
    rng = np.random.default_rng(0)
    x = rng.normal(size=3)
    y = rng.normal(size=4) + 10  # ensure no ties with x
    n1, n2 = 3, 4
    N = n1 + n2

    combined = np.concatenate([x, y])
    raw = dist._wilcoxon_raw_scores(N)
    scores = dist._block_averaged_scores(combined, raw)
    mu, var = dist._score_moments(scores, n1)

    expected_mu = n1 * (N + 1) / 2
    expected_var = n1 * n2 * (N + 1) / 12
    assert mu == pytest.approx(expected_mu)
    assert var == pytest.approx(expected_var)


def test_ansari_bradley_moments_match_textbook_formula_even_n():
    n1, n2 = 4, 4
    N = n1 + n2  # even
    raw = dist._ansari_bradley_raw_scores(N)
    # No ties: use raw scores directly as "scores" for a hypothetical
    # allocation to check the *moment formula* against the textbook one.
    mu, var = dist._score_moments(raw, n1)
    expected_mu = n1 * (N + 2) / 4
    expected_var = n1 * n2 * (N + 2) * (N - 2) / (48 * (N - 1))
    assert mu == pytest.approx(expected_mu)
    assert var == pytest.approx(expected_var)


def test_savage_scores_sum_to_zero():
    n = 11
    scores = dist._savage_raw_scores(n)
    assert scores.sum() == pytest.approx(0.0, abs=1e-9)


def test_antisavage_is_reversed_savage():
    n = 9
    savage = dist._savage_raw_scores(n)
    antisavage = dist._antisavage_raw_scores(n)
    np.testing.assert_allclose(antisavage, savage[::-1])


def test_ecwas_kossler_mukherjee_larger_for_clearly_different_distributions():
    rng = np.random.default_rng(1)
    similar_x = rng.normal(size=200)
    similar_y = rng.normal(size=200)
    different_x = rng.normal(size=200)
    different_y = rng.normal(loc=3.0, scale=3.0, size=200)

    ids = np.array([f"s{i}" for i in range(200)])
    r_similar = dist.ecwas_kossler_mukherjee(similar_x, similar_y, ids)
    r_different = dist.ecwas_kossler_mukherjee(different_x, different_y, ids)

    assert r_similar.meta.status is ResultStatus.VALID
    assert r_different.statistic > r_similar.statistic


def test_ecwas_reports_small_sample_warning_below_500():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    ids = np.array(["a", "b", "c"])
    r = dist.ecwas_kossler_mukherjee(x, y, ids)
    assert any("n>=500" in w for w in r.meta.warnings)


def test_ecwas_insufficient_data_for_singleton_groups():
    x = np.array([1.0])
    y = np.array([2.0])
    ids = np.array(["a", "b"])
    r = dist.ecwas_kossler_mukherjee(x, y, ids)
    assert r.meta.status is ResultStatus.INSUFFICIENT_DATA


def test_ecwas_manuscript_max_shape_runs_and_flags_monte_carlo():
    rng = np.random.default_rng(2)
    x = rng.normal(size=200)
    y = rng.normal(loc=1.0, size=200)
    ids = np.array([f"s{i}" for i in range(200)])
    r = dist.ecwas_manuscript_max_shape(x, y, ids, n_null_resamples=200, seed=0)
    assert r.meta.status is ResultStatus.VALID
    assert r.meta.configuration["shape_null_moments"] == "monte_carlo"
    assert r.shape_component == "max_antisavage_savage"
    assert np.isfinite(r.statistic)


def test_ecwas_variants_are_distinctly_named_and_not_conflated():
    rng = np.random.default_rng(3)
    x = rng.normal(size=200)
    y = rng.normal(loc=0.5, size=200)
    ids = np.array([f"s{i}" for i in range(200)])
    km = dist.ecwas_kossler_mukherjee(x, y, ids)
    fu = dist.ecwas_manuscript_max_shape(x, y, ids, n_null_resamples=200, seed=0)
    assert km.meta.method != fu.meta.method
    assert km.shape_component != fu.shape_component
