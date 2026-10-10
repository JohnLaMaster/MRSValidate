import numpy as np
import pytest
from scipy import stats as scipy_stats

from mrsvalidate.analysis import resampling as rs
from mrsvalidate.results import ResultStatus


def _mean_diff(a, b):
    return float(np.mean(a) - np.mean(b))


def test_bootstrap_ci_paired_reproducible_with_seed():
    rng = np.random.default_rng(0)
    a = rng.normal(size=40)
    b = a + rng.normal(scale=0.1, size=40)
    ids = np.array([f"s{i}" for i in range(40)])

    r1 = rs.bootstrap_ci(_mean_diff, a, b, sample_ids=ids, design="paired", seed=123, n_resamples=200)
    r2 = rs.bootstrap_ci(_mean_diff, a, b, sample_ids=ids, design="paired", seed=123, n_resamples=200)
    assert r1.ci_low == pytest.approx(r2.ci_low)
    assert r1.ci_high == pytest.approx(r2.ci_high)
    assert r1.ci_low <= r1.point_estimate <= r1.ci_high


def test_bootstrap_ci_defaults_to_manuscript_percentile_interval():
    rng = np.random.default_rng(4)
    a = rng.normal(size=40)
    ids = np.array([f"s{i}" for i in range(40)])
    r = rs.bootstrap_ci(np.mean, a, sample_ids=ids, n_resamples=500, seed=1)
    assert r.method == "percentile"
    expected = scipy_stats.bootstrap(
        (a,), np.mean, n_resamples=500, method="percentile", random_state=1, vectorized=False
    ).confidence_interval
    assert r.ci_low == pytest.approx(expected.low)
    assert r.ci_high == pytest.approx(expected.high)


def test_bootstrap_ci_insufficient_data():
    a = np.array([1.0])
    ids = np.array(["a"])
    r = rs.bootstrap_ci(np.mean, a, sample_ids=ids, n_resamples=100)
    assert r.meta.status is ResultStatus.INSUFFICIENT_DATA


def test_cluster_bootstrap_carries_whole_clusters():
    # Two clusters, each perfectly homogeneous internally: cluster bootstrap
    # replicate means must always equal one of the two cluster means (block
    # resampling), which independent-row resampling would not guarantee.
    values = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0])
    clusters = np.array(["c1", "c1", "c1", "c2", "c2", "c2"])
    ids = np.array([f"s{i}" for i in range(6)])

    r = rs.cluster_bootstrap_ci(
        np.mean, values, cluster_ids=clusters, sample_ids=ids, n_resamples=50, seed=1
    )
    assert r.meta.status is ResultStatus.VALID
    assert r.ci_low in (1.0, 5.0) or 1.0 <= r.ci_low <= 5.0
    assert r.point_estimate == pytest.approx(3.0)


def test_cluster_bootstrap_insufficient_with_one_cluster():
    values = np.array([1.0, 2.0, 3.0])
    clusters = np.array(["c1", "c1", "c1"])
    ids = np.array(["a", "b", "c"])
    r = rs.cluster_bootstrap_ci(np.mean, values, cluster_ids=clusters, sample_ids=ids, n_resamples=10)
    assert r.meta.status is ResultStatus.INSUFFICIENT_DATA


def test_permutation_test_paired_samples_uses_scipy_samples_type():
    rng = np.random.default_rng(5)
    a = rng.normal(size=20)
    b = a + rng.normal(scale=2.0, size=20)
    ids = np.array([f"s{i}" for i in range(20)])

    def stat(x, y):
        return np.mean(x - y)

    result = rs.permutation_test(
        stat, a, b, sample_ids=ids, design="paired_samples", n_resamples=200, seed=1
    )
    assert result.meta.configuration["scipy_permutation_type"] == "samples"
    assert 0.0 <= result.p_value <= 1.0
