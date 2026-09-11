import numpy as np
import pytest

from mrsvalidate.analysis.operating_envelope import _clopper_pearson_lower, operating_envelope
from mrsvalidate.results import ResultStatus


def test_clopper_pearson_lower_zero_for_zero_successes():
    lower = _clopper_pearson_lower(np.array([0]), np.array([10]), 0.025)
    assert lower[0] == 0.0


def test_clopper_pearson_lower_never_exceeds_empirical_rate():
    rng = np.random.default_rng(0)
    totals = rng.integers(5, 200, size=50)
    successes = np.array([rng.integers(0, t + 1) for t in totals])
    lower = _clopper_pearson_lower(successes, totals, 0.025)
    rate = successes / totals
    valid = totals > 0
    assert np.all(lower[valid] <= rate[valid] + 1e-9)


def _make_dataset(n_per_cell=50, success_rate_by_stratum=(0.95, 0.5)):
    rng = np.random.default_rng(1)
    conc = []
    info = []
    success = []
    for stratum_idx, rate in enumerate(success_rate_by_stratum):
        for _ in range(n_per_cell):
            conc.append(rng.uniform(0.0, 1.0))  # single bin [0,1]
            info.append(stratum_idx + 0.5)  # falls in stratum `stratum_idx`
            success.append(rng.random() < rate)
    ids = np.array([f"s{i}" for i in range(len(conc))])
    return np.array(conc), np.array(info), np.array(success), ids


def test_operating_envelope_robust_bound_is_min_across_strata():
    conc, info, success, ids = _make_dataset()
    r = operating_envelope(
        conc,
        info,
        success,
        ids,
        bin_edges=np.array([0.0, 1.0]),
        stratum_edges=np.array([0.0, 1.0, 2.0]),
        target_reliability=0.8,
        alpha=0.05,
    )
    assert r.meta.status is ResultStatus.VALID
    assert r.lower_confidence_bound.shape == (1, 2)
    assert r.robust_lower_bound[0] == pytest.approx(
        min(r.lower_confidence_bound[0, 0], r.lower_confidence_bound[0, 1])
    )
    # the low-reliability stratum (rate=0.5) should pull the robust bound below target
    assert not r.operating_bin_mask[0]


def test_operating_envelope_high_reliability_both_strata_passes():
    conc, info, success, ids = _make_dataset(
        n_per_cell=300, success_rate_by_stratum=(0.99, 0.97)
    )
    r = operating_envelope(
        conc,
        info,
        success,
        ids,
        bin_edges=np.array([0.0, 1.0]),
        stratum_edges=np.array([0.0, 1.0, 2.0]),
        target_reliability=0.9,
        alpha=0.05,
    )
    assert r.operating_bin_mask[0]


def test_operating_envelope_sparse_stratum_marks_insufficient_and_not_interpolated():
    conc = np.array([0.5, 0.5, 0.5])
    info = np.array([0.5, 0.5, 1.5])  # stratum 1 has only 1 observation
    success = np.array([True, True, True])
    ids = np.array(["a", "b", "c"])

    r = operating_envelope(
        conc,
        info,
        success,
        ids,
        bin_edges=np.array([0.0, 1.0]),
        stratum_edges=np.array([0.0, 1.0, 2.0]),
        target_reliability=0.5,
        alpha=0.05,
        min_stratum_n=2,
    )
    assert not r.sufficient_data_mask[0]
    assert np.isnan(r.robust_lower_bound[0])
    assert not r.operating_bin_mask[0]  # never silently "passes" due to sparse data


def test_operating_envelope_excludes_out_of_range_observations():
    conc = np.array([0.5, 5.0])  # second is out of bin range
    info = np.array([0.5, 0.5])
    success = np.array([True, True])
    ids = np.array(["a", "b"])
    r = operating_envelope(
        conc,
        info,
        success,
        ids,
        bin_edges=np.array([0.0, 1.0]),
        stratum_edges=np.array([0.0, 1.0]),
        target_reliability=0.5,
    )
    assert r.meta.n_used == 1
    assert r.meta.excluded_ids == ("b",)
