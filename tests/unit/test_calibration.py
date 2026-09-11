import numpy as np
import pytest

from mrsvalidate.analysis import calibration as cal
from mrsvalidate.results import ResultStatus


def test_perfectly_calibrated_pit_gives_zero_ece():
    rng = np.random.default_rng(0)
    pit = rng.uniform(0.0, 1.0, size=5000)  # uniform PIT == perfect calibration
    levels = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    ids = np.array([f"s{i}" for i in range(5000)])
    r = cal.expected_calibration_error(pit, levels, ids)
    assert r.meta.status is ResultStatus.VALID
    np.testing.assert_allclose(r.empirical_coverage, levels, atol=0.03)
    assert r.ece < 0.03


def test_overconfident_pit_gives_positive_ece():
    # All PIT values clustered at 0.5 -> massively overconfident/underdispersed
    pit = np.full(100, 0.5)
    levels = np.array([0.1, 0.5, 0.9])
    ids = np.array([f"s{i}" for i in range(100)])
    r = cal.expected_calibration_error(pit, levels, ids)
    assert r.ece > 0.0
    np.testing.assert_allclose(r.empirical_coverage, [0.0, 1.0, 1.0])


def test_invalid_pit_values_excluded():
    pit = np.array([0.5, 1.5, -0.1, 0.3])  # two out of [0,1]
    levels = np.array([0.5])
    ids = np.array(["a", "b", "c", "d"])
    r = cal.expected_calibration_error(pit, levels, ids)
    assert r.meta.n_used == 2
    assert r.meta.exclusion_reasons == {"pit_out_of_unit_interval_or_nonfinite": 2}


def test_weights_must_sum_to_one():
    pit = np.array([0.5])
    ids = np.array(["a"])
    with pytest.raises(ValueError):
        cal.expected_calibration_error(pit, np.array([0.1, 0.9]), ids, weights=np.array([0.4, 0.4]))
