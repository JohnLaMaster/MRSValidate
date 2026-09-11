import numpy as np
import pytest

from mrsvalidate.analysis import response_curves as rc
from mrsvalidate.results import ResultStatus


def test_response_curve_1d_hand_computed():
    outcome = np.array([1.0, 2.0, 3.0, 10.0, 12.0, 14.0])
    snr = np.array([5.0, 5.0, 5.0, 50.0, 50.0, 50.0])
    ids = np.array([f"s{i}" for i in range(6)])
    r = rc.response_curve(
        outcome, {"mSNR": snr}, ids,
        bin_edges={"mSNR": np.array([0.0, 10.0, 100.0])},
    )
    assert r.axis_names == ("mSNR",)
    assert r.values.shape == (2,)
    assert r.values[0] == pytest.approx(2.0)  # mean of [1,2,3]
    assert r.values[1] == pytest.approx(12.0)  # mean of [10,12,14]
    assert r.counts[0] == 3 and r.counts[1] == 3
    assert r.sufficient_data_mask.all()


def test_response_curve_1d_median_and_rmse_statistics():
    outcome = np.array([1.0, 2.0, 100.0])  # outlier present
    snr = np.array([5.0, 5.0, 5.0])
    ids = np.array(["a", "b", "c"])
    r_median = rc.response_curve(
        outcome, {"mSNR": snr}, ids,
        bin_edges={"mSNR": np.array([0.0, 10.0])}, statistic="median",
    )
    assert r_median.values[0] == pytest.approx(2.0)

    r_rmse = rc.response_curve(
        outcome, {"mSNR": snr}, ids,
        bin_edges={"mSNR": np.array([0.0, 10.0])}, statistic="rmse",
    )
    assert r_rmse.values[0] == pytest.approx(np.sqrt(np.mean(outcome**2)))


def test_response_curve_2d_hand_computed():
    outcome = np.array([1.0, 3.0, 5.0, 7.0])
    snr = np.array([5.0, 5.0, 50.0, 50.0])
    conc = np.array([1.0, 1.0, 1.0, 1.0])
    ids = np.array(["a", "b", "c", "d"])
    r = rc.response_curve(
        outcome, {"mSNR": snr, "concentration": conc}, ids,
        bin_edges={"mSNR": np.array([0.0, 10.0, 100.0]), "concentration": np.array([0.0, 2.0])},
    )
    assert r.axis_names == ("mSNR", "concentration")
    assert r.values.shape == (2, 1)
    assert r.values[0, 0] == pytest.approx(2.0)  # mean of [1,3]
    assert r.values[1, 0] == pytest.approx(6.0)  # mean of [5,7]


def test_response_curve_sparse_bin_marked_insufficient_not_interpolated():
    outcome = np.array([1.0, 2.0])
    snr = np.array([5.0, 95.0])
    ids = np.array(["a", "b"])
    r = rc.response_curve(
        outcome, {"mSNR": snr}, ids,
        bin_edges={"mSNR": np.array([0.0, 10.0, 100.0])}, min_bin_n=2,
    )
    assert not r.sufficient_data_mask.any()
    assert np.all(np.isnan(r.values))


def test_response_curve_excludes_out_of_range_and_nonfinite():
    outcome = np.array([1.0, np.nan, 3.0])
    snr = np.array([5.0, 5.0, 500.0])  # third is out of bin range
    ids = np.array(["a", "b", "c"])
    r = rc.response_curve(outcome, {"mSNR": snr}, ids, bin_edges={"mSNR": np.array([0.0, 10.0])})
    assert r.meta.n_used == 1
    assert set(r.meta.excluded_ids) == {"b", "c"}


def test_response_curve_rejects_more_than_two_axes():
    outcome = np.array([1.0])
    ids = np.array(["a"])
    with pytest.raises(ValueError):
        rc.response_curve(
            outcome, {"x": np.array([1.0]), "y": np.array([1.0]), "z": np.array([1.0])}, ids,
            bin_edges={"x": np.array([0.0, 2.0]), "y": np.array([0.0, 2.0]), "z": np.array([0.0, 2.0])},
        )


def test_response_curve_mismatched_bin_edges_keys_raises():
    outcome = np.array([1.0])
    ids = np.array(["a"])
    with pytest.raises(ValueError):
        rc.response_curve(
            outcome, {"mSNR": np.array([1.0])}, ids, bin_edges={"other": np.array([0.0, 2.0])}
        )
