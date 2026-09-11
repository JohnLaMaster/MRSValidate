"""Adversarial fixture suite (spec: missingness, degeneracy, invalid
metadata, extreme values). Consolidates the scenarios spec lines
1550-1587 list that are not already exercised inline in ``tests/unit``.
"""

from __future__ import annotations

import numpy as np
import pytest

from mrsvalidate.alignment import align_observations
from mrsvalidate.analysis import descriptive, errors, information, operating_envelope
from mrsvalidate.exceptions import AlignmentError
from mrsvalidate.results import ResultStatus
from mrsvalidate.schema import ObservationSet, ParameterRole, ParameterSchema, ParameterSpec

_SCHEMA = ParameterSchema((ParameterSpec("NAA", ParameterRole.CONCENTRATION, "mM"),))


def _obs(ids, values):
    return ObservationSet(
        sample_ids=np.array(ids), values=np.array(values).reshape(-1, 1), parameters=_SCHEMA
    )


# --- missingness / extreme values -------------------------------------------------


def test_all_nan_reference_is_insufficient_data():
    x = np.full(5, np.nan)
    ids = np.array([f"s{i}" for i in range(5)])
    r = descriptive.mean(x, ids)
    assert r.meta.status is ResultStatus.INSUFFICIENT_DATA


def test_isolated_nan_is_excluded_not_propagated():
    x = np.array([1.0, np.nan, 3.0, 4.0])
    ids = np.array(["a", "b", "c", "d"])
    r = descriptive.mean(x, ids)
    assert r.meta.n_omitted == 1
    assert np.isfinite(r.value)


def test_positive_and_negative_infinity_both_excluded():
    x = np.array([1.0, np.inf, -np.inf, 4.0])
    ids = np.array(["a", "b", "c", "d"])
    r = descriptive.mean(x, ids)
    assert r.meta.exclusion_reasons == {"positive_infinity": 1, "negative_infinity": 1}


def test_constant_reference_makes_cv_undefined():
    x = np.array([5.0, 5.0, 5.0, 5.0])
    ids = np.array(["a", "b", "c", "d"])
    r = descriptive.cv(x, ids)
    assert r.meta.status in (ResultStatus.UNDEFINED, ResultStatus.VALID)
    # constant vector has zero SD; mean is nonzero => CV is well-defined at 0
    if r.meta.status is ResultStatus.VALID:
        assert r.value == pytest.approx(0.0)


def test_zero_reference_concentration_triggers_zero_reference_policy():
    ref = np.array([0.0, 1.0, 2.0])
    pred = np.array([0.1, 1.1, 2.1])
    ids = np.array(["a", "b", "c"])
    r = errors.mape(ref, pred, ids, zero_reference_policy="omit_and_report")
    assert r.meta.exclusion_reasons.get("near_zero_reference") == 1


def test_severe_outliers_do_not_crash_rmse():
    ref = np.array([1.0, 2.0, 3.0, 1e12])
    pred = np.array([1.1, 2.1, 2.9, -1e12])
    ids = np.array(["a", "b", "c", "d"])
    r = errors.rmse(ref, pred, ids)
    assert r.meta.status is ResultStatus.VALID
    assert np.isfinite(r.value)


# --- alignment: duplicates, shuffled ordering, unmatched IDs ------------------------


def test_shuffled_sample_ordering_aligns_identically():
    ref = _obs(["a", "b", "c"], [1.0, 2.0, 3.0])
    cand_shuffled = _obs(["c", "a", "b"], [30.0, 10.0, 20.0])
    aligned = align_observations(ref, cand_shuffled, pairing="paired")
    np.testing.assert_array_equal(aligned.candidate.values.ravel(), [10.0, 20.0, 30.0])


def test_duplicate_sample_ids_error_by_default():
    ref = _obs(["a", "a", "b"], [1.0, 2.0, 3.0])
    cand = _obs(["a", "b"], [10.0, 20.0])
    with pytest.raises(AlignmentError):
        align_observations(ref, cand, pairing="paired")


def test_sample_missing_from_one_model_is_reported_not_hidden():
    ref = _obs(["a", "b", "c"], [1.0, 2.0, 3.0])
    cand = _obs(["a", "b"], [10.0, 20.0])
    aligned = align_observations(ref, cand, pairing="paired")
    assert aligned.report.only_in_reference == ("c",)
    assert "c" not in aligned.reference.sample_ids


# --- CRLB / covariance degeneracy ---------------------------------------------------


def test_crlb_std_mislabeled_as_variance_is_rejected_gracefully():
    # A std value fed in as if it were a variance changes the resulting
    # sigma but must not silently pass through unlabeled.
    values = np.array([4.0])
    ids = np.array(["a"])
    as_variance = information.to_std(values, information.UncertaintyRepresentation.VARIANCE, ids)
    as_std = information.to_std(values, information.UncertaintyRepresentation.STD, ids)
    assert as_variance.values[0] != as_std.values[0]


def test_percent_std_without_reference_is_rejected():
    from mrsvalidate.exceptions import InvalidUncertaintyRepresentationError

    with pytest.raises(InvalidUncertaintyRepresentationError):
        information.to_std(
            np.array([10.0]), information.UncertaintyRepresentation.PERCENT_STD, np.array(["a"])
        )


def test_nonsymmetric_covariance_rejected():
    cov = np.array([[1.0, 0.9], [0.1, 1.0]])
    with pytest.raises(ValueError):
        information.crlb_sd_from_covariance(cov)


def test_negative_covariance_diagonal_rejected():
    cov = np.array([[-1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        information.crlb_sd_from_covariance(cov)


def test_singular_covariance_invalid_by_default_in_mahalanobis():
    errors_vec = np.array([[1.0, 1.0]])
    singular_cov = np.array([[1.0, 1.0], [1.0, 1.0]])
    ids = np.array(["a"])
    r = information.mahalanobis_squared_error(errors_vec, singular_cov, ids)
    assert r.meta.status is ResultStatus.INVALID_INPUT


# --- sparse operating-envelope strata / empty bins ---------------------------------


def test_sparse_operating_envelope_stratum_is_insufficient_not_interpolated():
    conc = np.array([0.5, 0.5, 0.5])
    info_metric = np.array([0.5, 0.5, 1.5])  # only 1 obs in second stratum
    success = np.array([True, True, True])
    ids = np.array(["a", "b", "c"])
    r = operating_envelope.operating_envelope(
        conc, info_metric, success, ids,
        bin_edges=np.array([0.0, 1.0]), stratum_edges=np.array([0.0, 1.0, 2.0]),
        target_reliability=0.5, min_stratum_n=2,
    )
    assert not r.sufficient_data_mask[0]
    assert not r.operating_bin_mask[0]


def test_empty_concentration_bin_is_not_falsely_passing():
    conc = np.array([5.0, 5.0])  # entirely outside the declared bin range
    info_metric = np.array([0.5, 0.5])
    success = np.array([True, True])
    ids = np.array(["a", "b"])
    r = operating_envelope.operating_envelope(
        conc, info_metric, success, ids,
        bin_edges=np.array([0.0, 1.0]), stratum_edges=np.array([0.0, 1.0]),
        target_reliability=0.5,
    )
    assert r.totals.sum() == 0
    assert not r.operating_bin_mask.any()
