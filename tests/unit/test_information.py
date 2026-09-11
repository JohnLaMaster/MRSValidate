import numpy as np
import pytest

from mrsvalidate.analysis import information as info
from mrsvalidate.exceptions import InvalidUncertaintyRepresentationError
from mrsvalidate.results import ResultStatus


def test_to_std_from_variance():
    values = np.array([4.0, 9.0, -1.0])  # last is invalid (negative variance)
    ids = np.array(["a", "b", "c"])
    r = info.to_std(values, info.UncertaintyRepresentation.VARIANCE, ids)
    np.testing.assert_allclose(r.values, [2.0, 3.0])
    assert r.meta.excluded_ids == ("c",)


def test_to_std_from_std_passthrough():
    values = np.array([1.5, 2.5])
    ids = np.array(["a", "b"])
    r = info.to_std(values, info.UncertaintyRepresentation.STD, ids)
    np.testing.assert_allclose(r.values, [1.5, 2.5])


def test_to_std_percent_requires_reference():
    values = np.array([10.0])
    ids = np.array(["a"])
    with pytest.raises(InvalidUncertaintyRepresentationError):
        info.to_std(values, info.UncertaintyRepresentation.PERCENT_STD, ids)


def test_to_std_percent_computed_correctly():
    values = np.array([10.0])  # 10% of reference
    reference = np.array([50.0])
    ids = np.array(["a"])
    r = info.to_std(
        values, info.UncertaintyRepresentation.PERCENT_STD, ids, reference=reference
    )
    assert r.values[0] == pytest.approx(5.0)


def test_to_std_rejects_covariance_representation():
    with pytest.raises(InvalidUncertaintyRepresentationError):
        info.to_std(np.array([1.0]), info.UncertaintyRepresentation.COVARIANCE, np.array(["a"]))


def test_to_std_rejects_raw_string_representation():
    with pytest.raises(InvalidUncertaintyRepresentationError):
        info.to_std(np.array([1.0]), "std", np.array(["a"]))  # type: ignore[arg-type]


def test_crlb_sd_from_covariance():
    cov = np.array([[4.0, 0.5], [0.5, 9.0]])
    sd = info.crlb_sd_from_covariance(cov)
    np.testing.assert_allclose(sd, [2.0, 3.0])


def test_crlb_sd_from_covariance_rejects_asymmetric():
    cov = np.array([[4.0, 0.5], [0.6, 9.0]])
    with pytest.raises(ValueError):
        info.crlb_sd_from_covariance(cov)


def test_crlb_sd_from_covariance_rejects_negative_diagonal():
    cov = np.array([[-4.0, 0.0], [0.0, 9.0]])
    with pytest.raises(ValueError):
        info.crlb_sd_from_covariance(cov)


def test_standardized_error_basic():
    ref = np.array([1.0, 2.0])
    pred = np.array([1.5, 2.0])
    sigma = np.array([0.5, 1.0])
    ids = np.array(["a", "b"])
    r = info.standardized_error(ref, pred, sigma, ids)
    np.testing.assert_allclose(r.values, [1.0, 0.0])
    assert r.meta.configuration["expected_value_under_correct_gaussian_crlb"] == pytest.approx(
        np.sqrt(2 / np.pi)
    )


def test_standardized_error_excludes_nonpositive_sigma():
    ref = np.array([1.0, 2.0])
    pred = np.array([1.5, 2.0])
    sigma = np.array([0.5, 0.0])
    ids = np.array(["a", "b"])
    r = info.standardized_error(ref, pred, sigma, ids)
    assert r.meta.excluded_ids == ("b",)


def test_variance_ratio_basic():
    repeated = np.array([1.0, 1.2, 0.9, 1.1, 1.0])
    sigma_crlb = np.std(repeated, ddof=1)
    r = info.variance_ratio(repeated, sigma_crlb)
    assert r.ratio == pytest.approx(1.0)
    assert r.excess_variance == pytest.approx(0.0, abs=1e-9)


def test_variance_ratio_insufficient_data():
    r = info.variance_ratio(np.array([1.0]), 0.5)
    assert r.meta.status is ResultStatus.INSUFFICIENT_DATA


def test_mahalanobis_squared_error_diagonal_covariance_matches_scaled_euclidean():
    errors = np.array([[1.0, 2.0], [0.5, 0.5]])
    cov = np.diag([1.0, 4.0])
    ids = np.array(["a", "b"])
    r = info.mahalanobis_squared_error(errors, cov, ids)
    expected = np.array([1.0**2 / 1.0 + 2.0**2 / 4.0, 0.5**2 / 1.0 + 0.5**2 / 4.0])
    np.testing.assert_allclose(r.values, expected)


def test_mahalanobis_squared_error_singular_covariance_is_invalid_by_default():
    errors = np.array([[1.0, 2.0]])
    cov = np.array([[1.0, 1.0], [1.0, 1.0]])  # singular
    ids = np.array(["a"])
    r = info.mahalanobis_squared_error(errors, cov, ids)
    assert r.meta.status is ResultStatus.INVALID_INPUT


def test_mahalanobis_squared_error_pseudoinverse_extension_reports_rank():
    errors = np.array([[1.0, 2.0]])
    cov = np.array([[1.0, 1.0], [1.0, 1.0]])  # singular, rank 1
    ids = np.array(["a"])
    r = info.mahalanobis_squared_error(errors, cov, ids, allow_pseudoinverse=True)
    assert r.meta.status is ResultStatus.VALID
    assert r.meta.configuration["effective_ranks"] == [1]
