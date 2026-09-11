import numpy as np
import pytest

from mrsvalidate.analysis import descriptive as d
from mrsvalidate.results import ResultStatus

IDS = np.array(["a", "b", "c", "d", "e"])
X = np.array([1.0, 2.0, 3.0, 4.0, 100.0])


def test_mean_median():
    assert d.mean(X, IDS).value == pytest.approx(np.mean(X))
    assert d.median(X, IDS).value == pytest.approx(np.median(X))


def test_variance_bessel_corrected():
    r = d.variance(X, IDS)
    assert r.value == pytest.approx(np.var(X, ddof=1))


def test_variance_insufficient_data_for_single_point():
    r = d.variance(np.array([1.0]), np.array(["a"]))
    assert r.meta.status is ResultStatus.INSUFFICIENT_DATA
    assert np.isnan(r.value)


def test_robust_variance_and_sd_hand_computed():
    med = np.median(X)
    expected_var = np.median((X - med) ** 2)
    rv = d.robust_variance(X, IDS)
    rsd = d.robust_sd(X, IDS)
    assert rv.value == pytest.approx(expected_var)
    assert rsd.value == pytest.approx(np.sqrt(expected_var))


def test_iqr_and_tukey_range():
    q1, q3 = np.quantile(X, [0.25, 0.75])
    r_iqr = d.iqr(X, IDS)
    assert r_iqr.value == pytest.approx(q3 - q1)

    r_tukey = d.tukey_outlier_range(X, IDS)
    span = q3 - q1
    assert r_tukey.low == pytest.approx(q1 - 1.5 * span)
    assert r_tukey.high == pytest.approx(q3 + 1.5 * span)


def test_cv_normal_case():
    r = d.cv(X, IDS)
    assert r.meta.status is ResultStatus.VALID
    assert r.value == pytest.approx(np.std(X, ddof=1) / np.mean(X))


def test_cv_undefined_for_near_zero_mean():
    x = np.array([-1.0, 1.0, -2.0, 2.0])  # mean == 0
    ids = np.array(["a", "b", "c", "d"])
    r = d.cv(x, ids)
    assert r.meta.status is ResultStatus.UNDEFINED
    assert np.isnan(r.value)
    assert "near_zero_threshold" in r.meta.configuration


def test_robust_cv_undefined_for_near_zero_median():
    x = np.array([-1.0, 0.0, 1.0])
    ids = np.array(["a", "b", "c"])
    r = d.robust_cv(x, ids)
    assert r.meta.status is ResultStatus.UNDEFINED


def test_descriptive_excludes_nan_rows_and_reports_them():
    x = np.array([1.0, np.nan, 3.0])
    ids = np.array(["a", "b", "c"])
    r = d.mean(x, ids)
    assert r.meta.n_used == 2
    assert r.meta.n_omitted == 1
    assert r.meta.excluded_ids == ("b",)
    assert r.meta.exclusion_reasons == {"nan": 1}
    assert r.value == pytest.approx(2.0)


def test_all_invalid_is_insufficient_data():
    x = np.array([np.nan, np.nan])
    ids = np.array(["a", "b"])
    r = d.mean(x, ids)
    assert r.meta.status is ResultStatus.INSUFFICIENT_DATA
    assert np.isnan(r.value)
