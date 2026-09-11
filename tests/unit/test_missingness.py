import numpy as np

from mrsvalidate.missingness import analyze_missingness


def test_analyze_missingness_all_finite():
    ids = np.array(["a", "b", "c"])
    report = analyze_missingness(ids, np.array([1.0, 2.0, 3.0]))
    assert report.valid_mask.tolist() == [True, True, True]
    assert report.included_ids == ("a", "b", "c")
    assert report.excluded_ids == ()
    assert report.exclusion_reasons == {}


def test_analyze_missingness_isolated_nan():
    ids = np.array(["a", "b", "c"])
    report = analyze_missingness(ids, np.array([1.0, np.nan, 3.0]))
    assert report.valid_mask.tolist() == [True, False, True]
    assert report.included_ids == ("a", "c")
    assert report.excluded_ids == ("b",)
    assert report.exclusion_reasons == {"nan": 1}


def test_analyze_missingness_all_nan():
    ids = np.array(["a", "b"])
    report = analyze_missingness(ids, np.array([np.nan, np.nan]))
    assert report.included_ids == ()
    assert report.exclusion_reasons == {"nan": 2}


def test_analyze_missingness_distinguishes_signed_infinity():
    ids = np.array(["a", "b", "c"])
    report = analyze_missingness(ids, np.array([np.inf, -np.inf, 1.0]))
    assert report.exclusion_reasons == {"positive_infinity": 1, "negative_infinity": 1}


def test_analyze_missingness_across_multiple_columns_any_invalid_excludes_row():
    ids = np.array(["a", "b"])
    reference = np.array([1.0, 2.0])
    prediction = np.array([1.0, np.nan])
    report = analyze_missingness(ids, reference, prediction)
    assert report.included_ids == ("a",)
    assert report.exclusion_reasons == {"nan": 1}


def test_analyze_missingness_reason_priority_nan_over_infinity():
    ids = np.array(["a"])
    report = analyze_missingness(ids, np.array([np.nan]), np.array([np.inf]))
    assert report.exclusion_reasons == {"nan": 1}
