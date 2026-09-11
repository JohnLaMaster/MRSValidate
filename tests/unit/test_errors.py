import numpy as np
import pytest

from mrsvalidate.analysis import errors as e
from mrsvalidate.results import ResultStatus

IDS = np.array(["a", "b", "c"])


def test_perfect_identity_rmse_zero():
    ref = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.0, 2.0, 3.0])
    assert e.rmse(ref, pred, IDS).value == pytest.approx(0.0)
    assert e.rmese(ref, pred, IDS).value == pytest.approx(0.0)
    assert e.mpe(ref, pred, IDS).value == pytest.approx(0.0)
    assert e.mape(ref, pred, IDS).value == pytest.approx(0.0)


def test_rmse_positive_for_constant_bias():
    ref = np.array([1.0, 2.0, 3.0])
    pred = np.array([2.0, 3.0, 4.0])
    assert e.rmse(ref, pred, IDS).value > 0.0


def test_mpe_sign_convention_overestimation_positive():
    ref = np.array([10.0, 10.0, 10.0])
    pred = np.array([11.0, 11.0, 11.0])  # +10% each
    r = e.mpe(ref, pred, IDS)
    assert r.value == pytest.approx(10.0)


def test_mape_manuscript_literal_vs_absolute_reference_differ_for_negative_reference():
    ref = np.array([-10.0, 10.0])
    pred = np.array([-9.0, 11.0])  # errors: +1 and +1
    ids = np.array(["a", "b"])
    literal = e.mape(ref, pred, ids, definition="manuscript_literal")
    absref = e.mape(ref, pred, ids, definition="absolute_reference")
    # literal denominator is signed ref: |1|/-10 = -0.1 -> contributes -0.1 before *100
    expected_literal = 100.0 * np.mean(np.abs(pred - ref) / ref)
    expected_absref = 100.0 * np.mean(np.abs(pred - ref) / np.abs(ref))
    assert literal.value == pytest.approx(expected_literal)
    assert absref.value == pytest.approx(expected_absref)
    assert literal.value != absref.value


def test_meape_literal_and_corrected_are_distinct():
    ref = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    pred = np.array([2.0, 2.0, 5.0, 4.0, 90.0])
    ids = np.array(["a", "b", "c", "d", "e"])

    literal = e.meape(ref, pred, ids, definition="manuscript_literal")
    conventional = e.meape(ref, pred, ids, definition="conventional")

    assert literal.meta.configuration["definition"] != conventional.meta.configuration[
        "definition"
    ]
    assert literal.value != conventional.value
    # literal carries an extra 1/N factor relative to the conventional definition
    n = len(ref)
    assert literal.value == pytest.approx(conventional.value / n)


def test_bias_corrected_mape_literal_and_absolute_centered_are_distinct():
    # Mixed-sign errors with unequal magnitudes so mean(centered) != mean(|centered|).
    ref = np.array([1.0, 1.0, 1.0, 1.0])
    pred = np.array([4.0, 1.0, 1.0, -1.0])
    ids = np.array(["a", "b", "c", "d"])

    literal = e.bias_corrected_mape(ref, pred, ids, definition="manuscript_literal")
    centered = e.bias_corrected_mape(ref, pred, ids, definition="absolute_centered")

    assert literal.value == pytest.approx(100.0)
    assert centered.value == pytest.approx(125.0)
    assert literal.value != centered.value


def test_zero_reference_omit_and_report_excludes_row():
    ref = np.array([0.0, 1.0, 2.0])
    pred = np.array([0.1, 1.1, 2.1])
    ids = np.array(["a", "b", "c"])
    r = e.mpe(ref, pred, ids, zero_reference_policy="omit_and_report")
    assert r.meta.status is ResultStatus.VALID
    assert r.meta.exclusion_reasons.get("near_zero_reference") == 1
    assert "a" in r.meta.excluded_ids


def test_zero_reference_error_policy_marks_invalid_input():
    ref = np.array([0.0, 1.0, 2.0])
    pred = np.array([0.1, 1.1, 2.1])
    ids = np.array(["a", "b", "c"])
    r = e.mpe(ref, pred, ids, zero_reference_policy="error")
    assert r.meta.status is ResultStatus.INVALID_INPUT
    assert np.isnan(r.value)


def test_nan_rows_excluded_before_zero_reference_check():
    ref = np.array([np.nan, 1.0, 2.0])
    pred = np.array([0.1, 1.1, 2.1])
    ids = np.array(["a", "b", "c"])
    r = e.rmse(ref, pred, ids)
    assert r.meta.n_omitted == 1
    assert r.meta.exclusion_reasons == {"nan": 1}
