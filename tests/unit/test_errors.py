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


def test_meape_conventional_is_default_and_has_no_sample_size_factor():
    ref = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    pred = np.array([2.0, 2.0, 5.0, 4.0, 90.0])
    ids = np.array(["a", "b", "c", "d", "e"])
    r = e.meape(ref, pred, ids)
    assert r.meta.configuration["definition"] == "conventional"
    assert r.value == pytest.approx(100.0 * np.median(np.abs(pred - ref) / ref))


def test_meape_rejects_removed_manuscript_literal_definition():
    ref = np.array([1.0, 2.0, 3.0])
    ids = np.array(["a", "b", "c"])
    with pytest.raises(ValueError):
        e.meape(ref, ref, ids, definition="manuscript_literal")


@pytest.mark.parametrize(
    ("factors", "expected"),
    [
        ([1.1, 1.1, 1.1, 1.1], 0.0),     # constant +10 % bias: no spread
        ([0.9, 0.9, 0.9, 0.9], 0.0),     # constant -10 % bias: no spread
        ([0.95, 0.85, 0.95, 0.85], 5.0),  # bias -10 %, spread +/-5 %
        ([1.05, 1.15, 1.05, 1.15], 5.0),  # bias +10 %, spread +/-5 %
    ],
)
def test_bias_corrected_mape_removes_bias_of_either_sign(factors, expected):
    ref = np.array([2.0, 4.0, 5.0, 10.0])
    ids = np.array(["a", "b", "c", "d"])
    r = e.bias_corrected_mape(ref, ref * np.array(factors), ids)
    assert r.value == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    ("factors", "expected"),
    [
        ([1.1, 1.1, 1.1, 1.1, 1.1], 0.0),
        ([0.9, 0.9, 0.9, 0.9, 0.9], 0.0),
        ([0.95, 0.85, 0.95, 0.85, 0.90], 5.0),
        ([1.05, 1.15, 1.05, 1.15, 1.10], 5.0),
    ],
)
def test_bias_corrected_meape_removes_bias_of_either_sign(factors, expected):
    ref = np.array([2.0, 4.0, 5.0, 10.0, 8.0])
    ids = np.array(["a", "b", "c", "d", "e"])
    r = e.meape(ref, ref * np.array(factors), ids, definition="bias_corrected")
    assert r.value == pytest.approx(expected, abs=1e-9)


def test_bias_corrected_mape_is_mean_absolute_deviation_of_percent_errors():
    rng = np.random.default_rng(0)
    ref = rng.uniform(1.0, 10.0, size=200)
    pred = ref * (1.0 + rng.normal(-0.08, 0.05, size=200))
    ids = np.array([f"s{i}" for i in range(200)])
    pct = 100.0 * (pred - ref) / ref
    r = e.bias_corrected_mape(ref, pred, ids)
    assert r.value == pytest.approx(np.mean(np.abs(pct - pct.mean())))


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
