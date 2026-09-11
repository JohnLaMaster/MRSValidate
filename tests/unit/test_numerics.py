import numpy as np
import pytest

from mrsvalidate._numerics import (
    is_near_zero,
    iqr,
    near_zero_threshold,
    robust_sd,
    robust_variance,
    tukey_outlier_range,
)


def test_near_zero_threshold_uses_relative_scale():
    assert near_zero_threshold(1000.0, rel_tol=1e-12) == 1e-12 * 1000.0**2
    assert near_zero_threshold(0.0, rel_tol=1e-12) == 1e-12  # max(1, 0) == 1


def test_is_near_zero():
    assert is_near_zero(1e-20, scale=1.0, rel_tol=1e-12)
    assert not is_near_zero(1.0, scale=1.0, rel_tol=1e-12)


def test_robust_variance_matches_hand_computation():
    x = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    med = np.median(x)
    expected = np.median((x - med) ** 2)
    assert robust_variance(x) == pytest.approx(expected)
    assert robust_sd(x) == pytest.approx(np.sqrt(expected))


def test_iqr_and_tukey_range():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    q1, q3 = np.quantile(x, [0.25, 0.75])
    assert iqr(x) == pytest.approx(q3 - q1)
    lo, hi = tukey_outlier_range(x)
    span = q3 - q1
    assert lo == pytest.approx(q1 - 1.5 * span)
    assert hi == pytest.approx(q3 + 1.5 * span)
