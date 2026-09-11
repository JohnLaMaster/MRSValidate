"""Core descriptive statistics for manuscript-defined validation metrics.

Mean, median, variance, robust variance/SD, interquartile range (IQR),
Tukey's outlier range, coefficient of variation (CV) and robust CV,
each computed over a single batch of per-sample values.

Every function accepts a single 1D array of values plus its aligned
``sample_ids`` and returns a typed
:class:`~mrsvalidate.results.AnalysisResult` subclass carrying the
exact rows used. Non-finite values are excluded row-wise via
:func:`mrsvalidate.missingness.analyze_missingness` before any
computation.

Notes
-----
No denominator in this module is ever epsilon-patched. When a
denominator (the sample mean for :func:`cv`, the sample median for
:func:`robust_cv`) is effectively zero relative to the corresponding
scale, the statistic is reported as ``ResultStatus.UNDEFINED`` with
``value = nan`` instead of a huge or NaN-masked number. See
:func:`mrsvalidate._numerics.is_near_zero` for the exact scale-aware
near-zero threshold.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .._numerics import (
    DEFAULT_ABS_TOL,
    DEFAULT_REL_TOL,
    DEFAULT_QUANTILE_METHOD,
    QuantileMethod,
    is_near_zero,
    iqr as _iqr,
    robust_sd as _robust_sd,
    robust_variance as _robust_variance,
    tukey_outlier_range as _tukey_outlier_range,
)
from ..results import IntervalResult, ResultStatus, ScalarMetricResult
from ._shared import meta_for, valid_subset

__all__ = [
    "mean",
    "median",
    "variance",
    "robust_variance",
    "robust_sd",
    "iqr",
    "tukey_outlier_range",
    "cv",
    "robust_cv",
]


def mean(x: NDArray[np.float64], sample_ids: NDArray[np.str_]) -> ScalarMetricResult:
    """Arithmetic mean of `x` over the valid (finite) sample subset.

    Parameters
    ----------
    x : ndarray of float
        One value per sample (e.g. a metabolite concentration
        estimate). Rows containing ``NaN``, ``+inf`` or ``-inf`` are
        excluded before the mean is computed.
    sample_ids : ndarray of str
        Sample identifier aligned with `x`, same length. Recorded in
        the returned result's ``meta.included_ids`` /
        ``meta.excluded_ids``.

    Returns
    -------
    result : ScalarMetricResult
        ``result.value`` is ``mean(x)`` over the included rows.
        ``result.meta.status`` is ``ResultStatus.VALID`` unless every
        row was excluded, in which case it is
        ``ResultStatus.INSUFFICIENT_DATA`` and ``result.value`` is
        ``nan``.

    Notes
    -----
    .. math::

        \\bar{x} = \\frac{1}{n}\\sum_{i=1}^n x_i

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.descriptive import mean
    >>> x = np.array([1.0, 2.0, 3.0, np.nan])
    >>> ids = np.array(["a", "b", "c", "d"])
    >>> r = mean(x, ids)
    >>> r.value
    2.0
    >>> r.meta.n_used, r.meta.n_omitted
    (3, 1)
    """

    all_ids, report, (valid_x,) = valid_subset(sample_ids, x)
    meta = meta_for(all_ids=all_ids, report=report, method="mean")
    value = float(np.mean(valid_x)) if meta.status is ResultStatus.VALID else float("nan")
    return ScalarMetricResult(meta=meta, value=value)


def median(x: NDArray[np.float64], sample_ids: NDArray[np.str_]) -> ScalarMetricResult:
    """Median of `x` over the valid (finite) sample subset.

    Parameters
    ----------
    x : ndarray of float
        One value per sample. Rows containing ``NaN``, ``+inf`` or
        ``-inf`` are excluded before the median is computed.
    sample_ids : ndarray of str
        Sample identifier aligned with `x`, same length.

    Returns
    -------
    result : ScalarMetricResult
        ``result.value`` is ``median(x)`` over the included rows;
        ``ResultStatus.INSUFFICIENT_DATA`` (``value = nan``) if no
        valid rows remain.

    Notes
    -----
    .. math::

        \\operatorname{median}(x)

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.descriptive import median
    >>> x = np.array([1.0, 2.0, 100.0])
    >>> ids = np.array(["a", "b", "c"])
    >>> median(x, ids).value
    2.0
    """

    all_ids, report, (valid_x,) = valid_subset(sample_ids, x)
    meta = meta_for(all_ids=all_ids, report=report, method="median")
    value = float(np.median(valid_x)) if meta.status is ResultStatus.VALID else float("nan")
    return ScalarMetricResult(meta=meta, value=value)


def variance(x: NDArray[np.float64], sample_ids: NDArray[np.str_]) -> ScalarMetricResult:
    """Sample variance with Bessel's correction (``ddof=1``).

    Requires at least two valid observations to be well-defined; with
    fewer than two the statistic is reported as insufficient rather
    than guessed at.

    Parameters
    ----------
    x : ndarray of float
        One value per sample.
    sample_ids : ndarray of str
        Sample identifier aligned with `x`.

    Returns
    -------
    result : ScalarMetricResult
        ``result.value`` is the Bessel-corrected sample variance.
        ``ResultStatus.INSUFFICIENT_DATA`` (``value = nan``) if fewer
        than 2 valid rows remain.

    Notes
    -----
    .. math::

        s^2 = \\frac{1}{n-1}\\sum_{i=1}^n (x_i - \\bar{x})^2

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.descriptive import variance
    >>> x = np.array([1.0, 2.0, 3.0, 4.0])
    >>> ids = np.array(["a", "b", "c", "d"])
    >>> variance(x, ids).value
    1.6666666666666667
    """

    all_ids, report, (valid_x,) = valid_subset(sample_ids, x)
    meta = meta_for(all_ids=all_ids, report=report, method="variance", min_n=2)
    value = (
        float(np.var(valid_x, ddof=1)) if meta.status is ResultStatus.VALID else float("nan")
    )
    return ScalarMetricResult(meta=meta, value=value)


def robust_variance(
    x: NDArray[np.float64], sample_ids: NDArray[np.str_]
) -> ScalarMetricResult:
    """Manuscript's raw robust variance, ``median((x - median(x))**2)``.

    Named explicitly as a raw median squared deviation rather than the
    usual normal-consistent robust variance estimator, so that it
    reproduces the manuscript's quantity exactly (no ``1.4826**2``
    scaling applied).

    Parameters
    ----------
    x : ndarray of float
        One value per sample.
    sample_ids : ndarray of str
        Sample identifier aligned with `x`.

    Returns
    -------
    result : ScalarMetricResult
        ``result.value`` is ``median((x - median(x))**2)`` over the
        included rows.

    Notes
    -----
    .. math::

        \\operatorname{robust\\_var}(x) =
            \\operatorname{median}\\left((x_i - \\operatorname{median}(x))^2\\right)

    This is *not* rescaled by the ``1.4826**2`` normal-consistency
    constant used by some robust-statistics conventions; callers who
    want that normal-consistent quantity must apply the scaling
    themselves and name the result accordingly.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.descriptive import robust_variance
    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    >>> ids = np.array(["a", "b", "c", "d", "e"])
    >>> robust_variance(x, ids).value
    1.0
    """

    all_ids, report, (valid_x,) = valid_subset(sample_ids, x)
    meta = meta_for(all_ids=all_ids, report=report, method="robust_variance")
    value = (
        _robust_variance(valid_x) if meta.status is ResultStatus.VALID else float("nan")
    )
    return ScalarMetricResult(meta=meta, value=value)


def robust_sd(x: NDArray[np.float64], sample_ids: NDArray[np.str_]) -> ScalarMetricResult:
    """Robust standard deviation, ``sqrt(robust_variance(x))``.

    Parameters
    ----------
    x : ndarray of float
        One value per sample.
    sample_ids : ndarray of str
        Sample identifier aligned with `x`.

    Returns
    -------
    result : ScalarMetricResult
        ``result.value`` is the square root of the manuscript's raw
        robust variance (see :func:`robust_variance`).

    Notes
    -----
    .. math::

        \\operatorname{robust\\_sd}(x) =
            \\sqrt{\\operatorname{median}\\left((x_i - \\operatorname{median}(x))^2\\right)}

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.descriptive import robust_sd
    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    >>> ids = np.array(["a", "b", "c", "d", "e"])
    >>> robust_sd(x, ids).value
    1.0
    """

    all_ids, report, (valid_x,) = valid_subset(sample_ids, x)
    meta = meta_for(all_ids=all_ids, report=report, method="robust_sd")
    value = _robust_sd(valid_x) if meta.status is ResultStatus.VALID else float("nan")
    return ScalarMetricResult(meta=meta, value=value)


def iqr(
    x: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
    *,
    quantile_method: QuantileMethod = DEFAULT_QUANTILE_METHOD,
) -> ScalarMetricResult:
    """Interquartile range, ``Q0.75 - Q0.25``, of `x`.

    Parameters
    ----------
    x : ndarray of float
        One value per sample.
    sample_ids : ndarray of str
        Sample identifier aligned with `x`.
    quantile_method : {"linear", "lower", "higher", "nearest", \
"midpoint", "median_unbiased"}, optional
        Interpolation method passed through to :func:`numpy.quantile`.
        Default is ``"linear"``.

    Returns
    -------
    result : ScalarMetricResult
        ``result.value`` is ``Q3 - Q1`` computed on the included rows;
        ``result.meta.configuration["quantile_method"]`` records the
        method actually used.

    Notes
    -----
    .. math::

        \\text{IQR}(x) = Q_{0.75}(x) - Q_{0.25}(x)

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.descriptive import iqr
    >>> x = np.array([1.0, 2.0, 3.0, 4.0])
    >>> ids = np.array(["a", "b", "c", "d"])
    >>> iqr(x, ids).value
    1.5
    """

    all_ids, report, (valid_x,) = valid_subset(sample_ids, x)
    meta = meta_for(
        all_ids=all_ids,
        report=report,
        method="iqr",
        configuration={"quantile_method": quantile_method},
    )
    value = (
        _iqr(valid_x, method=quantile_method)
        if meta.status is ResultStatus.VALID
        else float("nan")
    )
    return ScalarMetricResult(meta=meta, value=value)


def tukey_outlier_range(
    x: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
    *,
    quantile_method: QuantileMethod = DEFAULT_QUANTILE_METHOD,
) -> IntervalResult:
    """Tukey's outlier range, ``[Q1 - 1.5*IQR, Q3 + 1.5*IQR]``, of `x`.

    Parameters
    ----------
    x : ndarray of float
        One value per sample.
    sample_ids : ndarray of str
        Sample identifier aligned with `x`.
    quantile_method : {"linear", "lower", "higher", "nearest", \
"midpoint", "median_unbiased"}, optional
        Interpolation method passed through to :func:`numpy.quantile`.
        Default is ``"linear"``.

    Returns
    -------
    result : IntervalResult
        ``result.low`` and ``result.high`` are the lower/upper Tukey
        fences; values of `x` outside ``[low, high]`` are
        conventionally flagged as outliers.

    Notes
    -----
    .. math::

        \\left[\\, Q_1 - 1.5 \\cdot \\text{IQR},\\ \\ Q_3 + 1.5 \\cdot \\text{IQR} \\,\\right]

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.descriptive import tukey_outlier_range
    >>> x = np.array([1.0, 2.0, 3.0, 4.0])
    >>> ids = np.array(["a", "b", "c", "d"])
    >>> r = tukey_outlier_range(x, ids)
    >>> r.low, r.high
    (-0.5, 5.5)
    """

    all_ids, report, (valid_x,) = valid_subset(sample_ids, x)
    meta = meta_for(
        all_ids=all_ids,
        report=report,
        method="tukey_outlier_range",
        configuration={"quantile_method": quantile_method},
    )
    if meta.status is ResultStatus.VALID:
        low, high = _tukey_outlier_range(valid_x, method=quantile_method)
    else:
        low = high = float("nan")
    return IntervalResult(meta=meta, low=low, high=high)


def cv(
    x: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
    *,
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
) -> ScalarMetricResult:
    """Coefficient of variation, ``SD(x) / mean(x)``.

    Undefined when the sample mean is effectively zero relative to the
    sample SD (near-zero policy applied to the squared mean, scale =
    the sample SD), rather than returning a huge or sign-flipped
    ratio.

    Parameters
    ----------
    x : ndarray of float
        One value per sample.
    sample_ids : ndarray of str
        Sample identifier aligned with `x`.
    abs_tol : float, optional
        Absolute near-zero tolerance applied to ``mean(x)**2``.
        Default is ``mrsvalidate._numerics.DEFAULT_ABS_TOL`` (``0.0``).
    rel_tol : float, optional
        Relative near-zero tolerance applied to ``mean(x)**2``, scaled
        by ``max(1, SD(x)**2)``. Default is
        ``mrsvalidate._numerics.DEFAULT_REL_TOL`` (``1e-12``).

    Returns
    -------
    result : ScalarMetricResult
        ``result.value`` is ``SD(x) / mean(x)`` (``ddof=1``).
        ``result.meta.status`` is ``ResultStatus.UNDEFINED``
        (``value = nan``) when ``mean(x)`` is effectively zero
        relative to ``SD(x)``, or ``ResultStatus.INSUFFICIENT_DATA``
        when fewer than two valid rows remain.

    Notes
    -----
    .. math::

        \\text{CV}(x) = \\frac{s}{\\bar{x}}, \\qquad
        s = \\sqrt{\\frac{1}{n-1}\\sum_{i=1}^n (x_i-\\bar{x})^2}

    No epsilon is added to ``mean(x)`` to avoid division by zero;
    instead the near-zero condition is reported explicitly via
    ``ResultStatus.UNDEFINED`` and ``meta.warnings``.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.descriptive import cv
    >>> x = np.array([8.0, 9.0, 10.0, 11.0, 12.0])
    >>> ids = np.array(["a", "b", "c", "d", "e"])
    >>> cv(x, ids).value
    0.15811388300841897

    A mean of (effectively) zero is undefined rather than a huge
    ratio:

    >>> x_zero_mean = np.array([-1.0, 1.0, -2.0, 2.0])
    >>> r = cv(x_zero_mean, np.array(["a", "b", "c", "d"]))
    >>> r.meta.status.value
    'undefined'
    >>> r.value
    nan
    """

    all_ids, report, (valid_x,) = valid_subset(sample_ids, x)
    meta_base = meta_for(all_ids=all_ids, report=report, method="cv", min_n=2)
    if meta_base.status is not ResultStatus.VALID:
        return ScalarMetricResult(meta=meta_base, value=float("nan"))

    centroid = float(np.mean(valid_x))
    scale = float(np.std(valid_x, ddof=1))
    threshold = max(abs_tol, rel_tol * max(1.0, scale**2))
    if is_near_zero(centroid**2, scale, abs_tol=abs_tol, rel_tol=rel_tol):
        meta = meta_for(
            all_ids=all_ids,
            report=report,
            method="cv",
            status=ResultStatus.UNDEFINED,
            configuration={
                "near_zero_abs_tol": abs_tol,
                "near_zero_rel_tol": rel_tol,
                "near_zero_threshold": threshold,
                "centroid": centroid,
            },
            warnings=("CV undefined: mean is effectively zero relative to SD",),
        )
        return ScalarMetricResult(meta=meta, value=float("nan"))

    meta = meta_for(
        all_ids=all_ids,
        report=report,
        method="cv",
        configuration={
            "near_zero_abs_tol": abs_tol,
            "near_zero_rel_tol": rel_tol,
            "near_zero_threshold": threshold,
        },
    )
    return ScalarMetricResult(meta=meta, value=scale / centroid)


def robust_cv(
    x: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
    *,
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
) -> ScalarMetricResult:
    """Robust coefficient of variation, ``robust_sd(x) / median(x)``.

    Undefined when the sample median is effectively zero relative to
    the raw robust SD (near-zero policy applied to the squared
    median, scale = :func:`robust_sd`), rather than returning a huge
    or sign-flipped ratio.

    Parameters
    ----------
    x : ndarray of float
        One value per sample.
    sample_ids : ndarray of str
        Sample identifier aligned with `x`.
    abs_tol : float, optional
        Absolute near-zero tolerance applied to ``median(x)**2``.
        Default is ``mrsvalidate._numerics.DEFAULT_ABS_TOL`` (``0.0``).
    rel_tol : float, optional
        Relative near-zero tolerance applied to ``median(x)**2``,
        scaled by ``max(1, robust_sd(x)**2)``. Default is
        ``mrsvalidate._numerics.DEFAULT_REL_TOL`` (``1e-12``).

    Returns
    -------
    result : ScalarMetricResult
        ``result.value`` is the manuscript's raw robust SD divided by
        the sample median. ``result.meta.status`` is
        ``ResultStatus.UNDEFINED`` (``value = nan``) when the median
        is effectively zero relative to the robust SD.

    Notes
    -----
    .. math::

        \\text{robust\\_CV}(x) = \\frac{\\operatorname{robust\\_sd}(x)}{\\operatorname{median}(x)}

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.descriptive import robust_cv
    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    >>> ids = np.array(["a", "b", "c", "d", "e"])
    >>> robust_cv(x, ids).value
    0.3333333333333333

    A median of (effectively) zero is undefined rather than a huge or
    sign-flipped ratio:

    >>> x_zero_median = np.array([-1.0, 0.0, 1.0])
    >>> r = robust_cv(x_zero_median, np.array(["a", "b", "c"]))
    >>> r.meta.status.value
    'undefined'
    """

    all_ids, report, (valid_x,) = valid_subset(sample_ids, x)
    meta_base = meta_for(all_ids=all_ids, report=report, method="robust_cv")
    if meta_base.status is not ResultStatus.VALID:
        return ScalarMetricResult(meta=meta_base, value=float("nan"))

    centroid = float(np.median(valid_x))
    scale = _robust_sd(valid_x)
    threshold = max(abs_tol, rel_tol * max(1.0, scale**2))
    if is_near_zero(centroid**2, scale, abs_tol=abs_tol, rel_tol=rel_tol):
        meta = meta_for(
            all_ids=all_ids,
            report=report,
            method="robust_cv",
            status=ResultStatus.UNDEFINED,
            configuration={
                "near_zero_abs_tol": abs_tol,
                "near_zero_rel_tol": rel_tol,
                "near_zero_threshold": threshold,
                "centroid": centroid,
            },
            warnings=("robust CV undefined: median is effectively zero relative to robust SD",),
        )
        return ScalarMetricResult(meta=meta, value=float("nan"))

    meta = meta_for(
        all_ids=all_ids,
        report=report,
        method="robust_cv",
        configuration={
            "near_zero_abs_tol": abs_tol,
            "near_zero_rel_tol": rel_tol,
            "near_zero_threshold": threshold,
        },
    )
    return ScalarMetricResult(meta=meta, value=scale / centroid)
