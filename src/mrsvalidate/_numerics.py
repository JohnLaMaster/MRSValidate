"""Shared numeric primitives for the scientific analysis layer.

Nothing here silently patches scientific results: no routine in this
module adds an epsilon to a scientific denominator. Callers decide,
explicitly, what counts as "near zero" for their statistic and record
the resolved tolerance in the result's ``AnalysisMeta.configuration``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

#: Default float64 relative tolerance for the near-zero policy (spec: "the
#: default float64 relative tolerance is approximately 1e-12").
DEFAULT_REL_TOL = 1e-12
DEFAULT_ABS_TOL = 0.0

QuantileMethod = Literal[
    "linear", "lower", "higher", "nearest", "midpoint", "median_unbiased"
]
DEFAULT_QUANTILE_METHOD: QuantileMethod = "linear"


def near_zero_threshold(
    scale: float, *, abs_tol: float = DEFAULT_ABS_TOL, rel_tol: float = DEFAULT_REL_TOL
) -> float:
    r"""Return the scale-aware near-zero threshold for a variance-like quantity.

    The threshold grows with the square of ``scale`` so that a quantity
    is judged "near zero" relative to the magnitude of the data it was
    computed from, rather than against a single fixed constant that
    would be too strict for large-scale data and too lax for
    small-scale data.

    Parameters
    ----------
    scale : float
        Reference data scale for the quantity being tested, e.g. the
        reference standard deviation when testing a variance, or the
        reference mean magnitude when testing a mean-like denominator.
    abs_tol : float, optional
        Absolute tolerance floor. Default is ``0.0``.
    rel_tol : float, optional
        Relative tolerance applied to ``max(1, scale**2)``. Default is
        ``1e-12``, the package's default float64 relative tolerance.

    Returns
    -------
    float
        The threshold below which a value of the tested quantity is
        considered near zero.

    Notes
    -----
    Implements the near-zero policy

    .. math::

        \tau(\text{scale}) = \max\!\big(\text{abs\_tol},\;
        \text{rel\_tol} \cdot \max(1, \text{scale}^2)\big)

    Examples
    --------
    >>> from mrsvalidate._numerics import near_zero_threshold
    >>> near_zero_threshold(1.0)
    1e-12
    >>> near_zero_threshold(1000.0)
    1e-06
    """

    return max(abs_tol, rel_tol * max(1.0, float(scale) ** 2))


def is_near_zero(
    value: float,
    scale: float,
    *,
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
) -> bool:
    """Return whether a variance-like value is at or below its near-zero threshold.

    Parameters
    ----------
    value : float
        The quantity to test, e.g. a variance or a squared denominator.
    scale : float
        Reference data scale used to compute the threshold; see
        :func:`near_zero_threshold`.
    abs_tol : float, optional
        Absolute tolerance floor. Default is ``0.0``.
    rel_tol : float, optional
        Relative tolerance applied to ``max(1, scale**2)``. Default is
        ``1e-12``.

    Returns
    -------
    bool
        ``True`` if ``value <= near_zero_threshold(scale, abs_tol=abs_tol,
        rel_tol=rel_tol)``, ``False`` otherwise.

    See Also
    --------
    near_zero_threshold : Computes the threshold used by this function.

    Examples
    --------
    >>> from mrsvalidate._numerics import is_near_zero
    >>> is_near_zero(1e-15, scale=1.0)
    True
    >>> is_near_zero(0.5, scale=1.0)
    False
    """

    return value <= near_zero_threshold(scale, abs_tol=abs_tol, rel_tol=rel_tol)


def all_finite(*arrays: NDArray[np.float64]) -> bool:
    """Return whether every element of every supplied array is finite.

    Parameters
    ----------
    *arrays : ndarray of float
        One or more arrays of any shape to check for ``NaN`` or
        infinite values.

    Returns
    -------
    bool
        ``True`` if every element of every array in ``arrays`` is
        finite (neither ``NaN`` nor ``+inf``/``-inf``), ``False``
        otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate._numerics import all_finite
    >>> all_finite(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    True
    >>> all_finite(np.array([1.0, np.nan]))
    False
    """

    return all(bool(np.all(np.isfinite(a))) for a in arrays)


def robust_variance(x: NDArray[np.float64]) -> float:
    """Compute the raw median squared deviation from the median.

    Parameters
    ----------
    x : ndarray of float
        One-dimensional array of observed values.

    Returns
    -------
    float
        The median of the squared deviations of ``x`` from its own
        median.

    Notes
    -----
    This is the manuscript's literal "robust variance" quantity,

    .. math::

        \\hat{\\sigma}^2_{\\text{robust}} =
        \\operatorname{median}\\!\\big((x - \\operatorname{median}(x))^2\\big).

    It is deliberately **not** rescaled by the usual normal-consistency
    constant (``1.4826**2``) — callers that want the normal-consistent
    robust variance/SD must apply that scaling themselves and name the
    result accordingly, since this function reproduces the manuscript
    quantity exactly.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate._numerics import robust_variance
    >>> robust_variance(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    1.0
    """

    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    return float(np.median((x - med) ** 2))


def robust_sd(x: NDArray[np.float64]) -> float:
    """Compute the square root of the raw median squared deviation.

    Parameters
    ----------
    x : ndarray of float
        One-dimensional array of observed values.

    Returns
    -------
    float
        The square root of :func:`robust_variance` applied to ``x``.

    See Also
    --------
    robust_variance : The un-rooted quantity this function takes the
        square root of, including the normal-consistency caveat.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate._numerics import robust_sd
    >>> robust_sd(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    1.0
    """

    return float(np.sqrt(robust_variance(x)))


def iqr(
    x: NDArray[np.float64], *, method: QuantileMethod = DEFAULT_QUANTILE_METHOD
) -> float:
    """Compute the interquartile range of a sample.

    Parameters
    ----------
    x : ndarray of float
        One-dimensional array of observed values.
    method : {"linear", "lower", "higher", "nearest", "midpoint", "median_unbiased"}, optional
        Quantile interpolation method forwarded to :func:`numpy.quantile`.
        Default is ``"linear"``.

    Returns
    -------
    float
        The interquartile range :math:`Q_{0.75} - Q_{0.25}`, computed
        with the given quantile method.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate._numerics import iqr
    >>> iqr(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    2.0
    """

    x = np.asarray(x, dtype=np.float64)
    q1, q3 = np.quantile(x, [0.25, 0.75], method=method)
    return float(q3 - q1)


def tukey_outlier_range(
    x: NDArray[np.float64], *, method: QuantileMethod = DEFAULT_QUANTILE_METHOD
) -> tuple[float, float]:
    """Compute Tukey's outlier range for a sample.

    Parameters
    ----------
    x : ndarray of float
        One-dimensional array of observed values.
    method : {"linear", "lower", "higher", "nearest", "midpoint", "median_unbiased"}, optional
        Quantile interpolation method forwarded to :func:`numpy.quantile`.
        Default is ``"linear"``.

    Returns
    -------
    tuple of float
        A ``(low, high)`` pair giving the Tukey outlier range.

    Notes
    -----
    The range is defined as

    .. math::

        [\\,Q_1 - 1.5 \\cdot \\text{IQR},\\ \\ Q_3 + 1.5 \\cdot \\text{IQR}\\,]

    where :math:`Q_1` and :math:`Q_3` are the first and third quartiles
    and :math:`\\text{IQR} = Q_3 - Q_1`. Values outside this range are
    conventionally flagged as outliers.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate._numerics import tukey_outlier_range
    >>> tukey_outlier_range(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    (-1.0, 7.0)
    """

    x = np.asarray(x, dtype=np.float64)
    q1, q3 = np.quantile(x, [0.25, 0.75], method=method)
    span = q3 - q1
    return (float(q1 - 1.5 * span), float(q3 + 1.5 * span))
