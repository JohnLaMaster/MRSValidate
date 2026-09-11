"""Reference-derived standardization (spec eqs. 47-49 / manuscript
``standardized_pooled``).

Location and scale are always derived from the **reference** array
only, then applied to both reference and prediction — this is what
makes values from different metabolites/outputs comparable when
pooled, and prevents the prediction's own (possibly biased)
distribution from leaking into the standardization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .._numerics import DEFAULT_ABS_TOL, DEFAULT_REL_TOL, is_near_zero, iqr as _iqr
from ..results import AnalysisResult, ResultStatus, build_meta
from ._shared import meta_for, valid_subset

__all__ = ["standardize", "StandardizationResult"]

StandardizationMethod = Literal["normal", "robust"]


@dataclass(frozen=True)
class StandardizationResult(AnalysisResult):
    """Result of standardizing `reference`/`prediction` against `reference`.

    Attributes
    ----------
    meta : AnalysisMeta
        Provenance and inclusion/exclusion record (inherited from
        :class:`~mrsvalidate.results.AnalysisResult`).
    z_reference : ndarray of float
        Standardized reference values, ``(reference - location) /
        scale``, aligned with ``meta.included_ids``. Empty when
        ``meta.status`` is not ``ResultStatus.VALID``.
    z_prediction : ndarray of float
        Standardized prediction values, ``(prediction - location) /
        scale``, aligned with ``meta.included_ids`` (same location and
        scale as `z_reference`, both derived from `reference` only).
        Empty when ``meta.status`` is not ``ResultStatus.VALID``.
    location : float
        The reference-derived location actually applied to both
        arrays (the reference mean for ``method="normal"``, the
        reference median for ``method="robust"``). ``nan`` only when
        too few valid rows remain to compute it.
    scale : float
        The reference-derived scale actually applied to both arrays
        (the reference SD for ``method="normal"``, the reference IQR
        for ``method="robust"``).
    method : {"normal", "robust"}
        Which location/scale convention was used.

    Notes
    -----
    ``z_reference``/``z_prediction`` are aligned with
    ``meta.included_ids``; ``location``/``scale`` are the
    reference-derived parameters actually applied to both arrays.
    """

    z_reference: NDArray[np.float64]
    z_prediction: NDArray[np.float64]
    location: float
    scale: float
    method: StandardizationMethod


def standardize(
    reference: NDArray[np.float64],
    prediction: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
    *,
    method: StandardizationMethod = "normal",
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
) -> StandardizationResult:
    """Standardize `reference` and `prediction` using reference-only location/scale.

    Computes a location and scale from `reference` alone (mean/SD for
    ``method="normal"``, median/IQR for ``method="robust"``) and
    applies the same two numbers to both `reference` and `prediction`.
    Deriving both parameters exclusively from `reference` is what
    makes the resulting z-scores comparable across different
    metabolites or output variables when pooled, since the
    prediction's own (possibly biased) distribution never influences
    the standardization.

    Parameters
    ----------
    reference : ndarray of float
        Ground-truth values, one per sample; the sole source of the
        location and scale parameters.
    prediction : ndarray of float
        Model-output values, one per sample, aligned with `reference`;
        standardized using `reference`'s location and scale.
    sample_ids : ndarray of str
        Sample identifier aligned with `reference`/`prediction`.
    method : {"normal", "robust"}, optional
        ``"normal"`` uses the reference mean and Bessel-corrected SD
        (``ddof=1``); ``"robust"`` uses the reference median and IQR.
        Default is ``"normal"``.
    abs_tol : float, optional
        Absolute near-zero tolerance applied to ``scale**2``. Default
        is ``mrsvalidate._numerics.DEFAULT_ABS_TOL`` (``0.0``).
    rel_tol : float, optional
        Relative near-zero tolerance applied to ``scale**2``, scaled
        by ``max(1, scale**2)``. Default is
        ``mrsvalidate._numerics.DEFAULT_REL_TOL`` (``1e-12``).

    Returns
    -------
    result : StandardizationResult
        ``result.z_reference``/``result.z_prediction`` are the
        standardized arrays; ``result.location``/``result.scale`` are
        the reference-derived parameters actually applied.
        ``result.meta.status`` is ``ResultStatus.UNDEFINED`` (empty
        z-arrays, ``location``/``scale`` still reported) when the
        reference scale is effectively zero, or
        ``ResultStatus.INSUFFICIENT_DATA`` when fewer than two valid
        rows remain.

    Raises
    ------
    ValueError
        If `method` is not one of ``"normal"`` or ``"robust"``.

    Notes
    -----
    ``method="normal"``:

    .. math::

        z = \\frac{v - \\bar{y}}{s_y}, \\qquad
        \\bar{y} = \\text{mean}(\\text{reference}), \\quad
        s_y = \\text{SD}(\\text{reference})

    ``method="robust"``:

    .. math::

        z = \\frac{v - \\operatorname{median}(y)}{\\text{IQR}(y)}

    where :math:`v` is each element of `reference` (giving
    `z_reference`) or `prediction` (giving `z_prediction`), and
    :math:`y` denotes `reference`. No epsilon is added to a
    near-zero scale; instead the result is reported as
    ``ResultStatus.UNDEFINED``.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.standardization import standardize
    >>> ref = np.array([1.0, 2.0, 3.0, 4.0])
    >>> pred = np.array([2.0, 2.0, 4.0, 4.0])
    >>> ids = np.array(["a", "b", "c", "d"])
    >>> r = standardize(ref, pred, ids, method="normal")
    >>> r.location, r.scale
    (2.5, 1.2909944487358056)
    >>> r.z_reference
    array([-1.161895  , -0.38729833,  0.38729833,  1.161895  ])

    A constant reference has zero scale and is reported as undefined
    rather than dividing by zero:

    >>> ref_constant = np.array([5.0, 5.0, 5.0, 5.0])
    >>> r2 = standardize(ref_constant, pred, ids, method="normal")
    >>> r2.meta.status.value
    'undefined'
    >>> r2.z_reference.size
    0
    """

    all_ids, report, (ref, pred) = valid_subset(sample_ids, reference, prediction)
    base_meta = meta_for(
        all_ids=all_ids, report=report, method=f"standardize[{method}]", min_n=2
    )
    if base_meta.status is not ResultStatus.VALID:
        empty = np.asarray([], dtype=np.float64)
        return StandardizationResult(
            meta=base_meta,
            z_reference=empty,
            z_prediction=empty,
            location=float("nan"),
            scale=float("nan"),
            method=method,
        )

    if method == "normal":
        location = float(np.mean(ref))
        scale = float(np.std(ref, ddof=1))
    elif method == "robust":
        location = float(np.median(ref))
        scale = _iqr(ref)
    else:
        raise ValueError(f"Unknown standardization method: {method!r}")

    threshold = max(abs_tol, rel_tol * max(1.0, scale**2))
    if is_near_zero(scale**2, scale, abs_tol=abs_tol, rel_tol=rel_tol):
        meta = build_meta(
            status=ResultStatus.UNDEFINED,
            all_ids=all_ids,
            included_ids=report.included_ids,
            exclusion_reasons=report.exclusion_reasons,
            method=f"standardize[{method}]",
            configuration={
                "method": method,
                "near_zero_threshold": threshold,
                "location": location,
                "scale": scale,
            },
            warnings=("standardization undefined: reference scale is effectively zero",),
        )
        empty = np.asarray([], dtype=np.float64)
        return StandardizationResult(
            meta=meta,
            z_reference=empty,
            z_prediction=empty,
            location=location,
            scale=scale,
            method=method,
        )

    meta = build_meta(
        status=ResultStatus.VALID,
        all_ids=all_ids,
        included_ids=report.included_ids,
        exclusion_reasons=report.exclusion_reasons,
        method=f"standardize[{method}]",
        configuration={
            "method": method,
            "near_zero_threshold": threshold,
            "location": location,
            "scale": scale,
        },
    )
    return StandardizationResult(
        meta=meta,
        z_reference=(ref - location) / scale,
        z_prediction=(pred - location) / scale,
        location=location,
        scale=scale,
        method=method,
    )
