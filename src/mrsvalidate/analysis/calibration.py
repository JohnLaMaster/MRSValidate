r"""Expected calibration error (ECE) for probabilistic predictions.

Assesses whether a predictive distribution's stated coverage matches
its empirical coverage (spec eq. 52 / manuscript eq. ``ECE``).

Notes
-----
.. math::

    a_m = \frac{1}{N}\sum_{n=1}^{N} \mathbb{1}[F_n(y_n) \le c_m]
    \qquad
    \text{ECE} = \sum_{m} w_m \, |a_m - c_m|, \qquad \sum_m w_m = 1

where :math:`c_m` are nominal confidence levels and :math:`a_m` the
corresponding empirical coverage.

This module accepts predictive distributions already reduced to PIT
values (:math:`F_n(y_n)`, the predictive CDF evaluated at the true
value) or other precomputed probability/coverage quantities — it never
trains or fits the uncertainty model itself. Recalibration is a
separate concern from assessing the originally supplied predictive
distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..results import AnalysisResult, ResultStatus, build_meta

__all__ = ["CalibrationResult", "expected_calibration_error"]


@dataclass(frozen=True)
class CalibrationResult(AnalysisResult):
    """Result of :func:`expected_calibration_error`.

    Attributes
    ----------
    meta : AnalysisMeta
        Provenance and inclusion/exclusion record (inherited from
        :class:`~mrsvalidate.results.AnalysisResult`).
    nominal_coverage : ndarray of float
        The nominal confidence levels :math:`c_m` the coverage was
        evaluated at (as supplied by the caller).
    empirical_coverage : ndarray of float
        The empirical coverage :math:`a_m` observed at each nominal
        level, i.e. the fraction of included PIT values at or below
        :math:`c_m`.
    absolute_calibration_gap : ndarray of float
        :math:`|a_m - c_m|` at each nominal level.
    weights : ndarray of float
        The per-level weights :math:`w_m` used to combine the gaps into
        `ece`; sums to 1.
    ece : float
        The expected calibration error, :math:`\\sum_m w_m |a_m - c_m|`.
    """

    nominal_coverage: NDArray[np.float64]  # c_m
    empirical_coverage: NDArray[np.float64]  # a_m
    absolute_calibration_gap: NDArray[np.float64]  # |a_m - c_m|
    weights: NDArray[np.float64]
    ece: float


def expected_calibration_error(
    pit_values: NDArray[np.float64],
    confidence_levels: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
    *,
    weights: NDArray[np.float64] | None = None,
) -> CalibrationResult:
    r"""Compute the expected calibration error at a set of nominal levels.

    Parameters
    ----------
    pit_values : ndarray of float
        :math:`F_n(y_n)`, the predictive CDF evaluated at the true
        value, for each observation. Must lie in ``[0, 1]``.
    confidence_levels : ndarray of float
        The nominal levels :math:`c_m` to evaluate empirical coverage
        at.
    sample_ids : ndarray of str
        Identifier for each observation.
    weights : ndarray of float, optional
        Per-level weights :math:`w_m` used to combine the calibration
        gaps into a single ECE value. Must sum to 1. Defaults to a
        uniform weight ``1 / M`` over the ``M`` confidence levels.

    Returns
    -------
    CalibrationResult
        See :class:`CalibrationResult`. ``meta.status`` is
        :attr:`~mrsvalidate.results.ResultStatus.INVALID_INPUT` (and
        `empirical_coverage`, `absolute_calibration_gap`, `ece` are
        ``nan``) if no PIT value lies in ``[0, 1]`` and is finite.

    Raises
    ------
    ValueError
        If `weights` is supplied and does not sum to 1.

    Notes
    -----
    .. math::

        a_m = \frac{1}{N}\sum_{n=1}^{N} \mathbb{1}[F_n(y_n) \le c_m]
        \qquad
        \text{ECE} = \sum_{m} w_m \, |a_m - c_m|, \qquad \sum_m w_m = 1

    Examples
    --------
    >>> import numpy as np
    >>> pit = np.array([0.1, 0.4, 0.5, 0.6, 0.9])
    >>> levels = np.array([0.25, 0.5, 0.75])
    >>> ids = np.array(["a", "b", "c", "d", "e"])
    >>> result = expected_calibration_error(pit, levels, ids)
    >>> result.empirical_coverage
    array([0.2, 0.6, 0.8])
    >>> round(result.ece, 4)
    0.0667
    """

    all_ids = tuple(np.asarray(sample_ids, dtype=str).tolist())
    pit = np.asarray(pit_values, dtype=np.float64)
    levels = np.asarray(confidence_levels, dtype=np.float64)

    valid = np.isfinite(pit) & (pit >= 0) & (pit <= 1)
    n_invalid = int(np.sum(~valid))
    included_ids = tuple(np.asarray(all_ids)[valid].tolist())
    status = ResultStatus.VALID if included_ids else ResultStatus.INVALID_INPUT

    m = levels.shape[0]
    w = np.asarray(weights, dtype=np.float64) if weights is not None else np.full(m, 1.0 / m)
    if not np.isclose(np.sum(w), 1.0):
        raise ValueError(f"weights must sum to 1, got {np.sum(w)}")

    if status is not ResultStatus.VALID:
        meta = build_meta(
            status=status,
            all_ids=all_ids,
            included_ids=included_ids,
            exclusion_reasons={"pit_out_of_unit_interval_or_nonfinite": n_invalid}
            if n_invalid
            else {},
            method="expected_calibration_error",
        )
        nan_m = np.full(m, np.nan)
        return CalibrationResult(
            meta=meta,
            nominal_coverage=levels,
            empirical_coverage=nan_m,
            absolute_calibration_gap=nan_m,
            weights=w,
            ece=float("nan"),
        )

    pit_valid = pit[valid]
    empirical_coverage = np.array([np.mean(pit_valid <= c) for c in levels])
    gap = np.abs(empirical_coverage - levels)
    ece = float(np.sum(w * gap))

    meta = build_meta(
        status=ResultStatus.VALID,
        all_ids=all_ids,
        included_ids=included_ids,
        exclusion_reasons={"pit_out_of_unit_interval_or_nonfinite": n_invalid} if n_invalid else {},
        method="expected_calibration_error",
        configuration={"n_confidence_levels": m, "binning": "fixed_nominal_levels"},
    )
    return CalibrationResult(
        meta=meta,
        nominal_coverage=levels,
        empirical_coverage=empirical_coverage,
        absolute_calibration_gap=gap,
        weights=w,
        ece=ece,
    )
