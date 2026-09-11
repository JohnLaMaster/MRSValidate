r"""Error consistency between two predictors.

Quantifies how similarly two predictors err on the same observations
(spec eq. 44 / manuscript ``eqn:error consistency``).

Notes
-----
.. math::

    \rho_{\text{EC}} = \frac{1}{N}\sum_{n=1}^{N}
        \operatorname{sign}(r_{i,n} \, r_{j,n}) \,
        \frac{\big|\,|r_{i,n}| - |r_{j,n}|\,\big|}{|r_{i,n}| + |r_{j,n}|}

where :math:`r_{i,n}` and :math:`r_{j,n}` are the residuals of
predictors *i* and *j* on observation *n*. A single aggregate near zero
is not sufficient to distinguish genuinely similar errors from
positive/negative cancellation, so the per-observation contributions
are always returned alongside the aggregate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..results import AnalysisResult, ResultStatus, build_meta

__all__ = ["error_consistency", "ErrorConsistencyResult"]


@dataclass(frozen=True)
class ErrorConsistencyResult(AnalysisResult):
    """Result of :func:`error_consistency`.

    Attributes
    ----------
    meta : AnalysisMeta
        Provenance and inclusion/exclusion record (inherited from
        :class:`~mrsvalidate.results.AnalysisResult`).
    rho_ec : float
        The aggregate error consistency, i.e. the mean of
        `contributions` over included observations. ``nan`` if no
        observation had finite residuals for both predictors.
    contributions : ndarray of float
        Per-observation error consistency contribution, aligned with
        ``meta.included_ids``. Rows where both residuals are exactly
        zero contribute exactly ``0`` by definition.
    zero_zero_pairs : int
        Number of included observations where both residuals are
        exactly zero (a degenerate case of the ``0/0`` contribution
        rule, tracked separately from the other exclusion reasons).
    """

    rho_ec: float
    contributions: NDArray[np.float64]  # aligned with meta.included_ids
    zero_zero_pairs: int


def error_consistency(
    residual_i: NDArray[np.float64],
    residual_j: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
) -> ErrorConsistencyResult:
    r"""Compute the error consistency between two predictors' residuals.

    For each observation, compares the sign and relative magnitude of
    two predictors' residuals: a contribution near ``+1`` means both
    predictors erred in the same direction with similar magnitude, a
    contribution near ``-1`` means they erred in opposite directions
    with similar magnitude, and values near ``0`` indicate dissimilar
    magnitudes regardless of sign agreement.

    Parameters
    ----------
    residual_i : ndarray of float
        Residuals (``prediction - reference`` or ``reference -
        prediction``, as long as consistent with `residual_j`) for
        predictor *i*.
    residual_j : ndarray of float
        Residuals for predictor *j*, aligned observation-for-observation
        with `residual_i`.
    sample_ids : ndarray of str
        Identifier for each observation.

    Returns
    -------
    ErrorConsistencyResult
        See :class:`ErrorConsistencyResult`. ``meta.status`` is
        :attr:`~mrsvalidate.results.ResultStatus.INSUFFICIENT_DATA` (and
        `rho_ec` is ``nan``, `contributions` empty) if no observation
        has finite residuals for both predictors.

    Notes
    -----
    .. math::

        \rho_{\text{EC}} = \frac{1}{N}\sum_{n=1}^{N}
            \operatorname{sign}(r_{i,n} \, r_{j,n}) \,
            \frac{\big|\,|r_{i,n}| - |r_{j,n}|\,\big|}{|r_{i,n}| + |r_{j,n}|}

    An observation where both residuals are exactly zero would divide
    ``0 / 0``; it is defined to contribute exactly ``0`` rather than
    being excluded, and the number of such observations is reported in
    `zero_zero_pairs`.

    A single aggregate `rho_ec` near zero is not sufficient to
    distinguish genuinely similar errors from positive/negative
    cancellation across observations, so `contributions` is always
    returned alongside the aggregate.

    Examples
    --------
    >>> import numpy as np
    >>> residual_i = np.array([3.0, -1.0, 0.0])
    >>> residual_j = np.array([1.0, 1.0, 0.0])
    >>> ids = np.array(["a", "b", "c"])
    >>> result = error_consistency(residual_i, residual_j, ids)
    >>> round(result.rho_ec, 4)
    0.1667
    >>> result.zero_zero_pairs
    1
    """

    all_ids = tuple(np.asarray(sample_ids, dtype=str).tolist())
    r_i = np.asarray(residual_i, dtype=np.float64)
    r_j = np.asarray(residual_j, dtype=np.float64)

    finite = np.isfinite(r_i) & np.isfinite(r_j)
    n_invalid = int(np.sum(~finite))
    included_ids = tuple(np.asarray(all_ids)[finite].tolist())
    status = ResultStatus.VALID if included_ids else ResultStatus.INSUFFICIENT_DATA

    if status is not ResultStatus.VALID:
        meta = build_meta(
            status=status,
            all_ids=all_ids,
            included_ids=included_ids,
            exclusion_reasons={"nan_or_inf": n_invalid} if n_invalid else {},
            method="error_consistency",
        )
        return ErrorConsistencyResult(
            meta=meta,
            rho_ec=float("nan"),
            contributions=np.asarray([], dtype=np.float64),
            zero_zero_pairs=0,
        )

    ri, rj = r_i[finite], r_j[finite]
    both_zero = (ri == 0) & (rj == 0)
    denom = np.abs(ri) + np.abs(rj)

    contributions = np.zeros_like(ri)
    nonzero = ~both_zero
    contributions[nonzero] = (
        np.sign(ri[nonzero] * rj[nonzero])
        * np.abs(np.abs(ri[nonzero]) - np.abs(rj[nonzero]))
        / denom[nonzero]
    )
    # both_zero rows contribute exactly 0 by definition (already the array default).

    meta = build_meta(
        status=ResultStatus.VALID,
        all_ids=all_ids,
        included_ids=included_ids,
        exclusion_reasons={"nan_or_inf": n_invalid} if n_invalid else {},
        method="error_consistency",
        configuration={"zero_zero_pairs": int(np.sum(both_zero))},
    )
    return ErrorConsistencyResult(
        meta=meta,
        rho_ec=float(np.mean(contributions)),
        contributions=contributions,
        zero_zero_pairs=int(np.sum(both_zero)),
    )
