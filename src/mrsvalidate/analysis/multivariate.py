"""Multivariate concordance correlation coefficient (Choi et al. 2025).

Implements the multivariate generalization of the concordance
correlation coefficient (spec eqs. 50-51 / manuscript eqs.
``multivariate rho_c`` / ``cccm_estimator``), which jointly assesses
agreement of ``d`` methods across ``p`` variables per subject.

Notes
-----
Treated as a separate estimator family from the scalar Lin CCC
(:mod:`mrsvalidate.analysis.agreement`), not a reinterpretation of it:
the two are not numerically comparable and neither is derived from the
other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..results import AnalysisResult, ResultStatus, build_meta

__all__ = ["multivariate_ccc", "MultivariateResult"]


@dataclass(frozen=True)
class MultivariateResult(AnalysisResult):
    """Result of the multivariate concordance correlation coefficient.

    Returned by :func:`multivariate_ccc`.

    Attributes
    ----------
    meta : AnalysisMeta
        Provenance/inclusion record describing which subjects were
        used and why any were excluded.
    rho_c_multivariate : float
        The multivariate concordance correlation coefficient,
        :math:`\\rho_C^{(M)}`. ``nan`` when `meta.status` is not
        ``ResultStatus.VALID`` (fewer than two valid subjects, fewer
        than two methods, or a zero denominator).
    n_methods : int
        Number of methods being compared, ``d`` -- the size of the
        second axis of the `measurements` array passed to
        :func:`multivariate_ccc`.
    n_variables : int
        Number of variables measured per method, ``p`` -- the size of
        the third axis of the `measurements` array passed to
        :func:`multivariate_ccc`.
    """

    rho_c_multivariate: float
    n_methods: int
    n_variables: int


def multivariate_ccc(
    measurements: NDArray[np.float64], sample_ids: NDArray[np.str_]
) -> MultivariateResult:
    """Multivariate concordance correlation coefficient across methods.

    Generalizes Lin's concordance correlation coefficient to the case
    where ``d >= 2`` methods each report ``p >= 1`` variables per
    subject, using the method-of-moments estimator of Choi et al.
    (2025) (manuscript eq. ``cccm_estimator``).

    Parameters
    ----------
    measurements : ndarray of float
        Array of shape ``(n_subjects, d_methods, p_variables)``. Each
        ``measurements[i, k, :]`` is the vector of ``p`` variables
        reported by method ``k`` for subject ``i``.
    sample_ids : ndarray of str
        Subject identifiers aligned with the first axis of
        `measurements`, used to build the inclusion/exclusion
        provenance record.

    Returns
    -------
    MultivariateResult
        `rho_c_multivariate` is :math:`\\rho_C^{(M)}`; `n_methods` and
        `n_variables` echo the ``d`` and ``p`` dimensions of
        `measurements`. `rho_c_multivariate` is ``nan`` when
        `meta.status` is not ``VALID`` (fewer than two valid subjects
        after dropping non-finite rows, fewer than two methods, or a
        zero denominator).

    Raises
    ------
    ValueError
        If `measurements` does not have exactly 3 dimensions.

    Notes
    -----
    A subject is dropped (marked invalid, counted under the
    ``"nan_or_inf"`` exclusion reason) if any entry across its
    ``(d_methods, p_variables)`` slice is non-finite; the estimator is
    then computed only from the fully-finite subjects.

    With :math:`M = \\frac{1}{n}\\sum_i Y_i` (shape :math:`d \\times
    p`) and :math:`M_2 = \\frac{1}{n}\\sum_i Y_i Y_i^\\top` (shape
    :math:`d \\times d`), where :math:`Y_i` is the :math:`d \\times p`
    measurement matrix for subject :math:`i`:

    .. math::

        \\rho_C^{(M)} = \\frac{\\mathbf{1}^\\top M_2 \\mathbf{1} -
        \\operatorname{tr}(M_2) + \\operatorname{tr}(M M^\\top) -
        \\mathbf{1}^\\top (M M^\\top) \\mathbf{1}}
        {(d-1)\\operatorname{tr}(M_2) + \\operatorname{tr}(M M^\\top) -
        \\mathbf{1}^\\top (M M^\\top) \\mathbf{1}}

    Examples
    --------
    Two methods that agree exactly on every variable give
    :math:`\\rho_C^{(M)} = 1`:

    >>> import numpy as np
    >>> from mrsvalidate.analysis.multivariate import multivariate_ccc
    >>> y_perfect = np.array([
    ...     [[1.0, 2.0], [1.0, 2.0]],
    ...     [[2.0, 1.0], [2.0, 1.0]],
    ...     [[3.0, 3.0], [3.0, 3.0]],
    ...     [[4.0, 0.0], [4.0, 0.0]],
    ... ])
    >>> ids = np.array(["s1", "s2", "s3", "s4"])
    >>> result = multivariate_ccc(y_perfect, ids)
    >>> round(result.rho_c_multivariate, 4)
    1.0
    >>> result.n_methods, result.n_variables
    (2, 2)

    A second method that is consistently offset from the first is
    less concordant even though the two still covary:

    >>> y_biased = np.array([
    ...     [[1.0, 2.0], [2.0, 3.0]],
    ...     [[2.0, 1.0], [3.0, 2.0]],
    ...     [[3.0, 3.0], [4.0, 4.0]],
    ...     [[4.0, 0.0], [5.0, 1.0]],
    ... ])
    >>> biased = multivariate_ccc(y_biased, ids)
    >>> round(biased.rho_c_multivariate, 4)
    0.7143
    """

    all_ids = tuple(np.asarray(sample_ids, dtype=str).tolist())
    y = np.asarray(measurements, dtype=np.float64)
    if y.ndim != 3:
        raise ValueError(
            f"measurements must have shape (n_subjects, d_methods, p_variables), got {y.shape}"
        )
    n, d, p = y.shape

    finite = np.all(np.isfinite(y.reshape(n, -1)), axis=1)
    included_ids = tuple(np.asarray(all_ids)[finite].tolist())
    n_invalid = n - len(included_ids)
    status = ResultStatus.VALID if len(included_ids) >= 2 and d >= 2 else ResultStatus.INSUFFICIENT_DATA

    if status is not ResultStatus.VALID:
        meta = build_meta(
            status=status,
            all_ids=all_ids,
            included_ids=included_ids,
            exclusion_reasons={"nan_or_inf": n_invalid} if n_invalid else {},
            method="multivariate_ccc",
        )
        return MultivariateResult(meta=meta, rho_c_multivariate=float("nan"), n_methods=d, n_variables=p)

    y_valid = y[finite]
    m_hat = np.mean(y_valid, axis=0)  # (d, p)
    m2_hat = np.mean(np.einsum("nip,njp->nij", y_valid, y_valid), axis=0)  # (d, d)

    ones = np.ones(d)
    mmT = m_hat @ m_hat.T
    tr_m2 = np.trace(m2_hat)
    tr_mmT = np.trace(mmT)
    ones_m2_ones = float(ones @ m2_hat @ ones)
    ones_mmT_ones = float(ones @ mmT @ ones)

    numerator = ones_m2_ones - tr_m2 + tr_mmT - ones_mmT_ones
    denominator = (d - 1) * tr_m2 + tr_mmT - ones_mmT_ones

    if denominator == 0:
        meta = build_meta(
            status=ResultStatus.UNDEFINED,
            all_ids=all_ids,
            included_ids=included_ids,
            exclusion_reasons={"nan_or_inf": n_invalid} if n_invalid else {},
            method="multivariate_ccc",
            warnings=("multivariate CCC undefined: zero denominator",),
        )
        return MultivariateResult(meta=meta, rho_c_multivariate=float("nan"), n_methods=d, n_variables=p)

    meta = build_meta(
        status=ResultStatus.VALID,
        all_ids=all_ids,
        included_ids=included_ids,
        exclusion_reasons={"nan_or_inf": n_invalid} if n_invalid else {},
        method="multivariate_ccc",
        configuration={"n_methods": d, "n_variables": p},
    )
    return MultivariateResult(
        meta=meta, rho_c_multivariate=float(numerator / denominator), n_methods=d, n_variables=p
    )
