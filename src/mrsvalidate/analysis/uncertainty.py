r"""Uncertainty-quality assessment via cobeau (correlation of error and uncertainty).

Evaluates and reports the relationship between predicted uncertainty
and observed error (spec eqs. 54-55 / manuscript ``cobeau`` / Rost
2019); it does not construct or fit the uncertainty estimates
themselves — a predictive marginal variance, decomposed uncertainty,
etc. must be generated outside this package.

Notes
-----
cobeau is Pearson's *r* (or, robustly, the median-centered correlation)
between a nonnegative error metric and the corresponding predicted
uncertainty. It is reused directly from
:mod:`mrsvalidate.analysis.regression` rather than reimplemented, since
it is mathematically identical to those primitives applied to a
different pair of variables.

Per-metabolite cobeau values are combined into a macro summary via the
manuscript's Fisher-*z*-weighted average (weights :math:`N_m - 3`), not
a naive arithmetic mean:

.. math::

    \bar{z} = \frac{\sum_m w_m \operatorname{arctanh}(\text{cobeau}_m)}{\sum_m w_m},
    \qquad w_m = N_m - 3,
    \qquad \text{cobeau}_{\text{Fisher}} = \tanh(\bar{z})
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..results import AnalysisResult, ResultStatus, ScalarMetricResult, build_meta
from .regression import CorrelationResult, median_centered_correlation, pearson_r

__all__ = ["CobeauResult", "cobeau", "cobeau_fisher_aggregate"]

CobeauMethod = Literal["pearson", "median_centered"]


@dataclass(frozen=True)
class CobeauResult(AnalysisResult):
    """Result of :func:`cobeau`.

    Attributes
    ----------
    meta : AnalysisMeta
        Provenance and inclusion/exclusion record (inherited from
        :class:`~mrsvalidate.results.AnalysisResult`).
    value : float
        The cobeau correlation coefficient between the error metric and
        the predicted uncertainty, in ``[-1, 1]``.
    p_value : float or None
        Two-sided p-value for the correlation, if defined for the
        chosen `method`.
    method : {"pearson", "median_centered"}
        Which underlying correlation estimator computed `value`.
    """

    value: float
    p_value: float | None
    method: CobeauMethod


def cobeau(
    error_metric: NDArray[np.float64],
    uncertainty: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
    *,
    method: CobeauMethod = "pearson",
) -> CobeauResult:
    """Compute the cobeau correlation for one metabolite/output.

    Correlates a nonnegative error metric against predicted uncertainty
    across observations, for a single metabolite/output.

    Parameters
    ----------
    error_metric : ndarray of float
        A nonnegative per-observation error metric (e.g. absolute
        error).
    uncertainty : ndarray of float
        The corresponding per-observation predicted uncertainty,
        aligned observation-for-observation with `error_metric`.
    sample_ids : ndarray of str
        Identifier for each observation.
    method : {"pearson", "median_centered"}, default "pearson"
        Which correlation estimator to use: ordinary Pearson *r*
        (:func:`~mrsvalidate.analysis.regression.pearson_r`) or the
        robust median-centered correlation
        (:func:`~mrsvalidate.analysis.regression.median_centered_correlation`).

    Returns
    -------
    CobeauResult
        See :class:`CobeauResult`. Status and exclusions are inherited
        directly from the underlying
        :class:`~mrsvalidate.analysis.regression.CorrelationResult`.

    Examples
    --------
    >>> import numpy as np
    >>> error_metric = np.array([0.5, 1.0, 1.5, 2.0])
    >>> uncertainty = np.array([0.4, 0.9, 1.6, 2.1])
    >>> ids = np.array(["a", "b", "c", "d"])
    >>> result = cobeau(error_metric, uncertainty, ids)
    >>> round(result.value, 4)
    0.9976
    >>> result.method
    'pearson'
    """

    inner: CorrelationResult = (
        pearson_r(error_metric, uncertainty, sample_ids)
        if method == "pearson"
        else median_centered_correlation(error_metric, uncertainty, sample_ids)
    )
    meta = build_meta(
        status=inner.meta.status,
        all_ids=inner.meta.included_ids + inner.meta.excluded_ids,
        included_ids=inner.meta.included_ids,
        exclusion_reasons=inner.meta.exclusion_reasons,
        method=f"cobeau[{method}]",
        configuration={**inner.meta.configuration, "cobeau_method": method},
        warnings=inner.meta.warnings,
    )
    return CobeauResult(meta=meta, value=inner.r, p_value=inner.p_value, method=method)


def cobeau_fisher_aggregate(
    cobeau_values: NDArray[np.float64],
    n_per_metabolite: NDArray[np.int64],
    metabolite_ids: NDArray[np.str_],
) -> ScalarMetricResult:
    r"""Combine per-metabolite cobeau values into a Fisher-z-weighted macro aggregate.

    Parameters
    ----------
    cobeau_values : ndarray of float
        Per-metabolite cobeau correlation coefficients, as produced by
        :func:`cobeau`.
    n_per_metabolite : ndarray of int
        Number of observations :math:`N_m` underlying each entry of
        `cobeau_values`, aligned index-for-index.
    metabolite_ids : ndarray of str
        Identifier for each metabolite/output.

    Returns
    -------
    ScalarMetricResult
        ``value`` is the Fisher-z-weighted aggregate cobeau,
        :math:`\tanh(\bar{z})`. ``nan`` (with
        ``meta.status`` :attr:`~mrsvalidate.results.ResultStatus.INSUFFICIENT_DATA`)
        if no metabolite qualifies.

    Notes
    -----
    .. math::

        \bar{z} = \frac{\sum_m w_m \operatorname{arctanh}(\text{cobeau}_m)}{\sum_m w_m},
        \qquad w_m = N_m - 3,
        \qquad \text{cobeau}_{\text{Fisher}} = \tanh(\bar{z})

    Metabolites with :math:`|\text{cobeau}_m| \ge 1` (arctanh undefined)
    or :math:`N_m \le 3` (nonpositive weight) are excluded from the
    aggregate and reported under the
    ``perfect_correlation_or_n_le_3`` exclusion reason.

    Examples
    --------
    >>> import numpy as np
    >>> cobeau_values = np.array([0.9, 0.95])
    >>> n_per_metabolite = np.array([10, 20])
    >>> ids = np.array(["m1", "m2"])
    >>> result = cobeau_fisher_aggregate(cobeau_values, n_per_metabolite, ids)
    >>> round(result.value, 4)
    0.9387
    """

    all_ids = tuple(np.asarray(metabolite_ids, dtype=str).tolist())
    values = np.asarray(cobeau_values, dtype=np.float64)
    n = np.asarray(n_per_metabolite, dtype=np.float64)

    valid = np.isfinite(values) & (np.abs(values) < 1.0) & (n > 3)
    n_invalid = int(np.sum(~valid))
    included_ids = tuple(np.asarray(all_ids)[valid].tolist())
    status = ResultStatus.VALID if included_ids else ResultStatus.INSUFFICIENT_DATA

    if status is not ResultStatus.VALID:
        meta = build_meta(
            status=status,
            all_ids=all_ids,
            included_ids=included_ids,
            exclusion_reasons={"perfect_correlation_or_n_le_3": n_invalid} if n_invalid else {},
            method="cobeau_fisher_aggregate",
        )
        return ScalarMetricResult(meta=meta, value=float("nan"))

    z = np.arctanh(values[valid])
    w = n[valid] - 3.0
    zbar = float(np.sum(w * z) / np.sum(w))
    meta = build_meta(
        status=ResultStatus.VALID,
        all_ids=all_ids,
        included_ids=included_ids,
        exclusion_reasons={"perfect_correlation_or_n_le_3": n_invalid} if n_invalid else {},
        method="cobeau_fisher_aggregate",
        configuration={"weighting": "N_m - 3 (Fisher z)"},
    )
    return ScalarMetricResult(meta=meta, value=float(np.tanh(zbar)))
