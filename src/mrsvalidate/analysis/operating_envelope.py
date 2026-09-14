r"""Robust operating-envelope estimation.

A first-class analysis, not a plotting convenience (spec: sparse-stratum
robust binomial envelope, manuscript operating-envelope section).

Notes
-----
For each predicted-concentration bin and externally supplied
information stratum, a one-sided exact binomial (Clopper-Pearson) lower
confidence bound is computed on the success probability
:math:`P(E_m \le \tau_m)`, with the overall per-bin guarantee taken as
the minimum lower bound across strata (simultaneous coverage via a
per-stratum Bonferroni allocation :math:`\alpha_g = \alpha / G`).

Sparse strata are reported as ``insufficient_data`` and are **never**
silently dropped from the minimum: a bin is only judged "inside the
envelope" once every stratum for that bin has enough data, so a
data-starved stratum can never make the envelope look artificially
large by being ignored.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import beta as _beta

from ..results import AnalysisResult, ResultStatus, build_meta

__all__ = ["OperatingEnvelopeResult", "operating_envelope"]


@dataclass(frozen=True)
class OperatingEnvelopeResult(AnalysisResult):
    """Result of :func:`operating_envelope`.

    Attributes
    ----------
    meta : AnalysisMeta
        Provenance and inclusion/exclusion record (inherited from
        :class:`~mrsvalidate.results.AnalysisResult`).
    bin_edges : ndarray of float
        The predicted-concentration bin edges, as supplied by the
        caller.
    stratum_edges : ndarray of float
        The information-stratum edges, as supplied by the caller.
    successes : ndarray of int, shape (n_bins, n_strata)
        Count of successful observations (``success`` is ``True``) in
        each bin/stratum cell.
    totals : ndarray of int, shape (n_bins, n_strata)
        Count of all in-range observations in each bin/stratum cell.
    raw_success_rate : ndarray of float, shape (n_bins, n_strata)
        The unadjusted empirical success rate ``successes / totals`` in
        each cell; ``nan`` where `totals` is 0.
    lower_confidence_bound : ndarray of float, shape (n_bins, n_strata)
        The one-sided Clopper-Pearson lower confidence bound on the
        success probability in each cell, at level `alpha_per_stratum`.
    robust_lower_bound : ndarray of float, shape (n_bins,)
        The minimum of `lower_confidence_bound` across strata, for each
        bin where `sufficient_data_mask` is ``True``; ``nan`` elsewhere.
    target_reliability : float
        The caller-specified minimum acceptable success probability, as
        supplied to :func:`operating_envelope`.
    alpha : float
        The overall (pre-Bonferroni-split) one-sided significance
        level, as supplied to :func:`operating_envelope`.
    alpha_per_stratum : float
        ``alpha / n_strata``, the per-stratum significance level after
        the Bonferroni allocation.
    sufficient_data_mask : ndarray of bool, shape (n_bins,)
        ``True`` for a bin iff every stratum in that bin has at least
        `min_stratum_n` observations.
    operating_bin_mask : ndarray of bool, shape (n_bins,)
        ``True`` for a bin iff `sufficient_data_mask` is ``True`` *and*
        `robust_lower_bound` meets or exceeds `target_reliability`.
    """

    bin_edges: NDArray[np.float64]
    stratum_edges: NDArray[np.float64]
    successes: NDArray[np.int64]  # (n_bins, n_strata)
    totals: NDArray[np.int64]  # (n_bins, n_strata)
    raw_success_rate: NDArray[np.float64]  # (n_bins, n_strata)
    lower_confidence_bound: NDArray[np.float64]  # (n_bins, n_strata)
    robust_lower_bound: NDArray[np.float64]  # (n_bins,)
    target_reliability: float
    alpha: float
    alpha_per_stratum: float
    sufficient_data_mask: NDArray[np.bool_]  # (n_bins,) — True iff every stratum has enough data
    operating_bin_mask: NDArray[np.bool_]  # (n_bins,)


def _clopper_pearson_lower(successes: NDArray[np.int64], totals: NDArray[np.int64], alpha: float) -> NDArray[np.float64]:
    r"""Compute the one-sided Clopper-Pearson lower confidence bound.

    Parameters
    ----------
    successes : ndarray of int
        Number of successes in each cell.
    totals : ndarray of int
        Number of trials in each cell, aligned with `successes`.
    alpha : float
        One-sided significance level.

    Returns
    -------
    ndarray of float
        The lower confidence bound for each cell. ``0`` where
        `successes` is 0 (the Beta distribution's first shape parameter
        would otherwise be invalid); ``nan`` where `totals` is 0.

    Notes
    -----
    .. math::

        L = \operatorname{Beta}^{-1}(\alpha;\, S,\, N - S + 1)

    with :math:`L = 0` when :math:`S = 0`.

    Examples
    --------
    >>> import numpy as np
    >>> successes = np.array([0, 5, 10])
    >>> totals = np.array([10, 10, 10])
    >>> np.round(_clopper_pearson_lower(successes, totals, 0.05), 4)
    array([0.    , 0.2224, 0.7411])
    """

    s = successes.astype(np.float64)
    n = totals.astype(np.float64)
    with np.errstate(invalid="ignore"):
        lower = _beta.ppf(alpha, s, n - s + 1.0)
    lower = np.where(successes == 0, 0.0, lower)
    lower = np.where(totals == 0, np.nan, lower)
    return lower


def operating_envelope(
    predicted_concentration: NDArray[np.float64],
    information_metric: NDArray[np.float64],
    success: NDArray[np.bool_],
    sample_ids: NDArray[np.str_],
    *,
    bin_edges: NDArray[np.float64],
    stratum_edges: NDArray[np.float64],
    target_reliability: float,
    alpha: float = 0.05,
    min_stratum_n: int = 1,
) -> OperatingEnvelopeResult:
    r"""Estimate the robust operating envelope over concentration bins and information strata.

    For each predicted-concentration bin, computes a simultaneous
    (across information strata) lower confidence bound on the
    probability of "success" and reports which bins meet the target
    reliability with sufficient supporting data in every stratum.

    Parameters
    ----------
    predicted_concentration : ndarray of float
        Predicted concentration for each observation, used to assign it
        to a bin via `bin_edges`.
    information_metric : ndarray of float
        An externally supplied information/quality metric for each
        observation, used to assign it to a stratum via `stratum_edges`.
    success : ndarray of bool
        Caller-computed success indicator :math:`E_m \le \tau_m` for
        each observation. This module does not recompute the error
        metric or tolerance — those come from
        :mod:`mrsvalidate.analysis.errors` or other caller-defined
        criteria.
    sample_ids : ndarray of str
        Identifier for each observation.
    bin_edges : ndarray of float
        Edges defining the predicted-concentration bins (``n_bins + 1``
        edges for ``n_bins`` bins), passed to :func:`numpy.digitize`.
    stratum_edges : ndarray of float
        Edges defining the information strata (``n_strata + 1`` edges
        for ``n_strata`` strata), passed to :func:`numpy.digitize`.
    target_reliability : float
        The minimum acceptable success probability a bin's robust lower
        bound must reach to be judged "inside the envelope".
    alpha : float, default 0.05
        Overall one-sided significance level for the confidence bound,
        before the per-stratum Bonferroni split.
    min_stratum_n : int, default 1
        Minimum number of observations a bin/stratum cell must have to
        be considered to have sufficient data.

    Returns
    -------
    OperatingEnvelopeResult
        See :class:`OperatingEnvelopeResult`.

    Notes
    -----
    The overall per-bin guarantee is the minimum Clopper-Pearson lower
    bound across strata, at the Bonferroni-corrected level
    :math:`\alpha_g = \alpha / G` (``G = n_strata``), which yields
    simultaneous coverage across the ``G`` strata within a bin.

    Sparse strata (fewer than `min_stratum_n` observations) are marked
    ``insufficient_data`` via `sufficient_data_mask` and are **never**
    silently dropped from the minimum: a bin is judged "inside the
    envelope" only once every stratum for that bin has enough data, so
    a data-starved stratum can never make the envelope look artificially
    large by being excluded from the minimum computation.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> conc = rng.uniform(0, 10, n)
    >>> info_metric = rng.uniform(0, 1, n)
    >>> success = rng.random(n) < 0.9
    >>> ids = np.array([f"s{i}" for i in range(n)])
    >>> bin_edges = np.array([0.0, 5.0, 10.0])
    >>> stratum_edges = np.array([0.0, 0.5, 1.0])
    >>> result = operating_envelope(
    ...     conc, info_metric, success, ids,
    ...     bin_edges=bin_edges, stratum_edges=stratum_edges,
    ...     target_reliability=0.8, alpha=0.05, min_stratum_n=5,
    ... )
    >>> result.sufficient_data_mask
    array([ True,  True])
    >>> result.operating_bin_mask
    array([ True, False])
    >>> round(result.alpha_per_stratum, 4)
    0.025
    """

    all_ids = tuple(np.asarray(sample_ids, dtype=str).tolist())
    conc = np.asarray(predicted_concentration, dtype=np.float64)
    info_metric = np.asarray(information_metric, dtype=np.float64)
    succ = np.asarray(success, dtype=bool)
    bin_edges = np.asarray(bin_edges, dtype=np.float64)
    stratum_edges = np.asarray(stratum_edges, dtype=np.float64)

    n_bins = bin_edges.shape[0] - 1
    n_strata = stratum_edges.shape[0] - 1

    finite = np.isfinite(conc) & np.isfinite(info_metric)
    bin_idx = np.digitize(conc, bin_edges) - 1
    stratum_idx = np.digitize(info_metric, stratum_edges) - 1
    in_range = finite & (bin_idx >= 0) & (bin_idx < n_bins) & (stratum_idx >= 0) & (stratum_idx < n_strata)

    included_ids = tuple(np.asarray(all_ids)[in_range].tolist())
    n_excluded = int(np.sum(~in_range))

    successes = np.zeros((n_bins, n_strata), dtype=np.int64)
    totals = np.zeros((n_bins, n_strata), dtype=np.int64)
    for b, g, s in zip(bin_idx[in_range], stratum_idx[in_range], succ[in_range]):
        totals[b, g] += 1
        if s:
            successes[b, g] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        raw_success_rate = np.where(totals > 0, successes / np.maximum(totals, 1), np.nan)

    alpha_per_stratum = alpha / n_strata
    lower_confidence_bound = _clopper_pearson_lower(successes, totals, alpha_per_stratum)

    sufficient_per_cell = totals >= min_stratum_n
    sufficient_data_mask = np.all(sufficient_per_cell, axis=1)

    robust_lower_bound = np.full(n_bins, np.nan, dtype=np.float64)
    for b in range(n_bins):
        if sufficient_data_mask[b]:
            robust_lower_bound[b] = np.min(lower_confidence_bound[b, :])

    operating_bin_mask = sufficient_data_mask & (robust_lower_bound >= target_reliability)

    status = ResultStatus.VALID if included_ids else ResultStatus.INSUFFICIENT_DATA
    meta = build_meta(
        status=status,
        all_ids=all_ids,
        included_ids=included_ids,
        exclusion_reasons={"out_of_bin_or_stratum_range_or_nonfinite": n_excluded}
        if n_excluded
        else {},
        method="operating_envelope",
        configuration={
            "n_bins": n_bins,
            "n_strata": n_strata,
            "min_stratum_n": min_stratum_n,
            "confidence_method": "clopper_pearson",
        },
    )
    return OperatingEnvelopeResult(
        meta=meta,
        bin_edges=bin_edges,
        stratum_edges=stratum_edges,
        successes=successes,
        totals=totals,
        raw_success_rate=raw_success_rate,
        lower_confidence_bound=lower_confidence_bound,
        robust_lower_bound=robust_lower_bound,
        target_reliability=target_reliability,
        alpha=alpha,
        alpha_per_stratum=alpha_per_stratum,
        sufficient_data_mask=sufficient_data_mask,
        operating_bin_mask=operating_bin_mask,
    )
