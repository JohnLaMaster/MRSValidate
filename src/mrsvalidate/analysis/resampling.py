"""Bootstrap and permutation inference.

This module implements the manuscript's resampling-based inference
procedures (spec eqs. 11-15):

- :func:`bootstrap_ci` wraps ``scipy.stats.bootstrap`` (paired vs.
  independent resampling of two or more samples).
- :func:`cluster_bootstrap_ci` implements clustered bootstrap directly
  (SciPy has no built-in clustered bootstrap): independent
  clusters/subjects are resampled with replacement, and every dependent
  observation in a chosen cluster is carried along together.
- :func:`permutation_test` wraps ``scipy.stats.permutation_test``,
  selecting ``permutation_type`` from an explicit ``design`` argument
  (independent / paired sample-swapping / paired-association), matching
  SciPy's own three-way distinction rather than guessing from shape.

Notes
-----
Wrappers select the resampling mode from the analysis design, not from
array shape, and always record the seed, resample count, and exact
method so results are reproducible (spec: "seed/configuration is
serialized", "return Monte Carlo/resample count and the exact inference
method").
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats as _stats

from ..results import AnalysisResult, ResultStatus, build_meta

__all__ = [
    "BootstrapResult",
    "PermutationResult",
    "bootstrap_ci",
    "cluster_bootstrap_ci",
    "permutation_test",
]

BootstrapDesign = Literal["paired", "independent"]
PermutationDesign = Literal["independent", "paired_samples", "paired_association"]

_PERMUTATION_TYPE_BY_DESIGN: dict[PermutationDesign, str] = {
    "independent": "independent",
    "paired_samples": "samples",
    "paired_association": "pairings",
}


@dataclass(frozen=True)
class BootstrapResult(AnalysisResult):
    """Result of a bootstrap confidence interval computation.

    Returned by :func:`bootstrap_ci` and :func:`cluster_bootstrap_ci`.

    Attributes
    ----------
    meta : AnalysisMeta
        Provenance and inclusion/exclusion record (status, sample
        counts, resolved configuration, random seed, warnings).
    point_estimate : float
        The statistic evaluated once on the full (unresampled) data.
    ci_low : float
        Lower bound of the confidence interval.
    ci_high : float
        Upper bound of the confidence interval.
    confidence_level : float
        Nominal confidence level used to build the interval (e.g.
        ``0.95`` for a 95% interval).
    n_resamples : int
        Number of bootstrap resamples drawn.
    seed : int or None
        Random seed used to seed the resampling generator, recorded so
        the interval is exactly reproducible.
    method : str
        Name of the interval-construction method actually used (e.g.
        ``"BCa"``, ``"percentile"``, ``"cluster_bootstrap_percentile"``).
    design : str
        Resampling design used: ``"paired"``, ``"independent"``, or
        ``"cluster"``.
    """

    point_estimate: float
    ci_low: float
    ci_high: float
    confidence_level: float
    n_resamples: int
    seed: int | None
    method: str
    design: str


@dataclass(frozen=True)
class PermutationResult(AnalysisResult):
    """Result of a permutation test.

    Returned by :func:`permutation_test`.

    Attributes
    ----------
    meta : AnalysisMeta
        Provenance and inclusion/exclusion record (status, sample
        counts, resolved configuration, random seed, warnings).
    observed_statistic : float
        The test statistic evaluated once on the observed (unpermuted)
        data.
    p_value : float
        Permutation p-value: the proportion of resampled statistics at
        least as extreme as ``observed_statistic`` (direction set by
        ``alternative``), including the observed arrangement itself.
    n_resamples : int
        Number of random permutations drawn to build the null
        distribution.
    seed : int or None
        Random seed used to seed the permutation generator, recorded
        so the p-value is exactly reproducible.
    design : {"independent", "paired_samples", "paired_association"}
        Permutation design that determined how observations were
        reshuffled under the null hypothesis.
    alternative : str
        Alternative hypothesis used: ``"two-sided"``, ``"less"``, or
        ``"greater"``.
    """

    observed_statistic: float
    p_value: float
    n_resamples: int
    seed: int | None
    design: PermutationDesign
    alternative: str


def bootstrap_ci(
    statistic: Callable[..., float],
    *samples: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
    design: BootstrapDesign = "paired",
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int | None = None,
    method: str = "BCa",
) -> BootstrapResult:
    """Bootstrap confidence interval for a statistic of one or more samples.

    Wraps ``scipy.stats.bootstrap`` and reports the point estimate, the
    resampled confidence interval, and the exact configuration used, so
    the interval can be reproduced later from the recorded seed.

    Parameters
    ----------
    statistic : callable
        Function ``statistic(*samples) -> float`` computing the
        quantity of interest from one array per sample (e.g. a
        difference of means, a correlation coefficient).
    *samples : ndarray of float
        Two or more equal-length 1-D arrays, one per sample, in the
        same row order as `sample_ids`.
    sample_ids : ndarray of str
        Identifier for each row, used only for provenance
        (``meta.included_ids``/``meta.excluded_ids``), not for
        alignment.
    design : {"paired", "independent"}, optional
        Resampling design. ``"paired"`` (default) draws the same
        resampled row indices for every sample together, preserving
        within-row pairing (e.g. repeated measures on the same
        subject). ``"independent"`` resamples each sample's rows
        separately, appropriate when the samples are not linked
        row-by-row.
    n_resamples : int, optional
        Number of bootstrap resamples to draw. Default 10000.
    confidence_level : float, optional
        Nominal confidence level of the returned interval. Default
        0.95.
    seed : int or None, optional
        Seed for the resampling random number generator. Recorded in
        the result so the interval is exactly reproducible.
    method : str, optional
        Interval-construction method passed straight through to
        ``scipy.stats.bootstrap`` (e.g. ``"BCa"``, ``"percentile"``,
        ``"basic"``). Default ``"BCa"``.

    Returns
    -------
    BootstrapResult
        The point estimate, interval bounds, and the resolved
        configuration (design, method, resample count, seed). If fewer
        than 2 rows are supplied, ``meta.status`` is
        ``ResultStatus.INSUFFICIENT_DATA`` and the numeric fields are
        ``NaN``.

    Notes
    -----
    The point estimate is always the statistic evaluated once on the
    full, unresampled data; ``ci_low``/``ci_high`` come from the
    resampling distribution. Because ``design`` is an explicit argument
    rather than inferred from array shape, callers must state whether
    the samples are paired (e.g. before/after on the same subjects) or
    independent — silently guessing this from shape would risk
    resampling paired data as if independent, which changes the
    interval.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.resampling import bootstrap_ci
    >>> before = np.array([10.0, 12.0, 9.0, 15.0, 11.0, 13.0])
    >>> after = np.array([12.0, 13.0, 11.0, 17.0, 12.0, 15.0])
    >>> ids = np.array(["s1", "s2", "s3", "s4", "s5", "s6"])
    >>> def mean_diff(a, b):
    ...     return float(np.mean(b) - np.mean(a))
    >>> result = bootstrap_ci(
    ...     mean_diff, before, after, sample_ids=ids,
    ...     design="paired", n_resamples=500, method="percentile", seed=0,
    ... )
    >>> result.meta.status.value
    'valid'
    >>> round(result.point_estimate, 3)
    1.667
    >>> result.ci_low < result.point_estimate < result.ci_high
    True
    """

    all_ids = tuple(np.asarray(sample_ids, dtype=str).tolist())
    arrays = tuple(np.asarray(s, dtype=np.float64) for s in samples)
    n = arrays[0].shape[0]
    if n < 2:
        meta = build_meta(
            status=ResultStatus.INSUFFICIENT_DATA,
            all_ids=all_ids,
            included_ids=(),
            method="bootstrap_ci",
        )
        return BootstrapResult(
            meta=meta,
            point_estimate=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            seed=seed,
            method=method,
            design=design,
        )

    point_estimate = float(statistic(*arrays))
    result = _stats.bootstrap(
        arrays,
        statistic,
        n_resamples=n_resamples,
        paired=(design == "paired"),
        vectorized=False,
        confidence_level=confidence_level,
        method=method,
        random_state=seed,
    )
    meta = build_meta(
        status=ResultStatus.VALID,
        all_ids=all_ids,
        included_ids=all_ids,
        method="bootstrap_ci",
        configuration={
            "backend": "scipy.stats.bootstrap",
            "design": design,
            "n_resamples": n_resamples,
            "confidence_level": confidence_level,
            "method": method,
        },
        random_seed=seed,
    )
    return BootstrapResult(
        meta=meta,
        point_estimate=point_estimate,
        ci_low=float(result.confidence_interval.low),
        ci_high=float(result.confidence_interval.high),
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        seed=seed,
        method=method,
        design=design,
    )


def cluster_bootstrap_ci(
    statistic: Callable[..., float],
    *samples: NDArray[np.float64],
    cluster_ids: NDArray[np.str_],
    sample_ids: NDArray[np.str_],
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> BootstrapResult:
    """Clustered bootstrap confidence interval for dependent observations.

    Resamples whole clusters (e.g. subjects) with replacement rather
    than individual rows: when a cluster is chosen, every dependent
    observation belonging to it (e.g. every metabolite measured on that
    subject) is carried along together. This is the appropriate
    resampling unit when rows within a cluster are not independent.
    SciPy has no built-in clustered bootstrap, so the percentile
    interval is implemented directly here.

    Parameters
    ----------
    statistic : callable
        Function ``statistic(*samples) -> float`` computing the
        quantity of interest from one array per sample.
    *samples : ndarray of float
        Two or more equal-length 1-D arrays, one per sample, in the
        same row order as `sample_ids` and `cluster_ids`.
    cluster_ids : ndarray of str
        Cluster label for each row (e.g. subject identifier). Rows
        sharing a label are resampled as a unit.
    sample_ids : ndarray of str
        Identifier for each row, used only for provenance
        (``meta.included_ids``/``meta.excluded_ids``).
    n_resamples : int, optional
        Number of bootstrap resamples to draw. Default 10000.
    confidence_level : float, optional
        Nominal confidence level of the returned interval. Default
        0.95.
    seed : int or None, optional
        Seed for the resampling random number generator. Recorded in
        the result so the interval is exactly reproducible.

    Returns
    -------
    BootstrapResult
        The point estimate and a percentile interval built from
        cluster-level resamples (``method="cluster_bootstrap_percentile"``,
        ``design="cluster"``). If fewer than 2 independent clusters are
        present, ``meta.status`` is ``ResultStatus.INSUFFICIENT_DATA``
        and the numeric fields are ``NaN``.

    Notes
    -----
    On each of the ``n_resamples`` iterations, ``n_clusters`` cluster
    labels are drawn with replacement from the unique labels in
    `cluster_ids`; the rows belonging to the chosen clusters (with
    repeats, if a cluster is drawn more than once) are concatenated and
    passed to `statistic`. The interval is the
    :math:`[\\alpha/2, 1-\\alpha/2]` empirical quantile of the resulting
    replicate distribution, with :math:`\\alpha = 1 -` `confidence_level`:

    .. math::

        \\text{CI} = \\left[\\, \\hat{Q}_{\\alpha/2}(\\theta^*),\\;
        \\hat{Q}_{1-\\alpha/2}(\\theta^*) \\,\\right]

    where :math:`\\theta^*` denotes the cluster-bootstrap replicates of
    the statistic.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.resampling import cluster_bootstrap_ci
    >>> values = np.array([1.0, 1.2, 2.0, 2.1, 2.2, 3.0, 3.3, 3.1])
    >>> clusters = np.array(["s1", "s1", "s2", "s2", "s2", "s3", "s3", "s3"])
    >>> ids = np.array([f"r{i}" for i in range(8)])
    >>> def mean_stat(a):
    ...     return float(np.mean(a))
    >>> result = cluster_bootstrap_ci(
    ...     mean_stat, values, cluster_ids=clusters, sample_ids=ids,
    ...     n_resamples=500, seed=0,
    ... )
    >>> result.meta.status.value
    'valid'
    >>> round(result.point_estimate, 3)
    2.238
    >>> result.ci_low < result.point_estimate < result.ci_high
    True
    """

    all_ids = tuple(np.asarray(sample_ids, dtype=str).tolist())
    arrays = tuple(np.asarray(s, dtype=np.float64) for s in samples)
    clusters = np.asarray(cluster_ids, dtype=str)
    unique_clusters = np.unique(clusters)
    n_clusters = unique_clusters.shape[0]

    if n_clusters < 2:
        meta = build_meta(
            status=ResultStatus.INSUFFICIENT_DATA,
            all_ids=all_ids,
            included_ids=(),
            method="cluster_bootstrap_ci",
            warnings=(f"only {n_clusters} independent cluster(s) available",),
        )
        return BootstrapResult(
            meta=meta,
            point_estimate=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            seed=seed,
            method="cluster_bootstrap",
            design="cluster",
        )

    cluster_to_idx = {c: np.flatnonzero(clusters == c) for c in unique_clusters}
    rng = np.random.default_rng(seed)
    point_estimate = float(statistic(*arrays))

    replicates = np.empty(n_resamples, dtype=np.float64)
    for k in range(n_resamples):
        chosen = rng.choice(unique_clusters, size=n_clusters, replace=True)
        idx = np.concatenate([cluster_to_idx[c] for c in chosen])
        replicates[k] = statistic(*(arr[idx] for arr in arrays))

    alpha = 1.0 - confidence_level
    lo, hi = np.quantile(replicates, [alpha / 2.0, 1.0 - alpha / 2.0])

    meta = build_meta(
        status=ResultStatus.VALID,
        all_ids=all_ids,
        included_ids=all_ids,
        method="cluster_bootstrap_ci",
        configuration={
            "design": "cluster",
            "n_resamples": n_resamples,
            "confidence_level": confidence_level,
            "n_clusters": int(n_clusters),
            "method": "percentile",
        },
        random_seed=seed,
    )
    return BootstrapResult(
        meta=meta,
        point_estimate=point_estimate,
        ci_low=float(lo),
        ci_high=float(hi),
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        seed=seed,
        method="cluster_bootstrap_percentile",
        design="cluster",
    )


def permutation_test(
    statistic: Callable[..., float],
    *samples: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
    design: PermutationDesign,
    n_resamples: int = 10_000,
    seed: int | None = None,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
) -> PermutationResult:
    """Permutation test for a statistic of two or more samples.

    Wraps ``scipy.stats.permutation_test``, selecting SciPy's
    ``permutation_type`` from the explicit `design` argument rather
    than guessing it from array shape.

    Parameters
    ----------
    statistic : callable
        Function ``statistic(*samples) -> float`` computing the test
        statistic from one array per sample.
    *samples : ndarray of float
        Two or more arrays, one per sample, in the same row order as
        `sample_ids`.
    sample_ids : ndarray of str
        Identifier for each row, used only for provenance
        (``meta.included_ids``).
    design : {"independent", "paired_samples", "paired_association"}
        Permutation design, mapped one-to-one onto
        ``scipy.stats.permutation_test``'s ``permutation_type``:

        - ``"independent"`` -> ``"independent"``: samples are pooled
          and reshuffled without regard to any pairing.
        - ``"paired_samples"`` -> ``"samples"``: within each paired
          row, the values are randomly swapped between samples.
        - ``"paired_association"`` -> ``"pairings"``: the pairing
          (association) between rows is randomly permuted while each
          sample's values are held fixed.
    n_resamples : int, optional
        Number of random permutations to draw for the null
        distribution. Default 10000.
    seed : int or None, optional
        Seed for the permutation random number generator. Recorded in
        the result so the p-value is exactly reproducible.
    alternative : {"two-sided", "less", "greater"}, optional
        Alternative hypothesis. Default ``"two-sided"``.

    Returns
    -------
    PermutationResult
        The observed statistic, permutation p-value, and the resolved
        configuration (design, SciPy permutation type, resample count,
        seed, alternative).

    Notes
    -----
    Matching SciPy's own three-way distinction between permutation
    types (rather than inferring one from array shape) avoids silently
    treating paired data as independent or vice versa, which would
    produce an invalid null distribution.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.resampling import permutation_test
    >>> group_a = np.array([1.0, 2.0, 1.5, 2.5, 1.8])
    >>> group_b = np.array([3.0, 3.5, 2.8, 3.2, 4.0])
    >>> ids = np.array([f"r{i}" for i in range(10)])
    >>> def mean_diff(a, b):
    ...     return float(np.mean(b) - np.mean(a))
    >>> result = permutation_test(
    ...     mean_diff, group_a, group_b, sample_ids=ids,
    ...     design="independent", n_resamples=999, seed=0,
    ... )
    >>> result.meta.status.value
    'valid'
    >>> round(result.observed_statistic, 3)
    1.54
    >>> result.p_value < 0.05
    True
    """

    all_ids = tuple(np.asarray(sample_ids, dtype=str).tolist())
    arrays = tuple(np.asarray(s, dtype=np.float64) for s in samples)
    permutation_type = _PERMUTATION_TYPE_BY_DESIGN[design]

    result = _stats.permutation_test(
        arrays,
        statistic,
        permutation_type=permutation_type,
        vectorized=False,
        n_resamples=n_resamples,
        alternative=alternative,
        random_state=seed,
    )
    meta = build_meta(
        status=ResultStatus.VALID,
        all_ids=all_ids,
        included_ids=all_ids,
        method="permutation_test",
        configuration={
            "backend": "scipy.stats.permutation_test",
            "design": design,
            "scipy_permutation_type": permutation_type,
            "n_resamples": n_resamples,
            "alternative": alternative,
        },
        random_seed=seed,
    )
    return PermutationResult(
        meta=meta,
        observed_statistic=float(result.statistic),
        p_value=float(result.pvalue),
        n_resamples=n_resamples,
        seed=seed,
        design=design,
        alternative=alternative,
    )
