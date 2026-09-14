"""Minimum sample size / cluster count rules of thumb.

This module implements the manuscript's minimum-sample-size and
minimum-cluster-count heuristics (spec eqs. 3-10).

Notes
-----
These are diagnostic checks, not computations that alter a result's
numeric value. Callers use them to decide whether to trust an
asymptotic result or fall back to bootstrap/permutation inference; a
``False`` return does not itself invalidate a computation, it is a
signal that the caller should consider resampling-based inference
instead.
"""

from __future__ import annotations

__all__ = ["MINIMUM_INDEPENDENT_CLUSTERS", "MINIMUM_INDEPENDENT_SAMPLES", "check_minimum_clusters", "check_minimum_sample_size"]

#: Manuscript's recommended minimum independent-sample size for
#: Pearson-r-based and similar asymptotic inference.
MINIMUM_INDEPENDENT_SAMPLES = 500

#: Standard rule of thumb for reliable cluster-robust inference (spec:
#: "at least 30 independent clusters").
MINIMUM_INDEPENDENT_CLUSTERS = 30


def check_minimum_sample_size(n: int, *, minimum: int = MINIMUM_INDEPENDENT_SAMPLES) -> bool:
    """Check whether a sample size meets the recommended minimum.

    Compares ``n`` independent observations against the manuscript's
    recommended minimum for asymptotic inference (e.g. Pearson-r-based
    procedures) to remain reliable.

    Parameters
    ----------
    n : int
        Number of independent samples actually available.
    minimum : int, optional
        Recommended minimum independent-sample size. Defaults to
        :data:`MINIMUM_INDEPENDENT_SAMPLES` (500).

    Returns
    -------
    bool
        ``True`` if ``n >= minimum``, ``False`` otherwise.

    Notes
    -----
    This is a diagnostic check only; it does not alter any computed
    statistic. A ``False`` result signals that the caller should
    consider bootstrap or permutation inference instead of relying on
    asymptotic approximations.

    Examples
    --------
    >>> check_minimum_sample_size(500)
    True
    >>> check_minimum_sample_size(499)
    False
    >>> check_minimum_sample_size(120, minimum=100)
    True
    """

    return n >= minimum


def check_minimum_clusters(n_clusters: int, *, minimum: int = MINIMUM_INDEPENDENT_CLUSTERS) -> bool:
    """Check whether a cluster count meets the recommended minimum.

    Compares the number of independent clusters (e.g. subjects,
    spectra) against the standard rule of thumb for reliable
    cluster-robust inference.

    Parameters
    ----------
    n_clusters : int
        Number of independent clusters actually available.
    minimum : int, optional
        Recommended minimum cluster count. Defaults to
        :data:`MINIMUM_INDEPENDENT_CLUSTERS` (30).

    Returns
    -------
    bool
        ``True`` if ``n_clusters >= minimum``, ``False`` otherwise.

    Notes
    -----
    This is a diagnostic check only; it does not alter any computed
    statistic. A ``False`` result signals that cluster-robust standard
    errors, ICC-based effective-sample-size corrections, or cluster
    bootstrap results computed from these clusters may be unreliable,
    per the "at least 30 independent clusters" rule of thumb.

    Examples
    --------
    >>> check_minimum_clusters(30)
    True
    >>> check_minimum_clusters(12)
    False
    >>> check_minimum_clusters(8, minimum=5)
    True
    """

    return n_clusters >= minimum
