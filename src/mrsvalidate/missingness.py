"""Missingness detection and exclusion-mask construction.

No routine in this package silently replaces ``NaN``, ``+inf``, or
``-inf`` with a plausible-looking value. Instead, rows containing any
non-finite value across the arrays relevant to a computation are
excluded through an explicit, auditable mask, and the reason for each
exclusion is tallied so it can be reported in ``AnalysisMeta``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

#: Exclusion-reason priority order. A row with multiple simultaneous
#: problems (e.g. NaN in one column, +inf in another) is attributed to
#: the first matching reason in this order, so a single row is never
#: double-counted across reasons.
_REASON_PRIORITY: tuple[str, ...] = ("nan", "positive_infinity", "negative_infinity")


@dataclass(frozen=True)
class MissingnessReport:
    """Result of scanning one or more equal-length arrays for invalid rows.

    Every row of the scanned columns is classified as either included
    (finite in every scanned column) or excluded (non-finite in at
    least one), and this report records which sample IDs fell into each
    group along with a tally of exclusion reasons.

    Attributes
    ----------
    valid_mask : ndarray of bool
        Boolean mask, one entry per row, ``True`` where the row is
        finite in every scanned column and therefore included.
    included_ids : tuple of str
        Sample IDs for which ``valid_mask`` is ``True``, in original
        order.
    excluded_ids : tuple of str
        Sample IDs for which ``valid_mask`` is ``False``, in original
        order.
    exclusion_reasons : dict[str, int]
        Count of excluded rows per reason (``"nan"``,
        ``"positive_infinity"``, ``"negative_infinity"``), in priority
        order, omitting reasons with zero occurrences.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.missingness import analyze_missingness
    >>> report = analyze_missingness(
    ...     np.array(["s1", "s2", "s3"]), np.array([1.0, np.nan, 3.0])
    ... )
    >>> report.included_ids
    ('s1', 's3')
    >>> report.excluded_ids
    ('s2',)
    """

    valid_mask: NDArray[np.bool_]
    included_ids: tuple[str, ...]
    excluded_ids: tuple[str, ...]
    exclusion_reasons: dict[str, int]


def _row_reason(row_values: NDArray[np.float64]) -> str | None:
    if np.any(np.isnan(row_values)):
        return "nan"
    if np.any(np.isposinf(row_values)):
        return "positive_infinity"
    if np.any(np.isneginf(row_values)):
        return "negative_infinity"
    return None


def analyze_missingness(
    sample_ids: NDArray[np.str_], *columns: NDArray[np.float64]
) -> MissingnessReport:
    """Scan one or more aligned columns for non-finite values and build an exclusion mask.

    Parameters
    ----------
    sample_ids : ndarray of str
        One identifier per row, aligned with every array in ``columns``.
    *columns : ndarray of float
        One or more one-dimensional arrays, each of the same length as
        ``sample_ids``, to be scanned jointly for non-finite values. At
        least one column must be supplied.

    Returns
    -------
    MissingnessReport
        A report whose ``valid_mask`` is ``True`` for rows that are
        finite across every supplied column, together with the
        corresponding ``included_ids``, ``excluded_ids``, and a tally of
        ``exclusion_reasons``.

    Raises
    ------
    ValueError
        If no columns are supplied, or if any column's length does not
        match ``sample_ids``.

    Notes
    -----
    A row is excluded if *any* supplied column is non-finite at that
    row. When a row has multiple simultaneous problems (e.g. ``NaN`` in
    one column and ``+inf`` in another), it is attributed to a single
    reason using the priority order ``("nan", "positive_infinity",
    "negative_infinity")``, so a row is never double-counted across
    reasons.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.missingness import analyze_missingness
    >>> sample_ids = np.array(["s1", "s2", "s3", "s4"])
    >>> col_a = np.array([1.0, np.nan, 3.0, np.inf])
    >>> col_b = np.array([10.0, 20.0, 30.0, 40.0])
    >>> report = analyze_missingness(sample_ids, col_a, col_b)
    >>> report.included_ids
    ('s1', 's3')
    >>> report.excluded_ids
    ('s2', 's4')
    >>> report.exclusion_reasons
    {'nan': 1, 'positive_infinity': 1}
    """

    if not columns:
        raise ValueError("analyze_missingness requires at least one column")

    sample_ids = np.asarray(sample_ids, dtype=str)
    n = sample_ids.shape[0]
    stacked = np.column_stack([np.asarray(c, dtype=np.float64) for c in columns])
    if stacked.shape[0] != n:
        raise ValueError(
            f"Column length {stacked.shape[0]} does not match sample_ids length {n}"
        )

    valid_mask = np.all(np.isfinite(stacked), axis=1)
    reasons: dict[str, int] = {}
    for i in np.flatnonzero(~valid_mask):
        reason = _row_reason(stacked[i])
        assert reason is not None
        reasons[reason] = reasons.get(reason, 0) + 1

    # Ensure deterministic key order matching priority, omitting zero counts.
    ordered_reasons = {r: reasons[r] for r in _REASON_PRIORITY if r in reasons}

    included_ids = tuple(sample_ids[valid_mask].tolist())
    excluded_ids = tuple(sample_ids[~valid_mask].tolist())
    return MissingnessReport(
        valid_mask=valid_mask,
        included_ids=included_ids,
        excluded_ids=excluded_ids,
        exclusion_reasons=ordered_reasons,
    )
