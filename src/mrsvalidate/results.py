"""Universal result envelope shared by every analysis in mrsvalidate.

Every scientific computation returns a frozen dataclass that carries an
:class:`AnalysisMeta` describing what was actually computed, on how much
data, and why anything was excluded. No routine may silently replace
``NaN``, infinity, missing samples, zero denominators, singular
covariance matrices, or failed fits with plausible-looking values —
those situations are reported through ``status`` and ``exclusion_reasons``
instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


class ResultStatus(str, Enum):
    """Outcome category attached to every analysis result.

    Every :class:`AnalysisResult` carries exactly one ``ResultStatus``
    (inside its ``meta.status``) describing whether, and why, the
    reported numeric value should be treated as scientifically usable.

    Attributes
    ----------
    VALID : str
        The statistic was computed on sufficient, well-defined data.
    UNDEFINED : str
        The statistic is mathematically undefined for the supplied data
        (e.g. correlation of a constant vector), not merely
        unavailable.
    INSUFFICIENT_DATA : str
        Too few valid observations remained after exclusions.
    INVALID_INPUT : str
        The input violated a documented precondition (e.g. nonpositive
        predictive variance, missing uncertainty representation).
    FAILED : str
        The underlying numerical procedure did not converge or raised
        an error that could not be resolved deterministically.
    NOT_REQUESTED : str
        The analysis was never attempted (e.g. an optional component of
        a batch run that the caller explicitly excluded).

    Examples
    --------
    >>> from mrsvalidate.results import ResultStatus
    >>> ResultStatus.VALID.value
    'valid'
    >>> ResultStatus.VALID == "valid"
    True
    """

    VALID = "valid"
    UNDEFINED = "undefined"
    INSUFFICIENT_DATA = "insufficient_data"
    INVALID_INPUT = "invalid_input"
    FAILED = "failed"
    NOT_REQUESTED = "not_requested"


@dataclass(frozen=True)
class AnalysisMeta:
    """Provenance and inclusion/exclusion record attached to every result.

    An ``AnalysisMeta`` is embedded in every :class:`AnalysisResult` and
    is the single place where a caller can check how many samples went
    into a statistic, which sample IDs were used or excluded and why,
    and which resolved configuration choices (e.g. quantile method,
    near-zero tolerance) were actually applied.

    Attributes
    ----------
    status : ResultStatus
        Outcome category of the computation; see :class:`ResultStatus`.
    n_total : int
        Total number of samples considered before any exclusion.
    n_used : int
        Number of samples actually used in the computation.
    n_omitted : int
        Number of samples excluded from the computation.
    included_ids : tuple of str, optional
        Sample IDs that were used, in the order they were used.
        Default is ``()``.
    excluded_ids : tuple of str, optional
        Sample IDs that were excluded. Default is ``()``.
    exclusion_reasons : Mapping[str, int], optional
        Count of excluded samples per reason (e.g. ``{"nan": 2}``).
        Default is an empty mapping.
    warnings : tuple of str, optional
        Human-readable warnings raised during computation that do not
        by themselves invalidate the result. Default is ``()``.
    method : str, optional
        Name of the method or estimator used. Default is ``""``.
    method_version : str, optional
        Version string of the method implementation, for provenance
        tracking across package releases. Default is ``"0.1.0"``.
    configuration : Mapping[str, Any], optional
        Every resolved choice that affects the numeric value
        (definition mode, quantile method, near-zero tolerances, etc.),
        recorded explicitly rather than left as an undocumented
        default. Default is an empty mapping.
    random_seed : int or None, optional
        Random seed used by any stochastic step of the computation
        (e.g. bootstrap resampling), or ``None`` if the computation is
        deterministic. Default is ``None``.

    Raises
    ------
    ValueError
        If ``n_total``, ``n_used``, or ``n_omitted`` is negative, or if
        ``n_used + n_omitted != n_total`` while ``status`` is not one of
        ``INVALID_INPUT``, ``FAILED``, or ``NOT_REQUESTED`` (for which
        the sample counts may be meaningless).

    Examples
    --------
    >>> from mrsvalidate.results import AnalysisMeta, ResultStatus
    >>> meta = AnalysisMeta(
    ...     status=ResultStatus.VALID, n_total=3, n_used=2, n_omitted=1
    ... )
    >>> meta.status
    <ResultStatus.VALID: 'valid'>
    >>> meta.n_used, meta.n_omitted
    (2, 1)
    """

    status: ResultStatus
    n_total: int
    n_used: int
    n_omitted: int
    included_ids: tuple[str, ...] = ()
    excluded_ids: tuple[str, ...] = ()
    exclusion_reasons: Mapping[str, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    method: str = ""
    method_version: str = "0.1.0"
    configuration: Mapping[str, Any] = field(default_factory=dict)
    random_seed: int | None = None

    def __post_init__(self) -> None:
        if self.n_used < 0 or self.n_omitted < 0 or self.n_total < 0:
            raise ValueError("AnalysisMeta counts must be nonnegative")
        if self.n_used + self.n_omitted != self.n_total and self.status not in (
            ResultStatus.INVALID_INPUT,
            ResultStatus.FAILED,
            ResultStatus.NOT_REQUESTED,
        ):
            raise ValueError(
                "n_used + n_omitted must equal n_total "
                f"(got {self.n_used} + {self.n_omitted} != {self.n_total})"
            )


@dataclass(frozen=True)
class AnalysisResult:
    """Common base for every typed analysis result.

    Concrete result types (:class:`ScalarMetricResult`,
    :class:`IntervalResult`, etc.) subclass this and add their own
    numeric fields on top of the shared ``meta`` field.

    Attributes
    ----------
    meta : AnalysisMeta
        Provenance and inclusion/exclusion record for this result,
        always present so a result can be filtered or audited without
        knowing its concrete subtype.

    Examples
    --------
    >>> from mrsvalidate.results import AnalysisMeta, AnalysisResult, ResultStatus
    >>> meta = AnalysisMeta(
    ...     status=ResultStatus.VALID, n_total=1, n_used=1, n_omitted=0
    ... )
    >>> result = AnalysisResult(meta=meta)
    >>> result.meta.status
    <ResultStatus.VALID: 'valid'>
    """

    meta: AnalysisMeta


@dataclass(frozen=True)
class ScalarMetricResult(AnalysisResult):
    """A single scalar statistic, such as a mean, RMSE, or Pearson correlation.

    Attributes
    ----------
    meta : AnalysisMeta
        Provenance and inclusion/exclusion record for this result.
    value : float
        The computed scalar statistic.

    Examples
    --------
    >>> from mrsvalidate.results import AnalysisMeta, ResultStatus, ScalarMetricResult
    >>> meta = AnalysisMeta(
    ...     status=ResultStatus.VALID, n_total=2, n_used=2, n_omitted=0
    ... )
    >>> result = ScalarMetricResult(meta=meta, value=0.5)
    >>> result.value
    0.5
    """

    value: float


@dataclass(frozen=True)
class IntervalResult(AnalysisResult):
    """A two-sided interval, such as Tukey's outlier range or a confidence interval.

    Attributes
    ----------
    meta : AnalysisMeta
        Provenance and inclusion/exclusion record for this result.
    low : float
        Lower bound of the interval.
    high : float
        Upper bound of the interval.

    Examples
    --------
    >>> from mrsvalidate.results import AnalysisMeta, IntervalResult, ResultStatus
    >>> meta = AnalysisMeta(
    ...     status=ResultStatus.VALID, n_total=5, n_used=5, n_omitted=0
    ... )
    >>> result = IntervalResult(meta=meta, low=-1.0, high=7.0)
    >>> (result.low, result.high)
    (-1.0, 7.0)
    """

    low: float
    high: float


@dataclass(frozen=True)
class PerSampleResult(AnalysisResult):
    """A vector of per-sample values, ordered like ``meta.included_ids``.

    Attributes
    ----------
    meta : AnalysisMeta
        Provenance and inclusion/exclusion record for this result. Its
        ``included_ids`` gives the sample ID for each entry of
        ``values``, in the same order; excluded samples contribute no
        entry here (see ``meta.excluded_ids`` for why they were
        dropped).
    values : ndarray of float
        One value per included sample, ordered exactly like
        ``meta.included_ids``.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.results import AnalysisMeta, PerSampleResult, ResultStatus
    >>> meta = AnalysisMeta(
    ...     status=ResultStatus.VALID,
    ...     n_total=2,
    ...     n_used=2,
    ...     n_omitted=0,
    ...     included_ids=("s1", "s2"),
    ... )
    >>> result = PerSampleResult(meta=meta, values=np.array([1.0, 2.0]))
    >>> result.values
    array([1., 2.])
    """

    values: NDArray[np.float64]


@dataclass(frozen=True)
class MetricCollectionResult(AnalysisResult):
    """Several related scalar statistics computed together.

    Used, for example, when one call reports both the literal and a
    corrected-intent value of a metric under a single ``meta``.

    Attributes
    ----------
    meta : AnalysisMeta
        Provenance and inclusion/exclusion record for this result.
    values : Mapping[str, float]
        Mapping from statistic name to its computed value.

    Examples
    --------
    >>> from mrsvalidate.results import (
    ...     AnalysisMeta,
    ...     MetricCollectionResult,
    ...     ResultStatus,
    ... )
    >>> meta = AnalysisMeta(
    ...     status=ResultStatus.VALID, n_total=3, n_used=3, n_omitted=0
    ... )
    >>> result = MetricCollectionResult(
    ...     meta=meta, values={"rmse": 0.1, "rmse_corrected": 0.09}
    ... )
    >>> result.values["rmse"]
    0.1
    """

    values: Mapping[str, float]


def build_meta(
    *,
    status: ResultStatus,
    all_ids: tuple[str, ...],
    included_ids: tuple[str, ...],
    exclusion_reasons: Mapping[str, int] | None = None,
    warnings: tuple[str, ...] = (),
    method: str = "",
    method_version: str = "0.1.0",
    configuration: Mapping[str, Any] | None = None,
    random_seed: int | None = None,
) -> AnalysisMeta:
    """Build an :class:`AnalysisMeta` from the full and included ID sets.

    Parameters
    ----------
    status : ResultStatus
        Outcome category of the computation.
    all_ids : tuple of str
        Every sample ID considered before exclusion, in their original
        order.
    included_ids : tuple of str
        Sample IDs that were actually used in the computation.
    exclusion_reasons : Mapping[str, int] or None, optional
        Count of excluded samples per reason. Default is ``None``,
        recorded as an empty mapping.
    warnings : tuple of str, optional
        Human-readable warnings raised during computation. Default is
        ``()``.
    method : str, optional
        Name of the method or estimator used. Default is ``""``.
    method_version : str, optional
        Version string of the method implementation. Default is
        ``"0.1.0"``.
    configuration : Mapping[str, Any] or None, optional
        Every resolved choice that affects the numeric value. Default
        is ``None``, recorded as an empty mapping.
    random_seed : int or None, optional
        Random seed used by any stochastic step of the computation.
        Default is ``None``.

    Returns
    -------
    AnalysisMeta
        A metadata record whose ``n_total``, ``n_used``, ``n_omitted``,
        and ``excluded_ids`` are derived automatically as
        ``set(all_ids) - set(included_ids)``, so the counts can never
        drift out of sync with the ID lists. ``excluded_ids`` preserves
        the order of ``all_ids``.

    Examples
    --------
    >>> from mrsvalidate.results import build_meta, ResultStatus
    >>> meta = build_meta(
    ...     status=ResultStatus.VALID,
    ...     all_ids=("s1", "s2", "s3"),
    ...     included_ids=("s1", "s3"),
    ...     exclusion_reasons={"nan": 1},
    ... )
    >>> meta.n_total, meta.n_used, meta.n_omitted
    (3, 2, 1)
    >>> meta.excluded_ids
    ('s2',)
    """

    included = set(included_ids)
    excluded_ids = tuple(i for i in all_ids if i not in included)
    return AnalysisMeta(
        status=status,
        n_total=len(all_ids),
        n_used=len(included_ids),
        n_omitted=len(excluded_ids),
        included_ids=tuple(included_ids),
        excluded_ids=excluded_ids,
        exclusion_reasons=dict(exclusion_reasons or {}),
        warnings=tuple(warnings),
        method=method,
        method_version=method_version,
        configuration=dict(configuration or {}),
        random_seed=random_seed,
    )


def is_usable(result: AnalysisResult) -> bool:
    """Return whether a result carries a scientifically usable value.

    Parameters
    ----------
    result : AnalysisResult
        Any analysis result whose ``meta.status`` should be checked.

    Returns
    -------
    bool
        ``True`` if ``result.meta.status == ResultStatus.VALID``,
        ``False`` otherwise. Provided as a named helper so downstream
        filtering code (``[r for r in results if is_usable(r)]``) reads
        intention rather than repeating the enum comparison everywhere.

    Examples
    --------
    >>> from mrsvalidate.results import (
    ...     AnalysisMeta,
    ...     ResultStatus,
    ...     ScalarMetricResult,
    ...     is_usable,
    ... )
    >>> ok_meta = AnalysisMeta(
    ...     status=ResultStatus.VALID, n_total=1, n_used=1, n_omitted=0
    ... )
    >>> bad_meta = AnalysisMeta(
    ...     status=ResultStatus.FAILED, n_total=1, n_used=0, n_omitted=0
    ... )
    >>> is_usable(ScalarMetricResult(meta=ok_meta, value=1.0))
    True
    >>> is_usable(ScalarMetricResult(meta=bad_meta, value=float("nan")))
    False
    """

    return result.meta.status == ResultStatus.VALID
