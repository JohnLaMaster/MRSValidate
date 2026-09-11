"""Distributional-comparison statistics: the manuscript's ECWAS family.

This module implements the extended Cramer-Wilcoxon-Ansari-Savage
(ECWAS) family of omnibus distributional-comparison statistics (spec
eqs. 45-46 / manuscript eqs. ``ecwas`` / ``funato``): :func:`ecwas_kossler_mukherjee`
and :func:`ecwas_manuscript_max_shape`.

Notes
-----
**Design.** The manuscript's design is an **independent two-sample**
comparison (spec: never silently switch this to paired because it is
more convenient) combining three standardized rank-based components:

- ``WI``: Wilcoxon-Mann-Whitney U (location)
- ``AB``: Ansari-Bradley (scale)
- ``AS`` (Kössler-Mukherjee) or ``max(AS, S)`` (Funato-style extension):
  AntiSavage / Savage (shape)

**Linear rank statistics.** Each component is a *linear rank
statistic*: assign every observation in the pooled, rank-ordered sample
a score from a fixed score function, sum the scores falling in group
1, and standardize by that statistic's exact null mean/variance under
random allocation (Hájek-Šidák finite-population sampling theory,
which automatically handles ties correctly via midrank-averaged scores
— no separate tie-correction formula needed). ``ecwas_kossler_mukherjee``
and ``ecwas_manuscript_max_shape`` are kept as two distinct,
individually named functions (spec: do not conflate the manuscript's
own simplification with the published adaptive/max-type families) and
both expose their component statistics in the result.

The published max-type Funato extension's null mean/variance for
``max(AS, S)`` has no simple closed form; this implementation estimates
it by Monte Carlo simulation under random relabeling (seeded, resample
count recorded) rather than asserting an unverified closed-form value.

**Critical values.** The composite statistic is a sum of three squared
standardized statistics — a squared Euclidean distance in 3-space —
which is asymptotically chi-squared with 3 df when the three components
are (asymptotically) independent standard normal under H0, the standard
result for this kind of quadratic-form combination (Kössler & Mukherjee
2019/2020 derive the exact limiting null distribution; this
implementation uses the standard chi2(df=3) approximation to it, not a
re-derivation of their exact asymptotics). Two distinct, switchable
methods are offered per call via ``critical_value_method``; they are
not equivalent and are not interchangeable defaults:

``"chi2_asymptotic"`` (fast)
    ``chi2.ppf(1-alpha, df=3)`` — exact under the independence/normality
    approximation above. This is the default for
    :func:`ecwas_kossler_mukherjee`.
``"monte_carlo"``
    A joint permutation null of the *whole* composite (all three
    components recomputed together on each random relabeling of the
    pooled sample), which does not assume independence between
    components and is more appropriate for small samples or tie-heavy
    data (spec: "favor permutation calibration ... instead of pretending
    continuous-distribution asymptotics remain exact"). Also yields a
    permutation p-value. This is the default for
    :func:`ecwas_manuscript_max_shape`, since its max-type shape
    component is non-normal even asymptotically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats as _stats

from ..results import AnalysisResult, ResultStatus, build_meta
from .sample_size import MINIMUM_INDEPENDENT_SAMPLES, check_minimum_sample_size

__all__ = [
    "ecwas_kossler_mukherjee",
    "ecwas_manuscript_max_shape",
    "ECWASResult",
]

ShapeComponent = Literal["antisavage", "savage", "max_antisavage_savage"]
CriticalValueMethod = Literal["chi2_asymptotic", "monte_carlo"]


@dataclass(frozen=True)
class ECWASResult(AnalysisResult):
    """Result of an ECWAS omnibus distributional-comparison test.

    Returned by :func:`ecwas_kossler_mukherjee` and
    :func:`ecwas_manuscript_max_shape`.

    Attributes
    ----------
    meta : AnalysisMeta
        Provenance and inclusion/exclusion record (status, sample
        counts, resolved configuration, random seed, warnings).
    statistic : float
        Composite ECWAS statistic: the sum of the three squared
        standardized component statistics
        (``wilcoxon_z**2 + ansari_bradley_z**2 + shape_z**2``).
    wilcoxon_z : float
        Standardized Wilcoxon-Mann-Whitney (location) component.
    ansari_bradley_z : float
        Standardized Ansari-Bradley (scale) component.
    shape_z : float
        Standardized shape component (AntiSavage, or the max-type
        AntiSavage/Savage combination, depending on `shape_component`).
    wilcoxon_statistic : float
        Raw (unstandardized) Wilcoxon-Mann-Whitney rank-sum statistic.
    ansari_bradley_statistic : float
        Raw (unstandardized) Ansari-Bradley rank-sum statistic.
    shape_statistic : float
        Raw (unstandardized) shape rank-sum statistic.
    shape_component : {"antisavage", "savage", "max_antisavage_savage"}
        Which shape component was used to build `shape_z`.
    n1 : int
        Number of observations in the first (``x``) sample.
    n2 : int
        Number of observations in the second (``y``) sample.
    critical_value : float
        Critical value of the composite statistic at `alpha`, obtained
        by `critical_value_method`.
    critical_value_method : {"chi2_asymptotic", "monte_carlo"}
        Method actually used to obtain `critical_value` (see the module
        docstring's Notes for the distinction between the two).
    alpha : float
        Nominal significance level used for `critical_value` and
        `reject_null`.
    p_value : float or None
        P-value for the composite statistic: the chi-squared
        upper-tail probability when `critical_value_method` is
        ``"chi2_asymptotic"``, or the permutation p-value when it is
        ``"monte_carlo"``.
    reject_null : bool
        Whether `statistic` exceeds `critical_value` (equivalently,
        whether the null hypothesis of equal distributions is rejected
        at `alpha`).
    """

    statistic: float
    wilcoxon_z: float
    ansari_bradley_z: float
    shape_z: float
    wilcoxon_statistic: float
    ansari_bradley_statistic: float
    shape_statistic: float
    shape_component: ShapeComponent
    n1: int
    n2: int
    critical_value: float
    critical_value_method: CriticalValueMethod
    alpha: float
    p_value: float | None
    reject_null: bool


def _empty_ecwas_result(meta, shape_component: ShapeComponent, n1: int, n2: int, alpha: float, method: CriticalValueMethod) -> ECWASResult:
    return ECWASResult(
        meta=meta, statistic=float("nan"), wilcoxon_z=float("nan"), ansari_bradley_z=float("nan"),
        shape_z=float("nan"), wilcoxon_statistic=float("nan"), ansari_bradley_statistic=float("nan"),
        shape_statistic=float("nan"), shape_component=shape_component, n1=n1, n2=n2,
        critical_value=float("nan"), critical_value_method=method, alpha=alpha, p_value=None,
        reject_null=False,
    )


def _block_averaged_scores(
    combined: NDArray[np.float64], raw_scores_by_rank: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Map each observation to the average of ``raw_scores_by_rank`` (indexed
    by plain rank 1..N) over its tie block, i.e. the midrank analogue of the
    given rank-based score function."""

    n = combined.shape[0]
    order = np.argsort(combined, kind="mergesort")
    sorted_vals = combined[order]
    out = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        out[order[i : j + 1]] = np.mean(raw_scores_by_rank[i : j + 1])
        i = j + 1
    return out


def _wilcoxon_raw_scores(n: int) -> NDArray[np.float64]:
    return np.arange(1, n + 1, dtype=np.float64)


def _ansari_bradley_raw_scores(n: int) -> NDArray[np.float64]:
    r = np.arange(1, n + 1, dtype=np.float64)
    return np.minimum(r, n + 1 - r)


def _savage_raw_scores(n: int) -> NDArray[np.float64]:
    r = np.arange(1, n + 1, dtype=np.float64)
    cumulative = np.cumsum(1.0 / (n - r + 1.0))
    return cumulative - 1.0


def _antisavage_raw_scores(n: int) -> NDArray[np.float64]:
    return _savage_raw_scores(n)[::-1]


def _score_moments(scores_all: NDArray[np.float64], n1: int) -> tuple[float, float]:
    n = scores_all.shape[0]
    n2 = n - n1
    mean_score = float(np.mean(scores_all))
    mu = n1 * mean_score
    if n <= 1:
        return mu, float("nan")
    var = (n1 * n2 / (n * (n - 1))) * float(np.sum((scores_all - mean_score) ** 2))
    return mu, var


def _standardized_component(
    x: NDArray[np.float64], y: NDArray[np.float64], raw_score_fn
) -> tuple[float, float, float]:
    combined = np.concatenate([x, y])
    n1 = x.shape[0]
    raw = raw_score_fn(combined.shape[0])
    scores_all = _block_averaged_scores(combined, raw)
    statistic = float(np.sum(scores_all[:n1]))
    mu, var = _score_moments(scores_all, n1)
    sigma = float(np.sqrt(var)) if var > 0 else float("nan")
    z = (statistic - mu) / sigma if sigma and np.isfinite(sigma) and sigma > 0 else float("nan")
    return statistic, z, sigma


def _base_checks(
    x: NDArray[np.float64], y: NDArray[np.float64], method: str
) -> tuple[ResultStatus, tuple[str, ...]]:
    warnings: list[str] = []
    if x.shape[0] < 2 or y.shape[0] < 2:
        return ResultStatus.INSUFFICIENT_DATA, tuple(warnings)
    if not check_minimum_sample_size(x.shape[0]) or not check_minimum_sample_size(y.shape[0]):
        warnings.append(
            f"manuscript recommends n>={MINIMUM_INDEPENDENT_SAMPLES} per group "
            "for reliable ECWAS asymptotics"
        )
    return ResultStatus.VALID, tuple(warnings)


def ecwas_kossler_mukherjee(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
    *,
    alpha: float = 0.05,
    critical_value_method: CriticalValueMethod = "chi2_asymptotic",
    n_null_resamples: int = 2000,
    seed: int | None = None,
) -> ECWASResult:
    """Kössler-Mukherjee ECWAS omnibus distributional-comparison test.

    Combines three standardized linear rank statistics — Wilcoxon-Mann-Whitney
    (location), Ansari-Bradley (scale), and AntiSavage (shape) — each
    standardized by its exact null mean/variance, into a single
    squared-Euclidean composite statistic.

    Parameters
    ----------
    x : ndarray of float
        First sample.
    y : ndarray of float
        Second sample, independent of `x` (spec: independent
        two-sample design, never paired).
    sample_ids : ndarray of str
        Identifiers for the pooled ``x``/``y`` rows (``x`` first, then
        ``y``), used only for provenance.
    alpha : float, optional
        Nominal significance level. Default 0.05.
    critical_value_method : {"chi2_asymptotic", "monte_carlo"}, optional
        How to obtain the composite statistic's critical value; see
        the module docstring's Notes for the distinction. Default
        ``"chi2_asymptotic"``.
    n_null_resamples : int, optional
        Number of random relabelings used to build the joint
        permutation null when `critical_value_method` is
        ``"monte_carlo"``. Ignored otherwise. Default 2000.
    seed : int or None, optional
        Seed for the permutation random number generator, used only
        when `critical_value_method` is ``"monte_carlo"``. Recorded in
        the result so it is exactly reproducible.

    Returns
    -------
    ECWASResult
        The composite statistic, its three standardized components,
        the critical value and p-value, and whether the null of equal
        distributions is rejected. ``meta.status`` is
        ``ResultStatus.INSUFFICIENT_DATA`` if either sample has fewer
        than 2 observations, in which case the numeric fields are
        ``NaN``. A warning is attached to ``meta.warnings`` (without
        invalidating the result) if either sample is smaller than
        :data:`~mrsvalidate.analysis.sample_size.MINIMUM_INDEPENDENT_SAMPLES`.

    Raises
    ------
    ValueError
        If `critical_value_method` is not one of
        ``"chi2_asymptotic"`` or ``"monte_carlo"``.

    Notes
    -----
    Each component score function is applied to the pooled,
    rank-ordered sample with midrank averaging over tied values, then
    summed over group 1 and standardized by its exact null mean and
    variance under random allocation (Hájek-Šidák sampling theory).
    The composite statistic

    .. math::

        Q = Z_{WI}^2 + Z_{AB}^2 + Z_{AS}^2

    is a squared Euclidean distance in the 3-space of standardized
    components. Two switchable ways to obtain its critical value are
    offered via `critical_value_method`:

    ``"chi2_asymptotic"``
        :math:`Q \\sim \\chi^2_3` asymptotically when the three
        components are (asymptotically) independent standard normal
        under :math:`H_0`; the critical value is
        ``chi2.ppf(1-alpha, df=3)`` and the p-value is
        ``chi2.sf(Q, df=3)``. Kössler & Mukherjee (2019/2020) derive
        the exact limiting null distribution; this implementation uses
        the standard chi-squared approximation to it rather than
        re-deriving their exact asymptotics.
    ``"monte_carlo"``
        A joint permutation null: all three components are
        recomputed together on each of `n_null_resamples` random
        relabelings of the pooled sample, and the critical value is
        the empirical :math:`1-\\alpha` quantile of the resulting
        composite-statistic null distribution. This does not assume
        independence between components and is more appropriate for
        small samples or tie-heavy data than the chi-squared
        approximation.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.distribution import ecwas_kossler_mukherjee
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(0, 1, size=20)
    >>> y = rng.normal(0.8, 1.5, size=20)
    >>> ids = np.array([f"r{i}" for i in range(40)])
    >>> result = ecwas_kossler_mukherjee(x, y, ids, seed=0)
    >>> result.meta.status.value
    'valid'
    >>> round(result.statistic, 3)
    16.701
    >>> round(result.critical_value, 3)
    7.815
    >>> result.reject_null
    True
    """

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    all_ids = tuple(np.asarray(sample_ids, dtype=str).tolist())
    status, warnings = _base_checks(x, y, "ecwas_kossler_mukherjee")

    if status is not ResultStatus.VALID:
        meta = build_meta(
            status=status, all_ids=all_ids, included_ids=(),
            method="ecwas_kossler_mukherjee", warnings=warnings,
        )
        return _empty_ecwas_result(meta, "antisavage", x.shape[0], y.shape[0], alpha, critical_value_method)

    wi_stat, wi_z, _ = _standardized_component(x, y, _wilcoxon_raw_scores)
    ab_stat, ab_z, _ = _standardized_component(x, y, _ansari_bradley_raw_scores)
    as_stat, as_z, _ = _standardized_component(x, y, _antisavage_raw_scores)
    composite = float(wi_z**2 + ab_z**2 + as_z**2)

    config = {
        "components": ["wilcoxon_mann_whitney", "ansari_bradley", "antisavage"],
        "design": "independent",
        "reference": "Kossler & Mukherjee",
        "alpha": alpha,
        "critical_value_method": critical_value_method,
    }
    p_value: float | None
    if critical_value_method == "chi2_asymptotic":
        critical_value = float(_stats.chi2.ppf(1 - alpha, df=3))
        p_value = float(_stats.chi2.sf(composite, df=3))
        config["critical_value_source"] = "chi2.ppf(1-alpha, df=3)"
        random_seed = None
    elif critical_value_method == "monte_carlo":
        combined = np.concatenate([x, y])
        n1, n = x.shape[0], combined.shape[0]
        wi_raw = _block_averaged_scores(combined, _wilcoxon_raw_scores(n))
        ab_raw = _block_averaged_scores(combined, _ansari_bradley_raw_scores(n))
        as_raw = _block_averaged_scores(combined, _antisavage_raw_scores(n))
        wi_mu, wi_var = _score_moments(wi_raw, n1)
        ab_mu, ab_var = _score_moments(ab_raw, n1)
        as_mu, as_var = _score_moments(as_raw, n1)
        wi_sigma, ab_sigma, as_sigma = np.sqrt(wi_var), np.sqrt(ab_var), np.sqrt(as_var)

        rng = np.random.default_rng(seed)
        idx = np.arange(n)
        null_composite = np.empty(n_null_resamples)
        for k in range(n_null_resamples):
            group1 = rng.permutation(idx)[:n1]
            zw = (np.sum(wi_raw[group1]) - wi_mu) / wi_sigma
            za = (np.sum(ab_raw[group1]) - ab_mu) / ab_sigma
            zs = (np.sum(as_raw[group1]) - as_mu) / as_sigma
            null_composite[k] = zw**2 + za**2 + zs**2

        critical_value = float(np.quantile(null_composite, 1 - alpha))
        p_value = float((1 + np.sum(null_composite >= composite)) / (n_null_resamples + 1))
        config["critical_value_source"] = "joint_permutation_null"
        config["n_null_resamples"] = n_null_resamples
        random_seed = seed
    else:
        raise ValueError(f"Unknown critical_value_method: {critical_value_method!r}")

    meta = build_meta(
        status=ResultStatus.VALID, all_ids=all_ids, included_ids=all_ids,
        method="ecwas_kossler_mukherjee", configuration=config, warnings=warnings,
        random_seed=random_seed,
    )
    return ECWASResult(
        meta=meta, statistic=composite, wilcoxon_z=wi_z, ansari_bradley_z=ab_z, shape_z=as_z,
        wilcoxon_statistic=wi_stat, ansari_bradley_statistic=ab_stat, shape_statistic=as_stat,
        shape_component="antisavage", n1=x.shape[0], n2=y.shape[0],
        critical_value=critical_value, critical_value_method=critical_value_method,
        alpha=alpha, p_value=p_value, reject_null=bool(composite > critical_value),
    )


def ecwas_manuscript_max_shape(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
    *,
    alpha: float = 0.05,
    critical_value_method: CriticalValueMethod = "monte_carlo",
    n_null_resamples: int = 2000,
    seed: int | None = None,
) -> ECWASResult:
    """Manuscript's Funato-style ECWAS extension with a max-type shape term.

    Combines standardized Wilcoxon-Mann-Whitney (location) and
    Ansari-Bradley (scale) components with a standardized
    ``max(AntiSavage, Savage)`` shape component into a squared-Euclidean
    composite statistic. This is a manuscript-specific simplification,
    explicitly **not** represented as the validated published Funato
    et al. procedure.

    Parameters
    ----------
    x : ndarray of float
        First sample.
    y : ndarray of float
        Second sample, independent of `x` (spec: independent
        two-sample design, never paired).
    sample_ids : ndarray of str
        Identifiers for the pooled ``x``/``y`` rows (``x`` first, then
        ``y``), used only for provenance.
    alpha : float, optional
        Nominal significance level. Default 0.05.
    critical_value_method : {"chi2_asymptotic", "monte_carlo"}, optional
        How to obtain the **composite** statistic's critical value
        (independent of the shape component's own standardization,
        which always uses Monte Carlo — see Notes). Default
        ``"monte_carlo"``.
    n_null_resamples : int, optional
        Number of random relabelings used both to standardize the
        max-type shape component and, when `critical_value_method` is
        ``"monte_carlo"``, to build the joint permutation null for the
        composite. Default 2000.
    seed : int or None, optional
        Seed for the permutation random number generator. Recorded in
        the result so it is exactly reproducible.

    Returns
    -------
    ECWASResult
        The composite statistic, its three standardized components
        (``shape_component="max_antisavage_savage"``), the critical
        value and p-value, and whether the null of equal distributions
        is rejected. ``meta.status`` is
        ``ResultStatus.INSUFFICIENT_DATA`` if either sample has fewer
        than 2 observations, in which case the numeric fields are
        ``NaN``. A warning is attached to ``meta.warnings`` (without
        invalidating the result) if either sample is smaller than
        :data:`~mrsvalidate.analysis.sample_size.MINIMUM_INDEPENDENT_SAMPLES`.

    Raises
    ------
    ValueError
        If `critical_value_method` is not one of
        ``"chi2_asymptotic"`` or ``"monte_carlo"``.

    Notes
    -----
    Unlike :func:`ecwas_kossler_mukherjee`'s AntiSavage-only shape
    component, ``max(AntiSavage, Savage)`` has no simple closed-form
    null mean/variance. Its standardization therefore *always* uses
    Monte Carlo simulation: `n_null_resamples` random relabelings of
    the pooled sample give an empirical null distribution for
    ``max(AS, S)``, whose mean and standard deviation standardize the
    observed value into `shape_z`.

    For the **composite** statistic

    .. math::

        Q = Z_{WI}^2 + Z_{AB}^2 + Z_{shape}^2

    two switchable critical-value methods are offered via
    `critical_value_method`, independently of the shape component's
    own (always Monte Carlo) standardization:

    ``"monte_carlo"`` (default here)
        A joint permutation null: all three components are recomputed
        together on each random relabeling, and the critical value is
        the empirical :math:`1-\\alpha` quantile of the resulting
        composite-statistic null distribution. This is the more
        defensible default for this function specifically, since the
        max-type shape component is itself non-normal even
        asymptotically, unlike the Kössler-Mukherjee AntiSavage
        component.
    ``"chi2_asymptotic"``
        ``chi2.ppf(1-alpha, df=3)`` / ``chi2.sf(Q, df=3)``, offered as
        a fast approximation on the same independence/asymptotic-normality
        grounds as :func:`ecwas_kossler_mukherjee`, despite the
        shape component's non-normality.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.distribution import ecwas_manuscript_max_shape
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(0, 1, size=20)
    >>> y = rng.normal(0.8, 1.5, size=20)
    >>> ids = np.array([f"r{i}" for i in range(40)])
    >>> result = ecwas_manuscript_max_shape(x, y, ids, seed=0, n_null_resamples=500)
    >>> result.meta.status.value
    'valid'
    >>> result.shape_component
    'max_antisavage_savage'
    >>> round(result.statistic, 3)
    18.387
    >>> result.reject_null
    True
    """

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    all_ids = tuple(np.asarray(sample_ids, dtype=str).tolist())
    status, warnings = _base_checks(x, y, "ecwas_manuscript_max_shape")

    if status is not ResultStatus.VALID:
        meta = build_meta(
            status=status, all_ids=all_ids, included_ids=(),
            method="ecwas_manuscript_max_shape", warnings=warnings,
        )
        return _empty_ecwas_result(
            meta, "max_antisavage_savage", x.shape[0], y.shape[0], alpha, critical_value_method
        )

    combined = np.concatenate([x, y])
    n1, n = x.shape[0], combined.shape[0]
    wi_raw = _block_averaged_scores(combined, _wilcoxon_raw_scores(n))
    ab_raw = _block_averaged_scores(combined, _ansari_bradley_raw_scores(n))
    as_raw = _block_averaged_scores(combined, _antisavage_raw_scores(n))
    s_raw = _block_averaged_scores(combined, _savage_raw_scores(n))

    wi_mu, wi_var = _score_moments(wi_raw, n1)
    ab_mu, ab_var = _score_moments(ab_raw, n1)
    wi_sigma, ab_sigma = np.sqrt(wi_var), np.sqrt(ab_var)

    wi_stat = float(np.sum(wi_raw[:n1]))
    ab_stat = float(np.sum(ab_raw[:n1]))
    as_stat = float(np.sum(as_raw[:n1]))
    s_stat = float(np.sum(s_raw[:n1]))
    max_stat = max(as_stat, s_stat)
    wi_z = (wi_stat - wi_mu) / wi_sigma
    ab_z = (ab_stat - ab_mu) / ab_sigma

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    null_max = np.empty(n_null_resamples)
    null_composite_wi = np.empty(n_null_resamples)
    null_composite_ab = np.empty(n_null_resamples)
    for k in range(n_null_resamples):
        group1 = rng.permutation(idx)[:n1]
        null_max[k] = max(float(np.sum(as_raw[group1])), float(np.sum(s_raw[group1])))
        null_composite_wi[k] = (np.sum(wi_raw[group1]) - wi_mu) / wi_sigma
        null_composite_ab[k] = (np.sum(ab_raw[group1]) - ab_mu) / ab_sigma

    mu_max = float(np.mean(null_max))
    sigma_max = float(np.std(null_max, ddof=1))
    shape_z = (max_stat - mu_max) / sigma_max if sigma_max > 0 else float("nan")
    composite = float(wi_z**2 + ab_z**2 + shape_z**2)

    config = {
        "components": ["wilcoxon_mann_whitney", "ansari_bradley", "max(antisavage, savage)"],
        "design": "independent",
        "reference": "manuscript_funato_style_extension",
        "shape_null_moments": "monte_carlo",
        "n_null_resamples": n_null_resamples,
        "alpha": alpha,
        "critical_value_method": critical_value_method,
    }
    if critical_value_method == "chi2_asymptotic":
        critical_value = float(_stats.chi2.ppf(1 - alpha, df=3))
        p_value = float(_stats.chi2.sf(composite, df=3))
        config["critical_value_source"] = "chi2.ppf(1-alpha, df=3)"
    elif critical_value_method == "monte_carlo":
        null_shape_z = (null_max - mu_max) / sigma_max
        null_composite = null_composite_wi**2 + null_composite_ab**2 + null_shape_z**2
        critical_value = float(np.quantile(null_composite, 1 - alpha))
        p_value = float((1 + np.sum(null_composite >= composite)) / (n_null_resamples + 1))
        config["critical_value_source"] = "joint_permutation_null"
    else:
        raise ValueError(f"Unknown critical_value_method: {critical_value_method!r}")

    meta = build_meta(
        status=ResultStatus.VALID, all_ids=all_ids, included_ids=all_ids,
        method="ecwas_manuscript_max_shape", configuration=config, warnings=warnings,
        random_seed=seed,
    )
    return ECWASResult(
        meta=meta, statistic=composite, wilcoxon_z=wi_z, ansari_bradley_z=ab_z, shape_z=shape_z,
        wilcoxon_statistic=wi_stat, ansari_bradley_statistic=ab_stat, shape_statistic=max_stat,
        shape_component="max_antisavage_savage", n1=x.shape[0], n2=y.shape[0],
        critical_value=critical_value, critical_value_method=critical_value_method,
        alpha=alpha, p_value=p_value, reject_null=bool(composite > critical_value),
    )
