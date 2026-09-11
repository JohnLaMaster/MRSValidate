"""Normality test suite.

This module implements five normality tests: Shapiro-Wilk,
Kolmogorov-Smirnov (fully specified null), Lilliefors (KS with
estimated parameters), Anderson-Darling, and D'Agostino-Pearson.

Notes
-----
Each test wraps a mature, tested primitive (SciPy or Statsmodels)
rather than reimplementing test statistics from scratch — matching the
"reuse mature scientific Python functionality" rule used throughout
this package. Every result additionally reports a **critical value**
at the caller's ``alpha``, obtained one of three ways depending on what
is actually available for that specific test (all three are
legitimate; this module switches between them per-test rather than
forcing one approach on every test):

- **Table lookup** (:func:`anderson_darling`, :func:`lilliefors`):
  Anderson-Darling uses SciPy's own ``critical_values`` (Stephens 1974
  adjusted critical values, returned directly by
  ``scipy.stats.anderson``); Lilliefors uses Statsmodels' internal
  10-million-replicate simulated critical-value table
  (``statsmodels.stats._lilliefors``), via 2D interpolation over
  ``(alpha, n)``. Neither table is reimplemented or re-simulated here
  — both are used as-is from their respective libraries.
- **Distributional equation** (:func:`dagostino_pearson`,
  :func:`kolmogorov_smirnov`): D'Agostino-Pearson's K² statistic is
  chi-squared with 2 df under H0, so its critical value is the exact
  ``chi2.ppf(1-alpha, df=2)``; the classical one-sample KS test
  (fully specified null, not estimated from the same sample) has a
  known asymptotic limiting distribution (``scipy.stats.kstwobign``),
  so its critical value is computed from that directly.
- **Monte Carlo simulation** (:func:`shapiro_wilk`): Shapiro-Wilk has
  no simple closed-form or bundled critical-value table in SciPy.
  Rather than transcribing an unverified table from memory, its
  critical value is obtained by simulating W under H0 for the given
  ``n`` (seeded, resample count recorded) — the same honest approach
  already used in this package for the ECWAS max-shape component's
  null moments.

This table-vs-equation-vs-Monte-Carlo distinction is preserved
per-function in each test's own docstring below, since it is the whole
point of documenting this module: callers should know, for each test,
whether the reported critical value comes from an external table, a
closed-form limiting distribution, or a seeded simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats as _stats

from ..results import AnalysisResult, ResultStatus, build_meta
from ._shared import meta_for, valid_subset

__all__ = [
    "shapiro_wilk",
    "kolmogorov_smirnov",
    "lilliefors",
    "anderson_darling",
    "dagostino_pearson",
    "normality_test_suite",
    "NormalityTestResult",
]

_ANDERSON_LEVELS_PCT = (15.0, 10.0, 5.0, 2.5, 1.0)


@dataclass(frozen=True)
class NormalityTestResult(AnalysisResult):
    """Result of a single normality test.

    Returned by :func:`shapiro_wilk`, :func:`kolmogorov_smirnov`,
    :func:`lilliefors`, :func:`anderson_darling`, and
    :func:`dagostino_pearson`; also the value type of the dict returned
    by :func:`normality_test_suite`.

    Attributes
    ----------
    meta : AnalysisMeta
        Provenance and inclusion/exclusion record (status, sample
        counts, resolved configuration, random seed, warnings).
    statistic : float
        The test statistic.
    p_value : float or None
        Two-sided p-value for the test, or ``None`` when the
        underlying primitive does not provide one (currently
        :func:`anderson_darling`, whose SciPy backend reports only
        critical values, not a p-value).
    critical_value : float
        Critical value of `statistic` at `alpha`, obtained by table
        lookup, a distributional equation, or Monte Carlo simulation
        depending on the test — see `critical_value_source` and the
        specific test function's docstring.
    alpha : float
        Nominal significance level used for `critical_value` and
        `reject_null`.
    reject_null : bool
        Whether the null hypothesis of normality is rejected at
        `alpha` (statistic beyond `critical_value` in the test's
        rejection direction).
    critical_value_source : {"table_lookup", "equation", "asymptotic_equation", "monte_carlo"}
        How `critical_value` was obtained for this particular result.
    test_name : str
        Name of the test that produced this result (e.g.
        ``"shapiro_wilk"``).
    """

    statistic: float
    p_value: float | None
    critical_value: float
    alpha: float
    reject_null: bool
    critical_value_source: str
    test_name: str


def _base_or_none(x: NDArray[np.float64], sample_ids: NDArray[np.str_], method: str, min_n: int = 3):
    all_ids, report, (valid_x,) = valid_subset(sample_ids, x)
    meta = meta_for(all_ids=all_ids, report=report, method=method, min_n=min_n)
    if meta.status is not ResultStatus.VALID:
        return meta, None
    return meta, valid_x


def shapiro_wilk(
    x: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
    *,
    alpha: float = 0.05,
    n_null_resamples: int = 20_000,
    seed: int | None = None,
) -> NormalityTestResult:
    """Shapiro-Wilk test for normality.

    Wraps ``scipy.stats.shapiro``. The Shapiro-Wilk W statistic is
    small when the sample is inconsistent with normality, so the
    critical value is the lower-tail threshold below which W is
    deemed inconsistent with normality.

    Parameters
    ----------
    x : ndarray of float
        Sample values.
    sample_ids : ndarray of str
        Identifier for each row, used for provenance
        (``meta.included_ids``/``meta.excluded_ids``); non-finite
        values are excluded.
    alpha : float, optional
        Nominal significance level. Default 0.05.
    n_null_resamples : int, optional
        Number of Monte Carlo replicates of W simulated under H0 for
        this sample size, used to obtain `critical_value`. Default
        20000.
    seed : int or None, optional
        Seed for the Monte Carlo random number generator. Recorded in
        the result so the critical value is exactly reproducible.

    Returns
    -------
    NormalityTestResult
        The W statistic, its SciPy p-value, and a critical value with
        ``critical_value_source="monte_carlo"``. ``meta.status`` is
        ``ResultStatus.INSUFFICIENT_DATA`` if fewer than 3 finite
        observations remain, in which case the numeric fields are
        ``NaN``.

    Notes
    -----
    SciPy has no simple closed-form or bundled critical-value table
    for Shapiro-Wilk. Rather than transcribing an unverified table
    from memory, this function simulates `n_null_resamples`
    independent standard-normal samples of the same size as `x`,
    computes W for each, and takes `critical_value` to be the `alpha`
    quantile of that simulated null distribution:

    .. math::

        W_{\\alpha} = \\hat{Q}_{\\alpha}\\!\\left(\\{W_k : k = 1, \\ldots,
        n_{\\text{resamples}}\\}\\right)

    ``reject_null`` is ``True`` when the observed ``statistic <
    critical_value`` (the rejection direction is reversed relative to
    the other tests in this module, since small W indicates
    non-normality).

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.normality import shapiro_wilk
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(loc=5.0, scale=2.0, size=30)
    >>> ids = np.array([f"r{i}" for i in range(30)])
    >>> result = shapiro_wilk(x, ids, seed=0, n_null_resamples=500)
    >>> result.meta.status.value
    'valid'
    >>> result.critical_value_source
    'monte_carlo'
    >>> round(result.statistic, 3)
    0.975
    >>> result.reject_null
    False
    """

    meta, valid_x = _base_or_none(x, sample_ids, "shapiro_wilk")
    if valid_x is None:
        return NormalityTestResult(
            meta=meta, statistic=float("nan"), p_value=None, critical_value=float("nan"),
            alpha=alpha, reject_null=False, critical_value_source="monte_carlo", test_name="shapiro_wilk",
        )

    n = valid_x.shape[0]
    result = _stats.shapiro(valid_x)

    rng = np.random.default_rng(seed)
    null_w = np.empty(n_null_resamples)
    for i in range(n_null_resamples):
        null_w[i] = _stats.shapiro(rng.normal(size=n)).statistic
    # W is small under non-normality, so the critical value is the alpha
    # quantile of the null distribution (reject when statistic < critical).
    critical_value = float(np.quantile(null_w, alpha))

    meta = build_meta(
        status=ResultStatus.VALID,
        all_ids=meta.included_ids + meta.excluded_ids,
        included_ids=meta.included_ids,
        exclusion_reasons=meta.exclusion_reasons,
        method="shapiro_wilk",
        configuration={
            "backend": "scipy.stats.shapiro",
            "critical_value_source": "monte_carlo",
            "n_null_resamples": n_null_resamples,
            "alpha": alpha,
        },
        random_seed=seed,
    )
    return NormalityTestResult(
        meta=meta, statistic=float(result.statistic), p_value=float(result.pvalue),
        critical_value=critical_value, alpha=alpha, reject_null=bool(result.statistic < critical_value),
        critical_value_source="monte_carlo", test_name="shapiro_wilk",
    )


def kolmogorov_smirnov(
    x: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
    *,
    mean: float,
    std: float,
    alpha: float = 0.05,
) -> NormalityTestResult:
    """Classical one-sample Kolmogorov-Smirnov test for normality.

    Tests ``x`` against a **fully specified** N(`mean`, `std`) whose
    parameters are given by the caller, not estimated from ``x`` — use
    :func:`lilliefors` instead when the parameters are estimated from
    the same sample, since the classical KS critical values are then
    invalid (Lilliefors' correction exists precisely for that case).

    Parameters
    ----------
    x : ndarray of float
        Sample values.
    sample_ids : ndarray of str
        Identifier for each row, used for provenance; non-finite
        values are excluded.
    mean : float
        Mean of the fully specified null normal distribution.
    std : float
        Standard deviation of the fully specified null normal
        distribution. Must be positive.
    alpha : float, optional
        Nominal significance level. Default 0.05.

    Returns
    -------
    NormalityTestResult
        The KS D statistic, ``scipy.stats.kstest``'s p-value, and a
        critical value with
        ``critical_value_source="asymptotic_equation"``.
        ``meta.status`` is ``ResultStatus.INVALID_INPUT`` if ``std <=
        0``, or ``ResultStatus.INSUFFICIENT_DATA`` if fewer than 3
        finite observations remain; in both cases the numeric fields
        are ``NaN``.

    Notes
    -----
    The critical value is computed directly from the asymptotic
    Kolmogorov distribution (``scipy.stats.kstwobign``), not looked up
    from a table or simulated:

    .. math::

        D_{\\alpha} \\approx \\frac{K_{1-\\alpha}}{\\sqrt{n}}, \\qquad
        K_{1-\\alpha} = \\texttt{kstwobign.ppf}(1 - \\alpha)

    ``reject_null`` is ``True`` when the observed statistic exceeds
    :math:`D_{\\alpha}`.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.normality import kolmogorov_smirnov
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(loc=5.0, scale=2.0, size=30)
    >>> ids = np.array([f"r{i}" for i in range(30)])
    >>> result = kolmogorov_smirnov(x, ids, mean=5.0, std=2.0)
    >>> result.meta.status.value
    'valid'
    >>> result.critical_value_source
    'asymptotic_equation'
    >>> round(result.statistic, 3)
    0.14
    >>> result.reject_null
    False
    """

    meta, valid_x = _base_or_none(x, sample_ids, "kolmogorov_smirnov")
    if valid_x is None or std <= 0:
        status = ResultStatus.INVALID_INPUT if std <= 0 else meta.status
        meta = build_meta(
            status=status, all_ids=meta.included_ids + meta.excluded_ids,
            included_ids=() if status is ResultStatus.INVALID_INPUT else meta.included_ids,
            method="kolmogorov_smirnov",
        )
        return NormalityTestResult(
            meta=meta, statistic=float("nan"), p_value=None, critical_value=float("nan"),
            alpha=alpha, reject_null=False, critical_value_source="asymptotic_equation", test_name="kolmogorov_smirnov",
        )

    n = valid_x.shape[0]
    result = _stats.kstest(valid_x, "norm", args=(mean, std))
    critical_value = float(_stats.kstwobign.ppf(1 - alpha) / np.sqrt(n))

    meta = build_meta(
        status=ResultStatus.VALID,
        all_ids=meta.included_ids + meta.excluded_ids,
        included_ids=meta.included_ids,
        exclusion_reasons=meta.exclusion_reasons,
        method="kolmogorov_smirnov",
        configuration={
            "backend": "scipy.stats.kstest",
            "null_mean": mean,
            "null_std": std,
            "critical_value_source": "asymptotic_equation (scipy.stats.kstwobign)",
            "alpha": alpha,
        },
    )
    return NormalityTestResult(
        meta=meta, statistic=float(result.statistic), p_value=float(result.pvalue),
        critical_value=critical_value, alpha=alpha, reject_null=bool(result.statistic > critical_value),
        critical_value_source="asymptotic_equation", test_name="kolmogorov_smirnov",
    )


def lilliefors(
    x: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
    *,
    alpha: float = 0.05,
) -> NormalityTestResult:
    """Lilliefors test for normality (KS with estimated parameters).

    A Kolmogorov-Smirnov-type test where the reference normal
    distribution's mean and standard deviation are estimated from
    ``x`` itself (Lilliefors 1967), wrapping
    ``statsmodels.stats.diagnostic.lilliefors``. Use this instead of
    :func:`kolmogorov_smirnov` whenever the null distribution's
    parameters come from the same sample being tested.

    Parameters
    ----------
    x : ndarray of float
        Sample values.
    sample_ids : ndarray of str
        Identifier for each row, used for provenance; non-finite
        values are excluded.
    alpha : float, optional
        Nominal significance level. Default 0.05.

    Returns
    -------
    NormalityTestResult
        The KS-type D statistic, Statsmodels' table-based p-value, and
        a critical value with ``critical_value_source="table_lookup"``.
        ``meta.status`` is ``ResultStatus.INSUFFICIENT_DATA`` if fewer
        than 3 finite observations remain, in which case the numeric
        fields are ``NaN``.

    Notes
    -----
    The critical value is looked up from Statsmodels' own
    pre-computed critical-value table
    (``statsmodels.stats._lilliefors.get_lilliefors_table``, built from
    a 10-million-replicate simulation under H0) via 2D interpolation
    over ``(alpha, n)`` — it is not re-simulated or re-derived here.
    ``reject_null`` is ``True`` when the observed statistic exceeds
    that critical value.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.normality import lilliefors
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(loc=5.0, scale=2.0, size=30)
    >>> ids = np.array([f"r{i}" for i in range(30)])
    >>> result = lilliefors(x, ids)
    >>> result.meta.status.value
    'valid'
    >>> result.critical_value_source
    'table_lookup'
    >>> round(result.statistic, 3)
    0.092
    >>> result.reject_null
    False
    """

    from statsmodels.stats.diagnostic import lilliefors as _sm_lilliefors
    from statsmodels.stats._lilliefors import get_lilliefors_table

    meta, valid_x = _base_or_none(x, sample_ids, "lilliefors")
    if valid_x is None:
        return NormalityTestResult(
            meta=meta, statistic=float("nan"), p_value=None, critical_value=float("nan"),
            alpha=alpha, reject_null=False, critical_value_source="table_lookup", test_name="lilliefors",
        )

    n = valid_x.shape[0]
    ksstat, pvalue = _sm_lilliefors(valid_x, dist="norm", pvalmethod="table")
    table = get_lilliefors_table("norm")
    critical_value = float(table.crit(np.asarray([alpha]), n)[0])

    meta = build_meta(
        status=ResultStatus.VALID,
        all_ids=meta.included_ids + meta.excluded_ids,
        included_ids=meta.included_ids,
        exclusion_reasons=meta.exclusion_reasons,
        method="lilliefors",
        configuration={
            "backend": "statsmodels.stats.diagnostic.lilliefors",
            "critical_value_source": "statsmodels _lilliefors table (10M-replicate simulation)",
            "alpha": alpha,
        },
    )
    return NormalityTestResult(
        meta=meta, statistic=float(ksstat), p_value=float(pvalue),
        critical_value=critical_value, alpha=alpha, reject_null=bool(ksstat > critical_value),
        critical_value_source="table_lookup", test_name="lilliefors",
    )


def anderson_darling(
    x: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
    *,
    alpha: Literal[0.15, 0.10, 0.05, 0.025, 0.01] = 0.05,
) -> NormalityTestResult:
    """Anderson-Darling test for normality.

    Wraps ``scipy.stats.anderson(x, dist="norm")``, which already
    returns Stephens (1974) adjusted critical values directly for a
    fixed set of significance levels.

    Parameters
    ----------
    x : ndarray of float
        Sample values.
    sample_ids : ndarray of str
        Identifier for each row, used for provenance; non-finite
        values are excluded.
    alpha : {0.15, 0.10, 0.05, 0.025, 0.01}, optional
        Nominal significance level. Must be one of the levels SciPy's
        Anderson-Darling implementation tabulates. Default 0.05.

    Returns
    -------
    NormalityTestResult
        The Anderson-Darling A² statistic and a critical value with
        ``critical_value_source="table_lookup"``. ``p_value`` is
        always ``None`` (SciPy's backend does not provide one for this
        test). ``meta.status`` is ``ResultStatus.INSUFFICIENT_DATA`` if
        fewer than 3 finite observations remain, in which case the
        numeric fields are ``NaN``.

    Raises
    ------
    ValueError
        If `alpha` is not one of the significance levels SciPy
        tabulates (0.15, 0.10, 0.05, 0.025, 0.01).

    Notes
    -----
    The critical value is looked up directly from
    ``result.critical_values`` on the object returned by
    ``scipy.stats.anderson`` (Stephens 1974 adjusted critical values)
    — used as-is, with no re-derivation. ``reject_null`` is ``True``
    when the observed statistic exceeds that critical value.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.normality import anderson_darling
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(loc=5.0, scale=2.0, size=30)
    >>> ids = np.array([f"r{i}" for i in range(30)])
    >>> result = anderson_darling(x, ids)
    >>> result.meta.status.value
    'valid'
    >>> result.critical_value_source
    'table_lookup'
    >>> result.p_value is None
    True
    >>> round(result.statistic, 3)
    0.224
    >>> result.reject_null
    False
    """

    meta, valid_x = _base_or_none(x, sample_ids, "anderson_darling")
    if valid_x is None:
        return NormalityTestResult(
            meta=meta, statistic=float("nan"), p_value=None, critical_value=float("nan"),
            alpha=alpha, reject_null=False, critical_value_source="table_lookup", test_name="anderson_darling",
        )

    pct = alpha * 100.0
    idx = [i for i, level in enumerate(_ANDERSON_LEVELS_PCT) if np.isclose(level, pct)]
    if not idx:
        raise ValueError(
            f"alpha must be one of {tuple(l / 100 for l in _ANDERSON_LEVELS_PCT)} "
            f"(SciPy's tabulated Anderson-Darling significance levels), got {alpha}"
        )

    result = _stats.anderson(valid_x, dist="norm")
    critical_value = float(result.critical_values[idx[0]])

    meta = build_meta(
        status=ResultStatus.VALID,
        all_ids=meta.included_ids + meta.excluded_ids,
        included_ids=meta.included_ids,
        exclusion_reasons=meta.exclusion_reasons,
        method="anderson_darling",
        configuration={
            "backend": "scipy.stats.anderson",
            "critical_value_source": "scipy Stephens (1974) adjusted critical-value table",
            "alpha": alpha,
        },
    )
    return NormalityTestResult(
        meta=meta, statistic=float(result.statistic), p_value=None,
        critical_value=critical_value, alpha=alpha, reject_null=bool(result.statistic > critical_value),
        critical_value_source="table_lookup", test_name="anderson_darling",
    )


def dagostino_pearson(
    x: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
    *,
    alpha: float = 0.05,
) -> NormalityTestResult:
    """D'Agostino-Pearson K² omnibus test for normality.

    Combines skewness and kurtosis into a single omnibus test
    statistic K², wrapping ``scipy.stats.normaltest``.

    Parameters
    ----------
    x : ndarray of float
        Sample values.
    sample_ids : ndarray of str
        Identifier for each row, used for provenance; non-finite
        values are excluded.
    alpha : float, optional
        Nominal significance level. Default 0.05.

    Returns
    -------
    NormalityTestResult
        The K² statistic, ``scipy.stats.normaltest``'s p-value, and a
        critical value with ``critical_value_source="equation"``.
        ``meta.status`` is ``ResultStatus.INSUFFICIENT_DATA`` if fewer
        than 8 finite observations remain (the test requires enough
        data for both skewness and kurtosis to be meaningfully
        estimated), in which case the numeric fields are ``NaN``.

    Notes
    -----
    K² is chi-squared with 2 degrees of freedom under the null
    hypothesis of normality, so its critical value is computed exactly
    from that equation rather than looked up from a table or
    simulated:

    .. math::

        K^2_{\\alpha} = \\texttt{chi2.ppf}(1 - \\alpha,\\; df=2)

    ``reject_null`` is ``True`` when the observed statistic exceeds
    :math:`K^2_{\\alpha}`.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.normality import dagostino_pearson
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(loc=5.0, scale=2.0, size=30)
    >>> ids = np.array([f"r{i}" for i in range(30)])
    >>> result = dagostino_pearson(x, ids)
    >>> result.meta.status.value
    'valid'
    >>> result.critical_value_source
    'equation'
    >>> round(result.statistic, 3)
    1.176
    >>> round(result.critical_value, 3)
    5.991
    >>> result.reject_null
    False
    """

    meta, valid_x = _base_or_none(x, sample_ids, "dagostino_pearson", min_n=8)
    if valid_x is None:
        return NormalityTestResult(
            meta=meta, statistic=float("nan"), p_value=None, critical_value=float("nan"),
            alpha=alpha, reject_null=False, critical_value_source="equation", test_name="dagostino_pearson",
        )

    result = _stats.normaltest(valid_x)
    critical_value = float(_stats.chi2.ppf(1 - alpha, df=2))

    meta = build_meta(
        status=ResultStatus.VALID,
        all_ids=meta.included_ids + meta.excluded_ids,
        included_ids=meta.included_ids,
        exclusion_reasons=meta.exclusion_reasons,
        method="dagostino_pearson",
        configuration={
            "backend": "scipy.stats.normaltest",
            "critical_value_source": "chi2.ppf(1-alpha, df=2)",
            "alpha": alpha,
        },
    )
    return NormalityTestResult(
        meta=meta, statistic=float(result.statistic), p_value=float(result.pvalue),
        critical_value=critical_value, alpha=alpha, reject_null=bool(result.statistic > critical_value),
        critical_value_source="equation", test_name="dagostino_pearson",
    )


def normality_test_suite(
    x: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
    *,
    alpha: float = 0.05,
    seed: int | None = None,
) -> dict[str, NormalityTestResult]:
    """Run all five normality tests on the same sample.

    Convenience wrapper around :func:`shapiro_wilk`,
    :func:`kolmogorov_smirnov`, :func:`lilliefors`,
    :func:`anderson_darling`, and :func:`dagostino_pearson`.

    Parameters
    ----------
    x : ndarray of float
        Sample values.
    sample_ids : ndarray of str
        Identifier for each row, used for provenance; non-finite
        values are excluded.
    alpha : float, optional
        Nominal significance level passed to every test. Default 0.05.
        (:func:`anderson_darling` requires one of its tabulated
        levels; using the default 0.05 always satisfies this.)
    seed : int or None, optional
        Seed passed through to :func:`shapiro_wilk`'s Monte Carlo
        critical-value simulation. Recorded in that sub-result so it
        is exactly reproducible.

    Returns
    -------
    dict of str to NormalityTestResult
        Keyed by test name: ``"shapiro_wilk"``, ``"kolmogorov_smirnov"``,
        ``"lilliefors"``, ``"anderson_darling"``, ``"dagostino_pearson"``.

    Notes
    -----
    ``kolmogorov_smirnov`` uses the sample's own mean and standard
    deviation as the "fully specified" null purely for this convenience
    wrapper. This is a self-referential null, not a genuine external
    specification; for the classical KS test in its intended sense
    (parameters known independently of ``x``), call
    :func:`kolmogorov_smirnov` directly with the true ``mean``/``std``,
    or prefer :func:`lilliefors` for an estimated-parameter test whose
    critical values already account for the parameter estimation.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.normality import normality_test_suite
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(loc=5.0, scale=2.0, size=30)
    >>> ids = np.array([f"r{i}" for i in range(30)])
    >>> results = normality_test_suite(x, ids, seed=0)
    >>> sorted(results.keys())
    ['anderson_darling', 'dagostino_pearson', 'kolmogorov_smirnov', 'lilliefors', 'shapiro_wilk']
    >>> all(r.meta.status.value == 'valid' for r in results.values())
    True
    >>> any(r.reject_null for r in results.values())
    False
    """

    all_ids, report, (valid_x,) = valid_subset(sample_ids, x)
    mean = float(np.mean(valid_x)) if valid_x.size else float("nan")
    std = float(np.std(valid_x, ddof=1)) if valid_x.size > 1 else float("nan")

    return {
        "shapiro_wilk": shapiro_wilk(x, sample_ids, alpha=alpha, seed=seed),
        "kolmogorov_smirnov": kolmogorov_smirnov(x, sample_ids, mean=mean, std=std, alpha=alpha),
        "lilliefors": lilliefors(x, sample_ids, alpha=alpha),
        "anderson_darling": anderson_darling(x, sample_ids, alpha=alpha),
        "dagostino_pearson": dagostino_pearson(x, sample_ids, alpha=alpha),
    }
