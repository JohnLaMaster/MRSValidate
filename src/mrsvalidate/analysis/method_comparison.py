"""Method comparison: Bland-Altman, Passing-Bablok, and Deming regression.

Implements spec eqs. 56-64 (manuscript ``ba_loa_heteroscedastic``,
``pb_slopes``/``pb_slope``/``pb_intercept``,
``deming_objective``/``deming_slope``/``deming_intercept``/
``deming_variances``/``deming_lambda``) plus the Silveira et al.
three-part equivalence-testing protocol built on top of them.

Notes
-----
Convention throughout, matching the manuscript's method-comparison
sections: ``reference`` is the established/ground-truth method,
``candidate``/``test`` is the DL model or method under evaluation, and
differences are ``d_n = candidate_n - reference_n``. Each function's own
docstring restates which array plays which regression role, since that
assignment (independent/dependent, or ``x``/``y``) differs by method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from scipy import stats as _stats

from .._numerics import iqr as _iqr, robust_sd as _robust_sd
from ..results import AnalysisResult, ResultStatus, build_meta

__all__ = [
    "bland_altman",
    "passing_bablok",
    "deming",
    "equivalence_test_1",
    "equivalence_test_2",
    "equivalence_test_3",
    "BlandAltmanResult",
    "PassingBablokResult",
    "DemingResult",
    "EquivalenceTest1Result",
    "EquivalenceTest2Result",
    "EquivalenceTest3Result",
]

BAAxis = Literal["reference", "mean"]
BALocation = Literal["mean", "median"]
BAScale = Literal["sd", "robust_sd"]

_Z95 = 1.96


def _finite_pairs(
    reference: NDArray[np.float64], candidate: NDArray[np.float64], sample_ids: NDArray[np.str_]
):
    all_ids = tuple(np.asarray(sample_ids, dtype=str).tolist())
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    valid = np.isfinite(ref) & np.isfinite(cand)
    included_ids = tuple(np.asarray(all_ids)[valid].tolist())
    return all_ids, included_ids, ref[valid], cand[valid]


@dataclass(frozen=True)
class BlandAltmanResult(AnalysisResult):
    """Result of a :func:`bland_altman` agreement analysis.

    Attributes
    ----------
    meta : AnalysisMeta
        Provenance and inclusion/exclusion record for this computation.
    bias : float
        Location of the differences ``d_n = candidate_n - reference_n``:
        their mean or median, per `location`. NaN if the result is not
        ``VALID``.
    scale : float
        Dispersion of the differences: the sample standard deviation or
        the robust SD, per `scale_method`.
    loa_low : float
        Lower 95% limit of agreement, ``bias - 1.96 * scale``.
    loa_high : float
        Upper 95% limit of agreement, ``bias + 1.96 * scale``.
    iqr_of_differences : float
        Interquartile range of the differences, a distribution-free
        companion to the (Gaussian-assuming) limits of agreement.
    x_axis : {"reference", "mean"}
        Which quantity the differences were assessed against; see
        `bland_altman`.
    location : {"mean", "median"}
        Which location estimator was used for `bias`.
    scale_method : {"sd", "robust_sd"}
        Which dispersion estimator was used for `scale`.
    conditional : bool
        Whether heteroscedastic location/scale models were additionally
        fit (see `bland_altman`).
    differences : ndarray of float
        Per-included-sample differences ``d_n = candidate_n - reference_n``.
    x_values : ndarray of float, same shape as `differences`
        Per-included-sample x-axis values (`reference` or the pairwise
        mean, per `x_axis`) that `differences` is plotted/summarized
        against.
    conditional_bias_coefficients : tuple of (float, float), or None
        ``(b0, b1)`` intercept and slope of the fitted
        ``bias(c) = b0 + b1 * c`` model, or None when `conditional` is
        False.
    conditional_scale_coefficients : tuple of (float, float), or None
        ``(s0, s1)`` intercept and slope of the fitted, Gaussian-consistency
        corrected ``SD(c) = s0 + s1 * c`` model, or None when
        `conditional` is False.
    """

    bias: float
    scale: float
    loa_low: float
    loa_high: float
    iqr_of_differences: float
    x_axis: BAAxis
    location: BALocation
    scale_method: BAScale
    conditional: bool
    differences: NDArray[np.float64]
    x_values: NDArray[np.float64]
    conditional_bias_coefficients: tuple[float, float] | None
    conditional_scale_coefficients: tuple[float, float] | None


def bland_altman(
    reference: NDArray[np.float64],
    candidate: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
    *,
    x_axis: BAAxis = "reference",
    location: BALocation = "mean",
    scale: BAScale = "sd",
    conditional: bool = False,
) -> BlandAltmanResult:
    r"""Compute a Bland-Altman agreement analysis between two methods.

    Summarizes the paired differences ``d_n = candidate_n - reference_n``
    by a location (bias) and dispersion (limits of agreement), optionally
    conditioned on the value of the plotting axis.

    Parameters
    ----------
    reference : ndarray of float, shape (n_samples,)
        Values from the established/ground-truth method.
    candidate : ndarray of float, shape (n_samples,)
        Values from the model or method under evaluation. Differences are
        computed as ``d_n = candidate_n - reference_n``.
    sample_ids : ndarray of str, shape (n_samples,)
        Per-sample identifiers, used only for the inclusion/exclusion
        bookkeeping recorded in the result's ``meta``.
    x_axis : {"reference", "mean"}, default "reference"
        What the differences are plotted/summarized against.
        ``"reference"`` (the default) follows the manuscript's *in
        silico*/Krouwer convention: differences are plotted against the
        error-free reference, which is well defined whenever a
        known-true reference exists. ``"mean"`` is the conventional
        Bland-Altman axis, the pairwise mean ``(reference + candidate) /
        2``, appropriate when both methods carry error.
    location : {"mean", "median"}, default "mean"
        Location estimator applied to the differences to obtain `bias`.
    scale : {"sd", "robust_sd"}, default "sd"
        Dispersion estimator applied to the differences: the sample
        standard deviation (``ddof=1``), or the robust SD (median
        absolute deviation from the median, not rescaled to
        normal-consistency; see :func:`mrsvalidate._numerics.robust_sd`).
    conditional : bool, default False
        If True, additionally fit transparent linear location/scale
        models of the differences against the chosen `x_axis` instead of
        relying solely on the constant-variance limits of agreement.

    Returns
    -------
    BlandAltmanResult
        Bias, scale, limits of agreement, and (if requested) the
        conditional location/scale model coefficients. See
        :class:`BlandAltmanResult` for the full field list.

    Notes
    -----
    When `conditional` is True, two ordinary least-squares fits are made
    against the chosen axis value :math:`c`:

    .. math::

        \mathrm{bias}(c) = b_0 + b_1 c

    for the location, and

    .. math::

        \mathrm{SD}(c) = s_0 + s_1 c

    for the scale, where the slope/intercept :math:`(s_0, s_1)` come from
    an ordinary least-squares regression of :math:`|d - \widehat{\mathrm{bias}}(c)|`
    on :math:`c`, rescaled by :math:`\sqrt{\pi/2}` (the Gaussian
    consistency correction for a mean absolute deviation) so that `s0`,
    `s1` are directly comparable to a standard deviation. These replace
    the single constant-variance limits of agreement with
    heteroscedastic, axis-dependent ones.

    Samples with a non-finite `reference` or `candidate` value are
    excluded before any statistic is computed; fewer than 2 valid pairs
    yields a NaN-filled result with
    ``meta.status = ResultStatus.INSUFFICIENT_DATA``.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.method_comparison import bland_altman
    >>> reference = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
    >>> candidate = np.array([10.6, 12.3, 13.7, 16.5, 18.1, 20.9])
    >>> ids = np.array(["s1", "s2", "s3", "s4", "s5", "s6"])
    >>> result = bland_altman(reference, candidate, ids)
    >>> print(f"{result.bias:.4f} {result.scale:.4f}")
    0.3500 0.4183
    >>> print(f"{result.loa_low:.4f} {result.loa_high:.4f}")
    -0.4699 1.1699

    Conditional (heteroscedastic) location/scale models against the
    reference axis:

    >>> cond = bland_altman(reference, candidate, ids, conditional=True)
    >>> b0, b1 = cond.conditional_bias_coefficients
    >>> print(f"{b0:.4f} {b1:.4f}")
    -0.0143 0.0243
    """

    all_ids, included_ids, ref, cand = _finite_pairs(reference, candidate, sample_ids)
    status = ResultStatus.VALID if len(ref) >= 2 else ResultStatus.INSUFFICIENT_DATA
    n_invalid = len(all_ids) - len(included_ids)

    if status is not ResultStatus.VALID:
        meta = build_meta(
            status=status,
            all_ids=all_ids,
            included_ids=included_ids,
            exclusion_reasons={"nan_or_inf": n_invalid} if n_invalid else {},
            method="bland_altman",
        )
        empty = np.asarray([], dtype=np.float64)
        return BlandAltmanResult(
            meta=meta,
            bias=float("nan"),
            scale=float("nan"),
            loa_low=float("nan"),
            loa_high=float("nan"),
            iqr_of_differences=float("nan"),
            x_axis=x_axis,
            location=location,
            scale_method=scale,
            conditional=conditional,
            differences=empty,
            x_values=empty,
            conditional_bias_coefficients=None,
            conditional_scale_coefficients=None,
        )

    d = cand - ref
    x_values = ref if x_axis == "reference" else (ref + cand) / 2.0

    bias = float(np.mean(d)) if location == "mean" else float(np.median(d))
    disp = float(np.std(d, ddof=1)) if scale == "sd" else _robust_sd(d)

    conditional_bias_coef = None
    conditional_scale_coef = None
    if conditional:
        b1, b0 = np.polyfit(x_values, d, 1)
        fitted_bias = b0 + b1 * x_values
        abs_resid = np.abs(d - fitted_bias)
        s1, s0 = np.polyfit(x_values, abs_resid, 1)
        gaussian_correction = float(np.sqrt(np.pi / 2.0))
        conditional_bias_coef = (float(b0), float(b1))
        conditional_scale_coef = (float(s0 * gaussian_correction), float(s1 * gaussian_correction))

    meta = build_meta(
        status=ResultStatus.VALID,
        all_ids=all_ids,
        included_ids=included_ids,
        exclusion_reasons={"nan_or_inf": n_invalid} if n_invalid else {},
        method="bland_altman",
        configuration={
            "x_axis": x_axis,
            "location": location,
            "scale_method": scale,
            "conditional": conditional,
            "loa_z": _Z95,
        },
    )
    return BlandAltmanResult(
        meta=meta,
        bias=bias,
        scale=disp,
        loa_low=bias - _Z95 * disp,
        loa_high=bias + _Z95 * disp,
        iqr_of_differences=_iqr(d),
        x_axis=x_axis,
        location=location,
        scale_method=scale,
        conditional=conditional,
        differences=d,
        x_values=x_values,
        conditional_bias_coefficients=conditional_bias_coef,
        conditional_scale_coefficients=conditional_scale_coef,
    )


@dataclass(frozen=True)
class PassingBablokResult(AnalysisResult):
    """Result of a :func:`passing_bablok` regression.

    Attributes
    ----------
    meta : AnalysisMeta
        Provenance and inclusion/exclusion record for this computation.
    slope : float
        Passing-Bablok slope estimate :math:`\\hat{\\beta}_1`: `candidate`
        modeled as a function of `reference`.
    intercept : float
        Passing-Bablok intercept estimate :math:`\\hat{\\beta}_0`.
    slope_ci : tuple of (float, float)
        Percentile bootstrap confidence interval for the slope, at
        `confidence_level`.
    intercept_ci : tuple of (float, float)
        Percentile bootstrap confidence interval for the intercept, at
        `confidence_level`.
    n_pairwise_slopes : int
        Number of pairwise slopes :math:`(x_j - x_i)/(y_j - y_i)` that
        entered the estimate, after excluding pairs with equal reference
        values.
    n_excluded_equal_reference_pairs : int
        Number of sample pairs excluded because their reference values
        were exactly equal (an undefined pairwise slope denominator).
    cusum_max_abs : float
        Observed maximum absolute cumulative sum of standardized,
        reference-ordered residuals; the linearity diagnostic statistic.
    cusum_p_value : float
        Permutation p-value for `cusum_max_abs` (not the classical
        tabulated Passing-Bablok CUSUM test).
    confidence_level : float
        Nominal confidence level used for `slope_ci` and `intercept_ci`.
    ci_method : str
        Name of the confidence-interval method actually used,
        ``"bootstrap_percentile"``.
    """

    slope: float
    intercept: float
    slope_ci: tuple[float, float]
    intercept_ci: tuple[float, float]
    n_pairwise_slopes: int
    n_excluded_equal_reference_pairs: int
    cusum_max_abs: float
    cusum_p_value: float
    confidence_level: float
    ci_method: str


def _pb_estimate(ref: NDArray[np.float64], cand: NDArray[np.float64]) -> tuple[float, float, int, int]:
    n = ref.shape[0]
    slopes = []
    for i in range(n - 1):
        dy = ref[i + 1 :] - ref[i]
        dx = cand[i + 1 :] - cand[i]
        mask = dy != 0
        slopes.append(dx[mask] / dy[mask])
    all_slopes = np.concatenate(slopes) if slopes else np.asarray([], dtype=np.float64)
    m = all_slopes.shape[0]
    n_excluded_pairs = (n * (n - 1) // 2) - m

    sorted_slopes = np.sort(all_slopes)
    k = int(np.sum(sorted_slopes < -1))
    idx_1indexed = m // 2 + k + 1
    if idx_1indexed > m:
        # Standard PB wrap-around: the shifted index is defined modulo M
        # (Passing & Bablok 1983); without this, a heavily negative-slope
        # sample (K close to M) would index past the end of the array.
        idx_1indexed -= m
    idx = min(max(idx_1indexed - 1, 0), m - 1)
    beta1 = float(sorted_slopes[idx])
    beta0 = float(np.median(cand - beta1 * ref))
    return beta1, beta0, m, n_excluded_pairs


def passing_bablok(
    reference: NDArray[np.float64],
    candidate: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
    *,
    confidence_level: float = 0.95,
    n_bootstrap: int = 2000,
    n_cusum_permutations: int = 2000,
    seed: int | None = None,
) -> PassingBablokResult:
    r"""Compute a Passing-Bablok regression of `candidate` on `reference`.

    Implements the pairwise-slope construction of manuscript eqs.
    ``pb_slopes``/``pb_slope``/``pb_intercept``: `candidate` is modeled
    as a function of `reference`, ``candidate = beta0 + beta1 *
    reference``, via the median of all valid pairwise slopes rather than
    least squares.

    Parameters
    ----------
    reference : ndarray of float, shape (n_samples,)
        Values from the established/ground-truth method; the
        independent variable of the fit.
    candidate : ndarray of float, shape (n_samples,)
        Values from the model or method under evaluation; the dependent
        variable of the fit.
    sample_ids : ndarray of str, shape (n_samples,)
        Per-sample identifiers, used for inclusion/exclusion bookkeeping.
    confidence_level : float, default 0.95
        Nominal two-sided confidence level for `slope_ci`/`intercept_ci`.
    n_bootstrap : int, default 2000
        Number of paired bootstrap resamples used to build the
        confidence intervals.
    n_cusum_permutations : int, default 2000
        Number of permutations used for the CUSUM linearity-diagnostic
        p-value.
    seed : int or None, default None
        Seed for the random number generator driving the bootstrap and
        permutation resampling; None draws fresh entropy.

    Returns
    -------
    PassingBablokResult
        Slope, intercept, their bootstrap confidence intervals, and the
        CUSUM linearity diagnostic. See :class:`PassingBablokResult` for
        the full field list.

    Notes
    -----
    Confidence intervals use a paired percentile bootstrap
    (``ci_method="bootstrap_percentile"``) rather than the classical
    analytical Passing-Bablok confidence interval — the latter is not
    reproduced here, and this substitution is recorded explicitly in the
    result rather than silently approximated.

    The linearity diagnostic (`cusum_max_abs`) uses a permutation p-value
    on the residual CUSUM rather than the original tabulated critical
    values.

    Fewer than 3 valid pairs yields a NaN-filled result with
    ``meta.status = ResultStatus.INSUFFICIENT_DATA``.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.method_comparison import passing_bablok
    >>> reference = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
    >>> candidate = np.array([10.6, 12.3, 13.7, 16.5, 18.1, 20.9])
    >>> ids = np.array(["s1", "s2", "s3", "s4", "s5", "s6"])
    >>> result = passing_bablok(
    ...     reference, candidate, ids,
    ...     n_bootstrap=200, n_cusum_permutations=200, seed=0,
    ... )
    >>> print(f"{result.slope:.4f} {result.intercept:.4f}")
    1.0300 -0.0200
    >>> result.n_pairwise_slopes, result.n_excluded_equal_reference_pairs
    (15, 0)
    """

    all_ids, included_ids, ref, cand = _finite_pairs(reference, candidate, sample_ids)
    n = len(ref)
    n_invalid = len(all_ids) - len(included_ids)
    status = ResultStatus.VALID if n >= 3 else ResultStatus.INSUFFICIENT_DATA

    if status is not ResultStatus.VALID:
        meta = build_meta(
            status=status,
            all_ids=all_ids,
            included_ids=included_ids,
            exclusion_reasons={"nan_or_inf": n_invalid} if n_invalid else {},
            method="passing_bablok",
        )
        return PassingBablokResult(
            meta=meta,
            slope=float("nan"),
            intercept=float("nan"),
            slope_ci=(float("nan"), float("nan")),
            intercept_ci=(float("nan"), float("nan")),
            n_pairwise_slopes=0,
            n_excluded_equal_reference_pairs=0,
            cusum_max_abs=float("nan"),
            cusum_p_value=float("nan"),
            confidence_level=confidence_level,
            ci_method="bootstrap_percentile",
        )

    beta1, beta0, m, n_excluded_pairs = _pb_estimate(ref, cand)

    rng = np.random.default_rng(seed)
    boot_slopes = np.empty(n_bootstrap)
    boot_intercepts = np.empty(n_bootstrap)
    idx_range = np.arange(n)
    for b in range(n_bootstrap):
        idx = rng.choice(idx_range, size=n, replace=True)
        try:
            s, i0, _, _ = _pb_estimate(ref[idx], cand[idx])
        except (IndexError, ValueError):
            s, i0 = beta1, beta0
        boot_slopes[b] = s
        boot_intercepts[b] = i0

    alpha = 1.0 - confidence_level
    slope_ci = tuple(np.quantile(boot_slopes, [alpha / 2, 1 - alpha / 2]))
    intercept_ci = tuple(np.quantile(boot_intercepts, [alpha / 2, 1 - alpha / 2]))

    order = np.argsort(ref)
    resid = (cand - (beta0 + beta1 * ref))[order]
    resid_std = np.std(resid, ddof=1) if np.std(resid, ddof=1) > 0 else 1.0
    cusum = np.cumsum(resid) / resid_std
    observed_stat = float(np.max(np.abs(cusum)))

    perm_stats = np.empty(n_cusum_permutations)
    for p in range(n_cusum_permutations):
        perm_resid = rng.permutation(resid)
        perm_cusum = np.cumsum(perm_resid) / resid_std
        perm_stats[p] = np.max(np.abs(perm_cusum))
    cusum_p = float((1 + np.sum(perm_stats >= observed_stat)) / (n_cusum_permutations + 1))

    meta = build_meta(
        status=ResultStatus.VALID,
        all_ids=all_ids,
        included_ids=included_ids,
        exclusion_reasons={"nan_or_inf": n_invalid} if n_invalid else {},
        method="passing_bablok",
        configuration={
            "ci_method": "bootstrap_percentile",
            "n_bootstrap": n_bootstrap,
            "cusum_p_value_method": "permutation",
            "n_cusum_permutations": n_cusum_permutations,
        },
        random_seed=seed,
    )
    return PassingBablokResult(
        meta=meta,
        slope=beta1,
        intercept=beta0,
        slope_ci=slope_ci,
        intercept_ci=intercept_ci,
        n_pairwise_slopes=m,
        n_excluded_equal_reference_pairs=n_excluded_pairs,
        cusum_max_abs=observed_stat,
        cusum_p_value=cusum_p,
        confidence_level=confidence_level,
        ci_method="bootstrap_percentile",
    )


@dataclass(frozen=True)
class DemingResult(AnalysisResult):
    """Result of a :func:`deming` errors-in-variables regression.

    Attributes
    ----------
    meta : AnalysisMeta
        Provenance and inclusion/exclusion record for this computation.
    slope : float
        Deming slope estimate :math:`\\hat{\\beta}_1` (`test`/`candidate`
        modeled as a linear function of the latent variable that
        `reference` approximates; see `reference_axis`/`test_axis`).
    intercept : float
        Deming intercept estimate :math:`\\hat{\\beta}_0`.
    variance_ratio_value : float
        The error-variance ratio actually used,
        ``reference_error_variance / test_error_variance``, per
        `variance_ratio_convention`.
    variance_ratio_convention : str
        Human-readable statement of which ratio `variance_ratio_value`
        is, always ``"lambda = var(reference) / var(test)"``.
    reference_axis : str
        Which regression axis `reference` plays, always ``"y"``.
    test_axis : str
        Which regression axis `test` plays, always ``"x"``.
    """

    slope: float
    intercept: float
    variance_ratio_value: float
    variance_ratio_convention: str
    reference_axis: str
    test_axis: str


def deming(
    reference: NDArray[np.float64],
    test: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
    *,
    reference_error_variance: float,
    test_error_variance: float,
) -> DemingResult:
    r"""Compute a Deming (errors-in-variables) regression of `test` on `reference`.

    Implements manuscript eqs. ``deming_objective``/``deming_slope``/
    ``deming_intercept``/``deming_variances``/``deming_lambda``: unlike
    ordinary least squares, Deming regression accounts for error in both
    variables via their known (or assumed) error-variance ratio.

    Parameters
    ----------
    reference : ndarray of float, shape (n_samples,)
        Values from the established/ground-truth method. Plays the role
        of ``y`` in the regression (see Notes).
    test : ndarray of float, shape (n_samples,)
        Values from the model or method under evaluation, i.e. the
        candidate. Plays the role of ``x`` in the regression.
    sample_ids : ndarray of str, shape (n_samples,)
        Per-sample identifiers, used for inclusion/exclusion bookkeeping.
    reference_error_variance : float
        Assumed (or externally estimated) error variance of `reference`.
        Must be finite and strictly positive. There is no implicit
        default and no bare ``lambda`` parameter — both named variances
        must always be supplied explicitly.
    test_error_variance : float
        Assumed (or externally estimated) error variance of `test`. Must
        be finite and strictly positive.

    Returns
    -------
    DemingResult
        Slope, intercept, and the resolved error-variance ratio. See
        :class:`DemingResult` for the full field list.

    Raises
    ------
    (none; invalid variances or degenerate data are reported through
    ``meta.status`` rather than raised)

    Notes
    -----
    ``lambda = reference_error_variance / test_error_variance``, matching
    the manuscript's convention (``y`` = reference, ``x`` = test/candidate),
    and is reported as `DemingResult.variance_ratio_value`. Writing
    :math:`\mu = 1/\lambda`, and with :math:`S_{yy}`, :math:`S_{xx}`,
    :math:`S_{xy}` the sample variances/covariance of `reference` and
    `test`, the slope used here is

    .. math::

        \hat{\beta}_1 = \frac{S_{xx} - \mu S_{yy}
            + \sqrt{(S_{xx} - \mu S_{yy})^2 + 4 \mu S_{xy}^2}}{2 S_{xy}}

    and the intercept is :math:`\hat{\beta}_0 = \bar{x} - \hat{\beta}_1 \bar{y}`,
    i.e. :math:`\mathrm{test} - \bar{x} = \hat{\beta}_1 (\mathrm{reference} - \bar{y})`:
    `test` is modeled as a linear function of the latent variable that
    `reference` approximates.

    **On a manuscript-internal inconsistency.** The manuscript's
    *displayed* slope equation, read with its own :math:`S_{yy}`,
    :math:`S_{xx}`, :math:`\lambda` symbols taken at face value, does
    not correspond to the same regression direction as its own displayed
    intercept equation (:math:`\beta_0 = \bar{x} - \beta_1 \bar{y}`) and
    stated objective (``x`` modeled as a linear function of the latent
    variable that ``y`` approximates). Taking the two displayed equations
    literally and combining them produces a line that does not pass
    through noiseless, perfectly linear data — which a correct
    errors-in-variables fit must always do. This implementation uses the
    slope formula in the form that *is* self-consistent with the
    displayed intercept equation and the stated objective (verified
    against the noiseless closed-form solution: see Examples below),
    rather than the literal juxtaposition of both displayed equations.

    A zero sample covariance between `reference` and `test`, or a
    negative discriminant, leaves the slope undefined
    (``meta.status = ResultStatus.UNDEFINED``). Non-finite or
    non-positive `reference_error_variance`/`test_error_variance` yields
    ``meta.status = ResultStatus.INVALID_INPUT``. Fewer than 2 valid
    pairs yields ``meta.status = ResultStatus.INSUFFICIENT_DATA``.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.method_comparison import deming
    >>> reference = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
    >>> test = np.array([10.6, 12.3, 13.7, 16.5, 18.1, 20.9])
    >>> ids = np.array(["s1", "s2", "s3", "s4", "s5", "s6"])
    >>> result = deming(
    ...     reference, test, ids,
    ...     reference_error_variance=1.0, test_error_variance=1.0,
    ... )
    >>> print(f"{result.slope:.4f} {result.intercept:.4f}")
    1.0303 -0.1038

    Self-consistency check: noiseless, perfectly linear data must be
    reproduced exactly, regardless of the assumed variance ratio.

    >>> perfect_reference = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> perfect_test = 2.0 * perfect_reference + 3.0
    >>> exact = deming(
    ...     perfect_reference, perfect_test, np.array(["a", "b", "c", "d", "e"]),
    ...     reference_error_variance=1.0, test_error_variance=1.0,
    ... )
    >>> print(f"{exact.slope:.4f} {exact.intercept:.4f}")
    2.0000 3.0000
    """

    all_ids, included_ids, ref, test_vals = _finite_pairs(reference, test, sample_ids)
    n = len(ref)
    n_invalid = len(all_ids) - len(included_ids)
    status = ResultStatus.VALID if n >= 2 else ResultStatus.INSUFFICIENT_DATA

    if (
        status is ResultStatus.VALID
        and (
            not np.isfinite(reference_error_variance)
            or not np.isfinite(test_error_variance)
            or reference_error_variance <= 0
            or test_error_variance <= 0
        )
    ):
        status = ResultStatus.INVALID_INPUT

    if status is not ResultStatus.VALID:
        meta = build_meta(
            status=status,
            all_ids=all_ids,
            included_ids=included_ids if status is ResultStatus.VALID else (),
            exclusion_reasons={"nan_or_inf": n_invalid} if n_invalid else {},
            method="deming",
            warnings=()
            if status is not ResultStatus.INVALID_INPUT
            else ("reference_error_variance and test_error_variance must be finite and positive",),
        )
        return DemingResult(
            meta=meta,
            slope=float("nan"),
            intercept=float("nan"),
            variance_ratio_value=float("nan"),
            variance_ratio_convention="lambda = var(reference) / var(test)",
            reference_axis="y",
            test_axis="x",
        )

    lam = reference_error_variance / test_error_variance
    mu = 1.0 / lam  # test_error_variance / reference_error_variance
    y_bar, x_bar = np.mean(ref), np.mean(test_vals)
    s_yy = np.sum((ref - y_bar) ** 2) / (n - 1)
    s_xx = np.sum((test_vals - x_bar) ** 2) / (n - 1)
    s_xy = np.sum((ref - y_bar) * (test_vals - x_bar)) / (n - 1)

    discriminant = (s_xx - mu * s_yy) ** 2 + 4 * mu * s_xy**2
    if s_xy == 0 or discriminant < 0:
        meta = build_meta(
            status=ResultStatus.UNDEFINED,
            all_ids=all_ids,
            included_ids=included_ids,
            method="deming",
            warnings=("Deming slope undefined: zero covariance between reference and test",),
        )
        return DemingResult(
            meta=meta,
            slope=float("nan"),
            intercept=float("nan"),
            variance_ratio_value=lam,
            variance_ratio_convention="lambda = var(reference) / var(test)",
            reference_axis="y",
            test_axis="x",
        )

    slope = (s_xx - mu * s_yy + np.sqrt(discriminant)) / (2 * s_xy)
    intercept = x_bar - slope * y_bar

    meta = build_meta(
        status=ResultStatus.VALID,
        all_ids=all_ids,
        included_ids=included_ids,
        exclusion_reasons={"nan_or_inf": n_invalid} if n_invalid else {},
        method="deming",
        configuration={
            "reference_error_variance": reference_error_variance,
            "test_error_variance": test_error_variance,
            "variance_ratio_convention": "lambda = var(reference) / var(test)",
        },
    )
    return DemingResult(
        meta=meta,
        slope=float(slope),
        intercept=float(intercept),
        variance_ratio_value=float(lam),
        variance_ratio_convention="lambda = var(reference) / var(test)",
        reference_axis="y",
        test_axis="x",
    )


# ---------------------------------------------------------------------------
# Silveira et al. three-part equivalence protocol (per-metabolite), spec eqs.
# 70-72 / manuscript "Table Silveira_table" and Sec. "Equivalence Testing".
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EquivalenceTest1Result(AnalysisResult):
    """Result of :func:`equivalence_test_1` (accuracy).

    Fits ``x - y = alpha + beta * (y - ybar) + nu`` and tests the null
    hypothesis H0: ``alpha = 0``.

    Attributes
    ----------
    meta : AnalysisMeta
        Provenance and inclusion/exclusion record for this computation.
    alpha : float
        Fitted intercept; the accuracy (translation) bias term under
        test.
    beta : float
        Fitted slope on the mean-centered reference.
    alpha_se : float
        Standard error of `alpha`.
    alpha_t : float
        t-statistic for H0: ``alpha = 0``, ``alpha / alpha_se``.
    alpha_p_value : float
        Two-sided p-value for `alpha_t` against a Student's t
        distribution with ``n - 2`` degrees of freedom.
    """

    alpha: float
    beta: float
    alpha_se: float
    alpha_t: float
    alpha_p_value: float


@dataclass(frozen=True)
class EquivalenceTest2Result(AnalysisResult):
    """Result of :func:`equivalence_test_2` (precision, Shukla 1973).

    Fits ``x - y = alpha + beta * (x + y) + theta`` and tests the null
    hypothesis H0: ``beta = 0``, which is equivalent to a unit
    reference/candidate variance ratio (``lambda = 1``).

    Attributes
    ----------
    meta : AnalysisMeta
        Provenance and inclusion/exclusion record for this computation.
    alpha : float
        Fitted intercept.
    beta : float
        Fitted slope on ``candidate + reference``; the precision term
        under test.
    beta_se : float
        Standard error of `beta`.
    beta_t : float
        t-statistic for H0: ``beta = 0``, ``beta / beta_se``.
    beta_p_value : float
        Two-sided p-value for `beta_t` against a Student's t
        distribution with ``n - 2`` degrees of freedom.
    bias_correction_applied : float or None
        The `bias_correction` value that was subtracted from `candidate`
        before fitting, or None if none was applied.
    """

    alpha: float
    beta: float
    beta_se: float
    beta_t: float
    beta_p_value: float
    bias_correction_applied: float | None


@dataclass(frozen=True)
class EquivalenceTest3Result(AnalysisResult):
    """Result of :func:`equivalence_test_3` (agreement).

    Joint test of H0: Deming intercept ``alpha = 0`` and slope
    ``beta = 1``, via a paired bootstrap with a Bonferroni-adjusted
    2.5%-per-parameter confidence level (spec: 5% global significance
    splits to a 97.5% confidence interval per parameter).

    Attributes
    ----------
    meta : AnalysisMeta
        Provenance and inclusion/exclusion record for this computation.
    alpha : float
        Point estimate of the Deming intercept (candidate on reference).
    beta : float
        Point estimate of the Deming slope.
    alpha_ci : tuple of (float, float)
        Bootstrap confidence interval for `alpha`, at
        `confidence_level_per_parameter`.
    beta_ci : tuple of (float, float)
        Bootstrap confidence interval for `beta`, at
        `confidence_level_per_parameter`.
    reject_joint_null : bool
        True if either `alpha_ci` excludes 0 or `beta_ci` excludes 1
        (i.e. the joint null of perfect agreement is rejected).
    confidence_level_per_parameter : float
        Per-parameter confidence level actually used,
        ``1 - alpha / 2`` (the Bonferroni-adjusted level).
    bias_correction_applied : float or None
        The `bias_correction` value that was subtracted from `candidate`
        before fitting, or None if none was applied.
    """

    alpha: float
    beta: float
    alpha_ci: tuple[float, float]
    beta_ci: tuple[float, float]
    reject_joint_null: bool
    confidence_level_per_parameter: float
    bias_correction_applied: float | None


def equivalence_test_1(
    reference: NDArray[np.float64],
    candidate: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
) -> EquivalenceTest1Result:
    """Silveira et al. equivalence Test 1: accuracy.

    Fits ``d = candidate - reference`` against the mean-centered
    reference, ``d = alpha + beta * (reference - mean(reference)) +
    nu``, and tests H0: ``alpha = 0`` (no systematic translation bias).

    Parameters
    ----------
    reference : ndarray of float, shape (n_samples,)
        Values from the established/ground-truth method.
    candidate : ndarray of float, shape (n_samples,)
        Values from the model or method under evaluation.
    sample_ids : ndarray of str, shape (n_samples,)
        Per-sample identifiers, used for inclusion/exclusion bookkeeping.

    Returns
    -------
    EquivalenceTest1Result
        Fitted intercept/slope and the accuracy test statistics. See
        :class:`EquivalenceTest1Result` for the full field list.

    Notes
    -----
    Backed by :func:`scipy.stats.linregress`. Fewer than 3 valid pairs
    yields a NaN-filled result with
    ``meta.status = ResultStatus.INSUFFICIENT_DATA``.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.method_comparison import equivalence_test_1
    >>> reference = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
    >>> candidate = np.array([10.6, 12.3, 13.7, 16.5, 18.1, 20.9])
    >>> ids = np.array(["s1", "s2", "s3", "s4", "s5", "s6"])
    >>> result = equivalence_test_1(reference, candidate, ids)
    >>> print(f"{result.alpha:.4f} {result.alpha_t:.4f} {result.alpha_p_value:.4f}")
    0.3500 1.8779 0.1336
    """
    all_ids, included_ids, ref, cand = _finite_pairs(reference, candidate, sample_ids)
    n = len(ref)
    n_invalid = len(all_ids) - len(included_ids)
    status = ResultStatus.VALID if n >= 3 else ResultStatus.INSUFFICIENT_DATA

    if status is not ResultStatus.VALID:
        meta = build_meta(
            status=status,
            all_ids=all_ids,
            included_ids=included_ids,
            exclusion_reasons={"nan_or_inf": n_invalid} if n_invalid else {},
            method="equivalence_test_1",
        )
        return EquivalenceTest1Result(
            meta=meta, alpha=float("nan"), beta=float("nan"),
            alpha_se=float("nan"), alpha_t=float("nan"), alpha_p_value=float("nan"),
        )

    d = cand - ref
    centered_ref = ref - np.mean(ref)
    fit = _stats.linregress(centered_ref, d)
    alpha, beta = float(fit.intercept), float(fit.slope)
    alpha_se = float(fit.intercept_stderr)
    df = n - 2
    alpha_t = alpha / alpha_se if alpha_se > 0 else float("nan")
    alpha_p = float(2 * _stats.t.sf(np.abs(alpha_t), df)) if np.isfinite(alpha_t) else float("nan")

    meta = build_meta(
        status=ResultStatus.VALID,
        all_ids=all_ids,
        included_ids=included_ids,
        exclusion_reasons={"nan_or_inf": n_invalid} if n_invalid else {},
        method="equivalence_test_1",
        configuration={"regression": "d ~ 1 + (reference - mean(reference))", "backend": "scipy.stats.linregress"},
    )
    return EquivalenceTest1Result(
        meta=meta, alpha=alpha, beta=beta, alpha_se=alpha_se, alpha_t=alpha_t, alpha_p_value=alpha_p
    )


def equivalence_test_2(
    reference: NDArray[np.float64],
    candidate: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
    *,
    bias_correction: float | None = None,
) -> EquivalenceTest2Result:
    """Silveira et al. equivalence Test 2: precision (Shukla 1973).

    Fits ``d = candidate - reference`` against ``candidate +
    reference``, ``d = alpha + beta * (candidate + reference) + theta``,
    and tests H0: ``beta = 0`` (equivalent to a reference/candidate
    variance ratio of 1).

    Parameters
    ----------
    reference : ndarray of float, shape (n_samples,)
        Values from the established/ground-truth method.
    candidate : ndarray of float, shape (n_samples,)
        Values from the model or method under evaluation.
    sample_ids : ndarray of str, shape (n_samples,)
        Per-sample identifiers, used for inclusion/exclusion bookkeeping.
    bias_correction : float or None, default None
        Value subtracted from `candidate` before fitting, per the
        manuscript's translation correction applied when Test 1's
        accuracy null (``alpha = 0``) is not satisfied. Typically Test
        1's fitted ``alpha``. None applies no correction.

    Returns
    -------
    EquivalenceTest2Result
        Fitted intercept/slope and the precision test statistics. See
        :class:`EquivalenceTest2Result` for the full field list.

    Notes
    -----
    Backed by :func:`scipy.stats.linregress`. Fewer than 3 valid pairs
    yields a NaN-filled result with
    ``meta.status = ResultStatus.INSUFFICIENT_DATA``.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.method_comparison import equivalence_test_2
    >>> reference = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
    >>> candidate = np.array([10.6, 12.3, 13.7, 16.5, 18.1, 20.9])
    >>> ids = np.array(["s1", "s2", "s3", "s4", "s5", "s6"])
    >>> result = equivalence_test_2(reference, candidate, ids)
    >>> print(f"{result.beta:.4f} {result.beta_t:.4f} {result.beta_p_value:.4f}")
    0.0149 0.5596 0.6056
    """
    all_ids, included_ids, ref, cand = _finite_pairs(reference, candidate, sample_ids)
    n = len(ref)
    n_invalid = len(all_ids) - len(included_ids)
    status = ResultStatus.VALID if n >= 3 else ResultStatus.INSUFFICIENT_DATA

    if status is not ResultStatus.VALID:
        meta = build_meta(
            status=status,
            all_ids=all_ids,
            included_ids=included_ids,
            exclusion_reasons={"nan_or_inf": n_invalid} if n_invalid else {},
            method="equivalence_test_2",
        )
        return EquivalenceTest2Result(
            meta=meta, alpha=float("nan"), beta=float("nan"),
            beta_se=float("nan"), beta_t=float("nan"), beta_p_value=float("nan"),
            bias_correction_applied=bias_correction,
        )

    if bias_correction is not None:
        cand = cand - bias_correction
    d = cand - ref
    sum_xy = cand + ref
    fit = _stats.linregress(sum_xy, d)
    alpha, beta = float(fit.intercept), float(fit.slope)
    beta_se = float(fit.stderr)
    df = n - 2
    beta_t = beta / beta_se if beta_se > 0 else float("nan")
    beta_p = float(2 * _stats.t.sf(np.abs(beta_t), df)) if np.isfinite(beta_t) else float("nan")

    meta = build_meta(
        status=ResultStatus.VALID,
        all_ids=all_ids,
        included_ids=included_ids,
        exclusion_reasons={"nan_or_inf": n_invalid} if n_invalid else {},
        method="equivalence_test_2",
        configuration={
            "regression": "d ~ 1 + (candidate + reference)",
            "backend": "scipy.stats.linregress",
            "bias_correction_applied": bias_correction,
        },
    )
    return EquivalenceTest2Result(
        meta=meta, alpha=alpha, beta=beta, beta_se=beta_se, beta_t=beta_t, beta_p_value=beta_p,
        bias_correction_applied=bias_correction,
    )


def equivalence_test_3(
    reference: NDArray[np.float64],
    candidate: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
    *,
    reference_error_variance: float,
    test_error_variance: float,
    bias_correction: float | None = None,
    alpha: float = 0.05,
    n_bootstrap: int = 2000,
    seed: int | None = None,
) -> EquivalenceTest3Result:
    """Silveira et al. equivalence Test 3: agreement.

    Joint test of H0: Deming intercept ``alpha = 0`` and slope ``beta =
    1``, via a paired bootstrap over the :func:`deming` regression of
    `candidate` on `reference`, using a per-parameter confidence level
    ``1 - alpha / 2`` (Bonferroni correction over the two joint-null
    parameters).

    Parameters
    ----------
    reference : ndarray of float, shape (n_samples,)
        Values from the established/ground-truth method.
    candidate : ndarray of float, shape (n_samples,)
        Values from the model or method under evaluation.
    sample_ids : ndarray of str, shape (n_samples,)
        Per-sample identifiers, used for inclusion/exclusion bookkeeping.
    reference_error_variance : float
        Assumed error variance of `reference`, passed through to
        :func:`deming`. Must be finite and strictly positive.
    test_error_variance : float
        Assumed error variance of `candidate`, passed through to
        :func:`deming`. Must be finite and strictly positive.
    bias_correction : float or None, default None
        Value subtracted from `candidate` before any fitting (point
        estimate and every bootstrap replicate), analogous to
        `equivalence_test_2`'s `bias_correction`.
    alpha : float, default 0.05
        Global two-sided significance level for the joint test. Split
        Bonferroni-style across the two joint-null parameters to a
        per-parameter confidence level of ``1 - alpha / 2``.
    n_bootstrap : int, default 2000
        Number of paired bootstrap resamples used to build the
        confidence intervals.
    seed : int or None, default None
        Seed for the random number generator driving the bootstrap
        resampling; None draws fresh entropy.

    Returns
    -------
    EquivalenceTest3Result
        Point estimates, their bootstrap confidence intervals, and the
        joint-null rejection decision. See :class:`EquivalenceTest3Result`
        for the full field list.

    Notes
    -----
    The point estimate and every bootstrap replicate are obtained by
    calling :func:`deming` (with `reference` as ``y`` and `candidate` as
    ``x``; see that function's Notes for the regression-direction
    convention and the manuscript-consistency caveat that governs its
    slope formula). If the point-estimate Deming fit is not
    ``ResultStatus.VALID`` (e.g. undefined slope, insufficient data),
    that status is propagated unchanged and no bootstrap is run.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.method_comparison import equivalence_test_3
    >>> reference = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
    >>> candidate = np.array([10.6, 12.3, 13.7, 16.5, 18.1, 20.9])
    >>> ids = np.array(["s1", "s2", "s3", "s4", "s5", "s6"])
    >>> result = equivalence_test_3(
    ...     reference, candidate, ids,
    ...     reference_error_variance=1.0, test_error_variance=1.0,
    ...     n_bootstrap=200, seed=0,
    ... )
    >>> print(f"{result.alpha:.4f} {result.beta:.4f}")
    -0.1038 1.0303
    >>> result.reject_joint_null, result.confidence_level_per_parameter
    (False, 0.975)
    """

    if bias_correction is not None:
        candidate = np.asarray(candidate, dtype=np.float64) - bias_correction

    point = deming(
        reference, candidate, sample_ids,
        reference_error_variance=reference_error_variance,
        test_error_variance=test_error_variance,
    )
    if point.meta.status is not ResultStatus.VALID:
        return EquivalenceTest3Result(
            meta=point.meta,
            alpha=point.intercept,
            beta=point.slope,
            alpha_ci=(float("nan"), float("nan")),
            beta_ci=(float("nan"), float("nan")),
            reject_joint_null=False,
            confidence_level_per_parameter=1.0 - alpha / 2.0,
            bias_correction_applied=bias_correction,
        )

    all_ids, included_ids, ref, cand = _finite_pairs(reference, candidate, sample_ids)
    n = len(ref)
    rng = np.random.default_rng(seed)
    boot_alpha = np.empty(n_bootstrap)
    boot_beta = np.empty(n_bootstrap)
    idx_range = np.arange(n)
    for b in range(n_bootstrap):
        idx = rng.choice(idx_range, size=n, replace=True)
        rep = deming(
            ref[idx], cand[idx], np.asarray(included_ids)[idx],
            reference_error_variance=reference_error_variance,
            test_error_variance=test_error_variance,
        )
        boot_alpha[b] = rep.intercept
        boot_beta[b] = rep.slope

    per_param_level = 1.0 - alpha / 2.0
    tail = (1.0 - per_param_level) / 2.0
    alpha_ci = tuple(np.quantile(boot_alpha, [tail, 1 - tail]))
    beta_ci = tuple(np.quantile(boot_beta, [tail, 1 - tail]))
    reject = not (alpha_ci[0] <= 0.0 <= alpha_ci[1] and beta_ci[0] <= 1.0 <= beta_ci[1])

    meta = build_meta(
        status=ResultStatus.VALID,
        all_ids=all_ids,
        included_ids=included_ids,
        method="equivalence_test_3",
        configuration={
            "ci_method": "bootstrap_percentile",
            "n_bootstrap": n_bootstrap,
            "bonferroni_global_alpha": alpha,
            "confidence_level_per_parameter": per_param_level,
            "bias_correction_applied": bias_correction,
        },
        random_seed=seed,
    )
    return EquivalenceTest3Result(
        meta=meta,
        alpha=point.intercept,
        beta=point.slope,
        alpha_ci=alpha_ci,
        beta_ci=beta_ci,
        reject_joint_null=reject,
        confidence_level_per_parameter=per_param_level,
        bias_correction_applied=bias_correction,
    )
