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

All resampling (bootstrap, permutation) is driven by
``np.random.default_rng(seed)``; the seed is recorded in each result's
``meta.random_seed``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats as _stats

from .._numerics import iqr as _iqr
from ..results import AnalysisResult, ResultStatus, build_meta

__all__ = [
    "BlandAltmanResult",
    "DemingResult",
    "EquivalenceTest1Result",
    "EquivalenceTest2Result",
    "EquivalenceTest3Result",
    "PassingBablokResult",
    "bland_altman",
    "deming",
    "equivalence_test_1",
    "equivalence_test_2",
    "equivalence_test_3",
    "passing_bablok",
]

BAAxis = Literal["reference", "mean"]
BALocation = Literal["mean", "median"]
BAScale = Literal["sd", "robust_sd"]
BALoAMethod = Literal["normal", "quantile"]
BACIMethod = Literal["bootstrap", "analytic", "none"]
PBEstimator = Literal["passing_bablok_1983", "manuscript_literal"]
PBCIMethod = Literal["analytic", "bootstrap"]
PBCusumMethod = Literal["bootstrap", "passing_bablok_1983", "permutation"]

_Z95 = 1.96
#: Normal-consistency constant for the MAD: ``1 / Phi^{-1}(0.75)``, ~1.4826.
_MAD_CONSISTENCY = float(1.0 / _stats.norm.ppf(0.75))
_GAUSSIAN_MEAN_ABS_CORRECTION = float(np.sqrt(np.pi / 2.0))
_BA_QUANTILE_METHOD = "linear"
_NAN_PAIR = (float("nan"), float("nan"))
#: Minimum number of valid CUSUM bootstrap replicates for a p-value.
_PB_CUSUM_MIN_VALID = 20


def _finite_pairs(
    reference: NDArray[np.float64], candidate: NDArray[np.float64], sample_ids: NDArray[np.str_]
):
    all_ids = tuple(np.asarray(sample_ids, dtype=str).tolist())
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    valid = np.isfinite(ref) & np.isfinite(cand)
    included_ids = tuple(np.asarray(all_ids)[valid].tolist())
    return all_ids, included_ids, ref[valid], cand[valid]


def _check_choice(name: str, value: object, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")


def _percentile_ci(samples: NDArray[np.float64], confidence_level: float, axis: int = 0):
    """Percentile interval of `samples` along `axis`, ignoring NaN replicates."""
    tail = (1.0 - confidence_level) / 2.0
    samples = np.asarray(samples, dtype=np.float64)
    finite_any = np.any(np.isfinite(samples), axis=axis)
    if samples.ndim == 1:
        if not finite_any:
            return _NAN_PAIR
        low, high = np.nanquantile(samples, [tail, 1.0 - tail])
        return (float(low), float(high))
    with np.errstate(all="ignore"):
        safe = np.where(np.isfinite(samples), samples, np.nan)
        low = np.full(safe.shape[1 - axis], np.nan)
        high = np.full(safe.shape[1 - axis], np.nan)
        if np.any(finite_any):
            cols = np.nonzero(finite_any)[0]
            q = np.nanquantile(np.take(safe, cols, axis=1 - axis), [tail, 1.0 - tail], axis=axis)
            low[cols], high[cols] = q[0], q[1]
    return (low, high)


# ---------------------------------------------------------------------------
# Bland-Altman
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlandAltmanResult(AnalysisResult):
    """Result of a :func:`bland_altman` agreement analysis.

    Attributes
    ----------
    meta : AnalysisMeta
        Provenance and inclusion/exclusion record for this computation.
    bias : float
        Constant (unconditional) location of the differences
        ``d_n = candidate_n - reference_n``: their mean or median, per
        `location`. NaN if the result is not ``VALID``.
    scale : float
        Constant dispersion of the differences, on the standard-deviation
        scale: the sample SD (``scale_method="sd"``) or the
        normal-consistent robust SD ``1.4826 * MAD`` (``"robust_sd"``).
    loa_low : float
        Constant lower 95% limit of agreement: ``bias - 1.96 * scale``
        (``loa_method="normal"``) or the empirical 2.5% quantile of the
        differences (``"quantile"``).
    loa_high : float
        Constant upper 95% limit of agreement: ``bias + 1.96 * scale``
        or the empirical 97.5% quantile of the differences.
    iqr_of_differences : float
        Interquartile range of the differences, a distribution-free
        companion to the limits of agreement.
    x_axis : {"reference", "mean"}
        Which quantity the differences were assessed against; see
        `bland_altman`.
    location : {"mean", "median"}
        Which location estimator was used for `bias` (and the estimator
        family of the conditional bias model).
    scale_method : {"sd", "robust_sd"}
        Which dispersion estimator was used for `scale` (and the
        estimator family of the conditional scale model).
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
        ``(s0, s1)`` intercept and slope of the fitted, consistency
        corrected ``SD(c) = s0 + s1 * c`` model (already on the SD scale,
        so the limits are ``bias(c) +/- 1.96 * SD(c)``), or None when
        `conditional` is False.
    loa_method : {"normal", "quantile"}
        How the constant limits of agreement were formed.
    conditional_loa_low : ndarray of float or None
        ``bias(c) - 1.96 * SD(c)`` evaluated at `x_values`, or None when
        `conditional` is False.
    conditional_loa_high : ndarray of float or None
        ``bias(c) + 1.96 * SD(c)`` evaluated at `x_values`, or None when
        `conditional` is False.
    ci_method : {"bootstrap", "analytic", "none"}
        How the confidence intervals below were obtained.
    confidence_level : float
        Nominal two-sided confidence level of every interval/band.
    bias_ci : tuple of (float, float)
        Confidence interval for `bias` (NaN for ``ci_method="none"``).
    loa_low_ci : tuple of (float, float)
        Confidence interval for `loa_low`.
    loa_high_ci : tuple of (float, float)
        Confidence interval for `loa_high`.
    ci_grid : ndarray of float or None
        Fixed grid of x-axis values (spanning ``min(x_values)`` to
        ``max(x_values)``) on which the pointwise conditional bands are
        evaluated; None unless `conditional` is True and
        ``ci_method="bootstrap"``.
    conditional_bias_band : tuple of (ndarray, ndarray) or None
        Pointwise ``(low, high)`` bootstrap band for ``bias(c)`` on
        `ci_grid`.
    conditional_loa_low_band : tuple of (ndarray, ndarray) or None
        Pointwise ``(low, high)`` bootstrap band for the conditional
        lower limit on `ci_grid`.
    conditional_loa_high_band : tuple of (ndarray, ndarray) or None
        Pointwise ``(low, high)`` bootstrap band for the conditional
        upper limit on `ci_grid`.
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
    loa_method: BALoAMethod = "normal"
    conditional_loa_low: NDArray[np.float64] | None = None
    conditional_loa_high: NDArray[np.float64] | None = None
    ci_method: BACIMethod = "none"
    confidence_level: float = 0.95
    bias_ci: tuple[float, float] = _NAN_PAIR
    loa_low_ci: tuple[float, float] = _NAN_PAIR
    loa_high_ci: tuple[float, float] = _NAN_PAIR
    ci_grid: NDArray[np.float64] | None = None
    conditional_bias_band: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
    conditional_loa_low_band: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
    conditional_loa_high_band: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None


def _theil_sen(y: NDArray[np.float64], x: NDArray[np.float64]) -> tuple[float, float]:
    """Return ``(intercept, slope)`` of a Theil-Sen fit of `y` on `x`.

    Numerically identical to the point estimate of
    ``scipy.stats.theilslopes(y, x)`` (median of the pairwise slopes over
    pairs with distinct `x`; intercept ``median(y) - slope * median(x)``,
    scipy's default ``method="separate"``), without the slope confidence
    interval that scipy also computes, which dominates the bootstrap cost.
    """
    i, j = np.triu_indices(x.shape[0], k=1)
    dx = x[j] - x[i]
    keep = dx != 0
    slope = float(np.median((y[j] - y[i])[keep] / dx[keep]))
    return float(np.median(y) - slope * np.median(x)), slope


def _ols_line(y: NDArray[np.float64], x: NDArray[np.float64]) -> tuple[float, float]:
    """Return ``(intercept, slope)`` of an OLS fit of `y` on `x`."""
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept), float(slope)


def _ba_core(
    ref: NDArray[np.float64],
    cand: NDArray[np.float64],
    *,
    x_axis: str,
    location: str,
    scale: str,
    loa_method: str,
    conditional: bool,
    grid: NDArray[np.float64] | None = None,
) -> dict:
    """Bland-Altman point estimates for one (possibly resampled) data set."""
    d = cand - ref
    x_values = ref if x_axis == "reference" else (ref + cand) / 2.0

    bias = float(np.mean(d)) if location == "mean" else float(np.median(d))
    if scale == "sd":
        disp = float(np.std(d, ddof=1))
    else:
        disp = float(np.median(np.abs(d - np.median(d)))) * _MAD_CONSISTENCY
    if loa_method == "normal":
        loa_low, loa_high = bias - _Z95 * disp, bias + _Z95 * disp
    else:
        q_low, q_high = np.quantile(d, [0.025, 0.975], method=_BA_QUANTILE_METHOD)
        loa_low, loa_high = float(q_low), float(q_high)

    out = {
        "d": d,
        "x_values": x_values,
        "bias": bias,
        "scale": disp,
        "loa_low": loa_low,
        "loa_high": loa_high,
        "bias_coef": None,
        "scale_coef": None,
    }
    if not conditional:
        return out

    if np.ptp(x_values) == 0:
        nan_coef = (float("nan"), float("nan"))
        out["bias_coef"] = nan_coef
        out["scale_coef"] = nan_coef
    else:
        fit_bias = _ols_line if location == "mean" else _theil_sen
        b0, b1 = fit_bias(d, x_values)
        abs_resid = np.abs(d - (b0 + b1 * x_values))
        if scale == "sd":
            s0, s1 = _ols_line(abs_resid, x_values)
            k = _GAUSSIAN_MEAN_ABS_CORRECTION
        else:
            s0, s1 = _theil_sen(abs_resid, x_values)
            k = _MAD_CONSISTENCY
        out["bias_coef"] = (b0, b1)
        out["scale_coef"] = (s0 * k, s1 * k)

    b0, b1 = out["bias_coef"]
    s0, s1 = out["scale_coef"]
    out["cond_loa_low"] = (b0 + b1 * x_values) - _Z95 * (s0 + s1 * x_values)
    out["cond_loa_high"] = (b0 + b1 * x_values) + _Z95 * (s0 + s1 * x_values)
    if grid is not None:
        bias_grid = b0 + b1 * grid
        sd_grid = s0 + s1 * grid
        out["grid_bias"] = bias_grid
        out["grid_loa_low"] = bias_grid - _Z95 * sd_grid
        out["grid_loa_high"] = bias_grid + _Z95 * sd_grid
    return out


def bland_altman(
    reference: NDArray[np.float64],
    candidate: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
    *,
    x_axis: BAAxis = "reference",
    location: BALocation = "median",
    scale: BAScale = "robust_sd",
    loa_method: BALoAMethod = "normal",
    conditional: bool = True,
    ci_method: BACIMethod = "bootstrap",
    confidence_level: float = 0.95,
    n_bootstrap: int = 2000,
    n_ci_grid: int = 50,
    seed: int | None = None,
) -> BlandAltmanResult:
    r"""Compute a Bland-Altman agreement analysis between two methods.

    Summarizes the paired differences ``d_n = candidate_n - reference_n``
    by a location (bias) and dispersion (limits of agreement), by default
    conditioned on the value of the plotting axis (regression-based
    limits of agreement, manuscript eq. ``ba_loa_heteroscedastic``) with
    robust estimators and bootstrap confidence intervals, as the
    manuscript prescribes.

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
    location : {"mean", "median"}, default "median"
        Location estimator for `bias`. ``"median"`` (the manuscript's
        robust, primary choice) also makes the conditional bias model a
        Theil-Sen fit; ``"mean"`` makes it an OLS fit.
    scale : {"sd", "robust_sd"}, default "robust_sd"
        Dispersion estimator. ``"sd"`` is the sample standard deviation
        (``ddof=1``). ``"robust_sd"`` is the normal-consistent robust SD
        ``1.4826 * MAD`` with ``MAD = median|d - median(d)|`` (the
        manuscript's unscaled :math:`\widetilde{\sigma}_d`, cf.
        :func:`mrsvalidate._numerics.robust_sd`, multiplied by
        ``1 / Phi^{-1}(0.75)``), so that ``bias +/- 1.96 * scale`` has
        approximately 95% coverage for normal differences; the unscaled
        MAD would give only ~81%.
    loa_method : {"normal", "quantile"}, default "normal"
        ``"normal"``: ``bias +/- 1.96 * scale``. ``"quantile"``: the
        empirical 2.5% and 97.5% quantiles of the differences (fully
        nonparametric; numpy's default ``"linear"`` quantile method).
        ``"quantile"`` is only defined for the constant limits, so it
        requires ``conditional=False``.
    conditional : bool, default True
        If True, additionally fit transparent linear location/scale
        models of the differences against the chosen `x_axis` (see
        Notes). The constant summaries (`bias`, `scale`, `loa_low`,
        `loa_high`) are always reported as well.
    ci_method : {"bootstrap", "analytic", "none"}, default "bootstrap"
        ``"bootstrap"``: paired percentile bootstrap over
        ``(reference, candidate)`` pairs, recomputing the whole analysis
        with the same settings in every replicate; provides intervals for
        the constant summaries and, when `conditional`, pointwise bands on
        `ci_grid`. ``"analytic"``: Bland & Altman (1999) t-based intervals,
        only valid for ``location="mean"``, ``scale="sd"``,
        ``loa_method="normal"``; no conditional bands are produced.
        ``"none"``: NaN intervals.
    confidence_level : float, default 0.95
        Nominal two-sided confidence level of all intervals/bands.
    n_bootstrap : int, default 2000
        Number of paired bootstrap resamples (``ci_method="bootstrap"``).
    n_ci_grid : int, default 50
        Number of equally spaced grid points spanning
        ``[min(x_values), max(x_values)]`` for the conditional bands.
    seed : int or None, default None
        Seed for ``np.random.default_rng`` driving the bootstrap; None
        draws fresh entropy.

    Returns
    -------
    BlandAltmanResult
        Bias, scale, limits of agreement, their confidence intervals and
        (if requested) the conditional location/scale models with
        pointwise confidence bands. See :class:`BlandAltmanResult` for the
        full field list.

    Raises
    ------
    ValueError
        For an unknown option value; for ``loa_method="quantile"`` with
        ``conditional=True`` (a conditional empirical-quantile limit is
        not defined here); or for ``ci_method="analytic"`` with any
        combination other than ``location="mean"``, ``scale="sd"``,
        ``loa_method="normal"`` (use ``ci_method="bootstrap"``).

    Notes
    -----
    When `conditional` is True, two linear fits are made against the
    chosen axis value :math:`c`:

    .. math::

        \mathrm{bias}(c) = b_0 + b_1 c, \qquad
        \mathrm{SD}(c) = s_0 + s_1 c,

    and the limits of agreement are
    :math:`\mathrm{bias}(c) \pm 1.96\,\mathrm{SD}(c)`. The bias model
    family follows `location` and the scale model family follows `scale`:

    * ``location="mean"``: OLS of :math:`d` on :math:`c`;
      ``location="median"``: Theil-Sen (:func:`scipy.stats.theilslopes`)
      of :math:`d` on :math:`c`.
    * ``scale="sd"``: OLS of :math:`|d - \widehat{\mathrm{bias}}(c)|` on
      :math:`c`, multiplied by :math:`\sqrt{\pi/2}` (for normal errors,
      :math:`E|e| = \sigma\sqrt{2/\pi}`); ``scale="robust_sd"``: Theil-Sen
      of :math:`|d - \widehat{\mathrm{bias}}(c)|` on :math:`c`, multiplied
      by :math:`1/\Phi^{-1}(0.75) \approx 1.4826` (for normal errors the
      conditional *median* of :math:`|e|` is :math:`0.6745\sigma`).

    Mixed combinations (e.g. ``location="median"`` with ``scale="sd"``)
    simply combine the two rules above. The fitted ``SD(c)`` is a linear
    model and is not clipped; it can become negative when extrapolated far
    outside the data. Conditional fits require at least two distinct
    x-axis values; otherwise the result is ``ResultStatus.UNDEFINED``.

    The analytic intervals (Bland & Altman 1999) are
    :math:`\bar d \pm t_{n-1,1-\alpha/2}\, s/\sqrt{n}` for the bias and
    :math:`\mathrm{LoA} \pm t_{n-1,1-\alpha/2}\, s
    \sqrt{1/n + 1.96^2 / (2(n-1))}` for each limit. Bootstrap replicates
    in which a conditional fit is undefined (all resampled x-axis values
    equal) are dropped from the percentile computation and counted in
    ``meta.configuration["n_bootstrap_failed"]``.

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

    Default (manuscript) analysis: robust median/1.4826*MAD summaries,
    Theil-Sen conditional models, and paired bootstrap intervals:

    >>> result = bland_altman(reference, candidate, ids, n_bootstrap=500, seed=0)
    >>> print(f"{result.bias:.4f} {result.scale:.4f}")
    0.4000 0.3707
    >>> print(f"{result.loa_low:.4f} {result.loa_high:.4f}")
    -0.3265 1.1265
    >>> print("{:.4f} {:.4f}".format(*result.bias_ci))
    -0.1000 0.7500
    >>> b0, b1 = result.conditional_bias_coefficients
    >>> print(f"{b0:.4f} {b1:.4f}")
    -0.0500 0.0300
    >>> result.ci_grid.shape, result.conditional_bias_band[0].shape
    ((50,), (50,))

    Classical parametric constant-variance analysis with the analytic
    (Bland & Altman 1999) intervals:

    >>> classic = bland_altman(
    ...     reference, candidate, ids,
    ...     location="mean", scale="sd", conditional=False, ci_method="analytic",
    ... )
    >>> print(f"{classic.bias:.4f} {classic.scale:.4f}")
    0.3500 0.4183
    >>> print(f"{classic.loa_low:.4f} {classic.loa_high:.4f}")
    -0.4699 1.1699
    >>> print("{:.4f} {:.4f}".format(*classic.bias_ci))
    -0.0890 0.7890
    """

    _check_choice("x_axis", x_axis, ("reference", "mean"))
    _check_choice("location", location, ("mean", "median"))
    _check_choice("scale", scale, ("sd", "robust_sd"))
    _check_choice("loa_method", loa_method, ("normal", "quantile"))
    _check_choice("ci_method", ci_method, ("bootstrap", "analytic", "none"))
    if loa_method == "quantile" and conditional:
        raise ValueError(
            "loa_method='quantile' is only defined for constant limits of agreement; "
            "pass conditional=False (or use loa_method='normal')."
        )
    if ci_method == "analytic" and not (location == "mean" and scale == "sd" and loa_method == "normal"):
        raise ValueError(
            "ci_method='analytic' (Bland & Altman 1999) is only valid for location='mean', "
            "scale='sd', loa_method='normal'; use ci_method='bootstrap' for robust or "
            "quantile-based limits."
        )
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level!r}")

    all_ids, included_ids, ref, cand = _finite_pairs(reference, candidate, sample_ids)
    n = len(ref)
    status = ResultStatus.VALID if n >= 2 else ResultStatus.INSUFFICIENT_DATA
    n_invalid = len(all_ids) - len(included_ids)
    warnings: list[str] = []
    if status is ResultStatus.VALID and conditional:
        x_check = ref if x_axis == "reference" else (ref + cand) / 2.0
        if np.ptp(x_check) == 0:
            status = ResultStatus.UNDEFINED
            warnings.append("conditional Bland-Altman models need at least two distinct x-axis values")

    configuration = {
        "x_axis": x_axis,
        "location": location,
        "scale_method": scale,
        "loa_method": loa_method,
        "conditional": conditional,
        "loa_z": _Z95,
        "robust_sd_consistency_constant": _MAD_CONSISTENCY,
        "loa_quantiles": (0.025, 0.975) if loa_method == "quantile" else None,
        "quantile_method": _BA_QUANTILE_METHOD if loa_method == "quantile" else None,
        "conditional_bias_model": (
            None if not conditional else ("ols" if location == "mean" else "theil_sen")
        ),
        "conditional_scale_model": (
            None
            if not conditional
            else (
                "ols_abs_residual * sqrt(pi/2)"
                if scale == "sd"
                else "theil_sen_abs_residual * 1/Phi^-1(0.75)"
            )
        ),
        "ci_method": ci_method,
        "confidence_level": confidence_level,
        "n_bootstrap": n_bootstrap if ci_method == "bootstrap" else None,
        "bootstrap_design": "paired_percentile" if ci_method == "bootstrap" else None,
        "n_ci_grid": n_ci_grid if (conditional and ci_method == "bootstrap") else None,
    }

    if status is not ResultStatus.VALID:
        meta = build_meta(
            status=status,
            all_ids=all_ids,
            included_ids=included_ids,
            exclusion_reasons={"nan_or_inf": n_invalid} if n_invalid else {},
            warnings=tuple(warnings),
            method="bland_altman",
            configuration=configuration,
            random_seed=seed,
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
            loa_method=loa_method,
            ci_method=ci_method,
            confidence_level=confidence_level,
        )

    settings = {
        "x_axis": x_axis,
        "location": location,
        "scale": scale,
        "loa_method": loa_method,
        "conditional": conditional,
    }
    grid = None
    if conditional and ci_method == "bootstrap":
        x_all = ref if x_axis == "reference" else (ref + cand) / 2.0
        grid = np.linspace(float(np.min(x_all)), float(np.max(x_all)), n_ci_grid)
    point = _ba_core(ref, cand, grid=grid, **settings)

    bias_ci = loa_low_ci = loa_high_ci = _NAN_PAIR
    bias_band = loa_low_band = loa_high_band = None
    if ci_method == "analytic":
        d = point["d"]
        s = point["scale"]
        t_crit = float(_stats.t.ppf(0.5 + confidence_level / 2.0, n - 1))
        half_bias = t_crit * s / np.sqrt(n)
        half_loa = t_crit * s * np.sqrt(1.0 / n + _Z95**2 / (2.0 * (n - 1)))
        bias_ci = (float(np.mean(d) - half_bias), float(np.mean(d) + half_bias))
        loa_low_ci = (float(point["loa_low"] - half_loa), float(point["loa_low"] + half_loa))
        loa_high_ci = (float(point["loa_high"] - half_loa), float(point["loa_high"] + half_loa))
        if conditional:
            warnings.append(
                "pointwise conditional confidence bands are only produced with ci_method='bootstrap'"
            )
    elif ci_method == "bootstrap":
        rng = np.random.default_rng(seed)
        boot_const = np.full((n_bootstrap, 3), np.nan)
        boot_grid = np.full((n_bootstrap, 3, n_ci_grid), np.nan) if grid is not None else None
        n_failed = 0
        for b in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            rep = _ba_core(ref[idx], cand[idx], grid=grid, **settings)
            boot_const[b] = (rep["bias"], rep["loa_low"], rep["loa_high"])
            if boot_grid is not None:
                if np.all(np.isfinite(rep["bias_coef"])):
                    boot_grid[b, 0] = rep["grid_bias"]
                    boot_grid[b, 1] = rep["grid_loa_low"]
                    boot_grid[b, 2] = rep["grid_loa_high"]
                else:
                    n_failed += 1
        bias_ci = _percentile_ci(boot_const[:, 0], confidence_level)
        loa_low_ci = _percentile_ci(boot_const[:, 1], confidence_level)
        loa_high_ci = _percentile_ci(boot_const[:, 2], confidence_level)
        if boot_grid is not None:
            bias_band = _percentile_ci(boot_grid[:, 0, :], confidence_level)
            loa_low_band = _percentile_ci(boot_grid[:, 1, :], confidence_level)
            loa_high_band = _percentile_ci(boot_grid[:, 2, :], confidence_level)
        configuration["n_bootstrap_failed"] = n_failed
        if n_failed:
            warnings.append(
                f"{n_failed} bootstrap replicates had an undefined conditional fit and were dropped"
            )

    meta = build_meta(
        status=ResultStatus.VALID,
        all_ids=all_ids,
        included_ids=included_ids,
        exclusion_reasons={"nan_or_inf": n_invalid} if n_invalid else {},
        warnings=tuple(warnings),
        method="bland_altman",
        configuration=configuration,
        random_seed=seed,
    )
    return BlandAltmanResult(
        meta=meta,
        bias=point["bias"],
        scale=point["scale"],
        loa_low=point["loa_low"],
        loa_high=point["loa_high"],
        iqr_of_differences=_iqr(point["d"]),
        x_axis=x_axis,
        location=location,
        scale_method=scale,
        conditional=conditional,
        differences=point["d"],
        x_values=point["x_values"],
        conditional_bias_coefficients=point["bias_coef"],
        conditional_scale_coefficients=point["scale_coef"],
        loa_method=loa_method,
        conditional_loa_low=point.get("cond_loa_low"),
        conditional_loa_high=point.get("cond_loa_high"),
        ci_method=ci_method,
        confidence_level=confidence_level,
        bias_ci=bias_ci,
        loa_low_ci=loa_low_ci,
        loa_high_ci=loa_high_ci,
        ci_grid=grid,
        conditional_bias_band=bias_band,
        conditional_loa_low_band=loa_low_band,
        conditional_loa_high_band=loa_high_band,
    )


# ---------------------------------------------------------------------------
# Passing-Bablok
# ---------------------------------------------------------------------------


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
        Confidence interval for the slope at `confidence_level`, per
        `ci_method`.
    intercept_ci : tuple of (float, float)
        Confidence interval for the intercept at `confidence_level`, per
        `ci_method`.
    n_pairwise_slopes : int
        Number of pairwise slopes that entered the estimate (``N``),
        after the exclusions of the chosen `estimator`.
    n_excluded_equal_reference_pairs : int
        Number of sample pairs with equal reference values that were
        excluded: all of them for ``"manuscript_literal"``; only those
        that are also equal in `candidate` (identical points) for
        ``"passing_bablok_1983"``.
    cusum_max_abs : float
        Observed maximum absolute cumulative sum of the linearity
        diagnostic (of the PB-1983 residual scores for
        ``cusum_method="bootstrap"``/``"passing_bablok_1983"``, or of the
        standardized reference-ordered residuals for the permutation
        variant).
    cusum_p_value : float
        p-value of the CUSUM linearity test: wild-bootstrap Monte Carlo
        p-value for ``"bootstrap"``, asymptotic Kolmogorov for
        ``"passing_bablok_1983"``, permutation for ``"permutation"``.
    confidence_level : float
        Nominal confidence level used for `slope_ci` and `intercept_ci`.
    ci_method : {"analytic", "bootstrap"}
        Confidence-interval method used: the Passing & Bablok (1983)
        rank-based interval, or a paired percentile bootstrap.
    estimator : {"passing_bablok_1983", "manuscript_literal"}
        Which slope-estimator variant was used (see `passing_bablok`).
    n_infinite_slopes : int
        Number of ``+/-inf`` pairwise slopes included (pairs with equal
        reference but different candidate values; PB 1983 only).
    n_excluded_minus_one_slopes : int
        Number of pairwise slopes exactly equal to -1 that were discarded
        (PB 1983 only).
    shift_k : int
        ``K``, the number of included pairwise slopes below -1.
    cusum_method : {"bootstrap", "passing_bablok_1983", "permutation"}
        Which CUSUM linearity test was used.
    cusum_statistic : float
        The test statistic the p-value refers to: ``H = max|cusum| /
        sqrt(n_pos + n_neg + 1)`` for ``"bootstrap"`` and
        ``"passing_bablok_1983"``, `cusum_max_abs` for the permutation
        variant.
    cusum_critical_value : float
        Critical value of `cusum_statistic` at `cusum_alpha`: the
        ``1 - cusum_alpha`` quantile of the bootstrap replicates ``H*`` for
        ``"bootstrap"`` (informational; the decision uses the p-value),
        ``kstwobign.isf(cusum_alpha)`` (~1.358 at 5%) for PB 1983, NaN for
        the permutation variant.
    cusum_reject_linearity : bool
        True if linearity is rejected at `cusum_alpha`.
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
    estimator: PBEstimator = "passing_bablok_1983"
    n_infinite_slopes: int = 0
    n_excluded_minus_one_slopes: int = 0
    shift_k: int = 0
    cusum_method: PBCusumMethod = "passing_bablok_1983"
    cusum_statistic: float = float("nan")
    cusum_critical_value: float = float("nan")
    cusum_reject_linearity: bool = False


#: Up to this many pairs (n ~ 724) the pairwise slopes are built with
#: gathers through ``np.triu_indices``; above it they are built row by row,
#: which needs no index arrays and is faster once the gathers miss cache.
_PB_TRIU_MAX_PAIRS = 1 << 18


class _PBWorkspace:
    """Reusable buffers for repeated Passing-Bablok fits at a fixed ``n``.

    Allocating fresh ``n(n-1)/2``-sized temporaries for every replicate is
    dominated by page faults, so resampling loops reuse one workspace.
    Memory: ``P = n(n-1)/2`` pairs; ``42 P`` bytes up to
    `_PB_TRIU_MAX_PAIRS` pairs (two int64 index arrays, three float64 and
    two bool buffers; <= 11 MB), ``18 P`` bytes above it (e.g. ~9 MB at
    n = 1000, ~36 MB at n = 2000, ~225 MB at n = 5000).
    """

    def __init__(self, n: int):
        self.n = n
        self.n_pairs = n * (n - 1) // 2
        self.use_triu = self.n_pairs <= _PB_TRIU_MAX_PAIRS
        if self.use_triu:
            # Not marked read-only: numpy's take/fancy indexing is several
            # times slower with a non-writeable index array.
            self.i, self.j = np.triu_indices(n, k=1)
            self.tmp = np.empty(self.n_pairs)
        self.num = np.empty(self.n_pairs)
        self.den = np.empty(self.n_pairs)
        self.mask = np.empty(self.n_pairs, dtype=bool)
        self.mask2 = np.empty(self.n_pairs, dtype=bool)

    def raw_slopes(self, ref: NDArray[np.float64], cand: NDArray[np.float64]) -> NDArray[np.float64]:
        """``(cand_j - cand_i) / (ref_j - ref_i)`` for all ``i < j``, in a reused buffer.

        Pairs with equal reference give ``+/-inf`` by the sign of the
        candidate difference (``x - x`` is ``+0.0`` in IEEE arithmetic) and
        identical points give NaN.
        """
        num, den = self.num, self.den
        if self.use_triu:
            i, j, tmp = self.i, self.j, self.tmp
            np.take(cand, j, out=num)
            np.subtract(num, np.take(cand, i, out=tmp), out=num)
            np.take(ref, j, out=den)
            np.subtract(den, np.take(ref, i, out=tmp), out=den)
        else:
            n, start = self.n, 0
            for a in range(n - 1):
                stop = start + n - 1 - a
                np.subtract(cand[a + 1 :], cand[a], out=num[start:stop])
                np.subtract(ref[a + 1 :], ref[a], out=den[start:stop])
                start = stop
        with np.errstate(divide="ignore", invalid="ignore"):
            np.divide(num, den, out=num)
        return num

    def fit(self, ref: NDArray[np.float64], cand: NDArray[np.float64], estimator: str) -> tuple[float, float]:
        """``(slope, intercept)``; identical to :func:`_pb_estimate`.

        Excluded slopes are overwritten with ``+inf`` rather than removed:
        the ``m`` retained slopes are then the ``m`` smallest entries, so
        every order statistic of rank ``<= m`` is unchanged, and the needed
        one(s) are found with an in-place ``np.partition``.
        """
        s = self.raw_slopes(ref, cand)
        excl, tmp = self.mask, self.mask2
        if estimator == "manuscript_literal":
            np.isfinite(s, out=excl)
            np.logical_not(excl, out=excl)
        else:
            np.isnan(s, out=excl)
            np.logical_or(excl, np.equal(s, -1.0, out=tmp), out=excl)
        m = self.n_pairs - int(np.count_nonzero(excl))
        if m == 0:
            return float("nan"), float("nan")
        np.copyto(s, np.inf, where=excl)
        k = int(np.count_nonzero(np.less(s, -1.0, out=tmp)))  # before reordering
        positions = [_clamp_index(idx, m)[0] - 1 for idx in _pb_slope_indices(m, k, estimator)]
        lo = positions[0]
        s.partition(lo)
        beta1 = float(s[lo])
        if len(positions) == 2:
            hi = positions[1]
            # After partitioning at `lo`, everything to its right is >= s[lo],
            # so the next order statistic is their minimum (hi is lo or lo+1).
            upper = beta1 if hi == lo else float(s[lo + 1 :].min())
            with np.errstate(invalid="ignore"):
                beta1 = float(0.5 * (beta1 + upper))
        with np.errstate(invalid="ignore"):
            return beta1, float(np.median(cand - beta1 * ref))


def _raw_pairwise_slopes(ref: NDArray[np.float64], cand: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fresh array of all pairwise slopes (see :meth:`_PBWorkspace.raw_slopes`)."""
    return _PBWorkspace(ref.shape[0]).raw_slopes(ref, cand)


def _pb_slopes(ref: NDArray[np.float64], cand: NDArray[np.float64], estimator: str) -> dict:
    """Sorted pairwise slopes and exclusion counts for one data set."""
    raw = _raw_pairwise_slopes(ref, cand)
    n_inf = 0
    n_minus_one = 0
    if estimator == "manuscript_literal":
        keep = np.isfinite(raw)  # drops every equal-reference pair
        n_excl_equal_ref = int(raw.shape[0] - np.count_nonzero(keep))
        slopes = raw[keep]
    else:
        identical = np.isnan(raw)
        minus_one = raw == -1.0
        n_excl_equal_ref = int(np.count_nonzero(identical))
        n_minus_one = int(np.count_nonzero(minus_one))
        slopes = raw[~(identical | minus_one)]
        n_inf = int(np.count_nonzero(np.isinf(slopes)))
    k = int(np.count_nonzero(slopes < -1))
    slopes.sort()
    return {
        "slopes": slopes,
        "k": k,
        "n_excluded_equal_reference": n_excl_equal_ref,
        "n_infinite": n_inf,
        "n_minus_one": n_minus_one,
    }


def _clamp_index(idx_1indexed: int, m: int) -> tuple[int, bool]:
    clamped = min(max(idx_1indexed, 1), m)
    return clamped, clamped != idx_1indexed


def _clamped(sorted_slopes: NDArray[np.float64], idx_1indexed: int) -> tuple[float, bool]:
    clamped, was_clamped = _clamp_index(idx_1indexed, sorted_slopes.shape[0])
    return float(sorted_slopes[clamped - 1]), was_clamped


def _pb_slope_indices(m: int, k: int, estimator: str) -> list[int]:
    """1-indexed order statistics of the slopes whose mean is the PB slope."""
    if estimator == "manuscript_literal":
        idx = m // 2 + k + 1
        if idx > m:
            # Wrap-around: the shifted index is taken modulo M; without
            # this, a heavily negative-slope sample (K close to M) would
            # index past the end of the array.
            idx -= m
        return [idx]
    if m % 2 == 1:
        return [(m + 1) // 2 + k]
    return [m // 2 + k, m // 2 + k + 1]


def _pb_estimate(
    ref: NDArray[np.float64], cand: NDArray[np.float64], estimator: str = "passing_bablok_1983"
) -> dict:
    """Passing-Bablok slope/intercept plus the sorted slopes (for the analytic CI).

    Resampling loops use the faster, equivalent :meth:`_PBWorkspace.fit`.
    """
    info = _pb_slopes(ref, cand, estimator)
    slopes, k = info["slopes"], info["k"]
    m = slopes.shape[0]
    info["index_clamped"] = False
    if m == 0:
        info["slope"] = float("nan")
        info["intercept"] = float("nan")
        return info
    picks = [_clamp_index(idx, m) for idx in _pb_slope_indices(m, k, estimator)]
    values = slopes[[p - 1 for p, _ in picks]]
    with np.errstate(invalid="ignore"):
        beta1 = float(values[0]) if len(values) == 1 else float(0.5 * (values[0] + values[1]))
        info["index_clamped"] = any(c for _, c in picks)
        info["slope"] = beta1
        info["intercept"] = float(np.median(cand - beta1 * ref))
    return info


def _pb_analytic_ci(
    ref: NDArray[np.float64],
    cand: NDArray[np.float64],
    info: dict,
    confidence_level: float,
) -> tuple[tuple[float, float], tuple[float, float], bool]:
    n = ref.shape[0]
    slopes, k = info["slopes"], info["k"]
    m = slopes.shape[0]
    z = float(_stats.norm.ppf(0.5 + confidence_level / 2.0))
    c_gamma = z * np.sqrt(n * (n - 1) * (2 * n + 5) / 18.0)
    m1 = int(np.floor((m - c_gamma) / 2.0 + 0.5))  # round half up
    m2 = m - m1 + 1
    b_low, c1 = _clamped(slopes, m1 + k)
    b_high, c2 = _clamped(slopes, m2 + k)
    with np.errstate(invalid="ignore"):
        i_a = float(np.median(cand - b_high * ref))
        i_b = float(np.median(cand - b_low * ref))
    intercept_ci = (min(i_a, i_b), max(i_a, i_b)) if np.isfinite(i_a) and np.isfinite(i_b) else (i_a, i_b)
    return (b_low, b_high), intercept_ci, c1 or c2


def _pb_cusum_1983(
    ref: NDArray[np.float64], cand: NDArray[np.float64], b0: float, b1: float
) -> dict:
    """PB-1983 CUSUM statistic ``H`` for the line ``cand = b0 + b1 * ref``."""
    # Residuals written so that the median observation is exactly zero.
    e = (cand - b1 * ref) - b0
    tol = 1e-12 * max(1.0, float(np.max(np.abs(cand))), float(np.max(np.abs(b1 * ref))))
    pos = e > tol
    neg = e < -tol
    n_pos, n_neg = int(np.count_nonzero(pos)), int(np.count_nonzero(neg))
    out = {"n_pos": n_pos, "n_neg": n_neg}
    if n_pos == 0 or n_neg == 0 or not np.isfinite(b1):
        out.update(max_abs=float("nan"), statistic=float("nan"))
        return out
    r = np.zeros_like(e)
    r[pos] = np.sqrt(n_neg / n_pos)
    r[neg] = -np.sqrt(n_pos / n_neg)
    distance = (ref + b1 * (cand - b0)) / np.sqrt(1.0 + b1**2)
    order = np.argsort(distance, kind="stable")
    cusum = np.cumsum(r[order])
    max_abs = float(np.max(np.abs(cusum)))
    statistic = max_abs / np.sqrt(n_pos + n_neg + 1)
    out.update(max_abs=max_abs, statistic=float(statistic))
    return out


def _pb_cusum_wild_bootstrap(
    ref: NDArray[np.float64],
    cand: NDArray[np.float64],
    b0: float,
    b1: float,
    estimator: str,
    n_boot: int,
    rng: np.random.Generator,
    variance_ratio: float | None = None,
) -> NDArray[np.float64]:
    """Bootstrap replicates ``H*`` of the PB-1983 CUSUM statistic.

    Wild (Rademacher) bootstrap under the fitted line (Stute, Gonzalez
    Manteiga & Presedo Quindimil 1998): each point is written as its foot
    on the fitted line plus a residual along a direction ``d``, the
    residual is multiplied by ``w_i = +/-1``, the PB line is re-estimated
    with the same `estimator`, and ``H`` is recomputed exactly as for the
    data. ``variance_ratio=None`` uses the unit normal ``d = (-b1, 1) /
    sqrt(1 + b1^2)`` (perpendicular residuals; isotropic errors);
    ``variance_ratio = lam = var(ref error) / var(cand error)`` uses
    ``d = (-lam * b1, 1)``, the direction conjugate to the line under the
    error covariance ``diag(lam, 1)``, for which the residual along ``d``
    is uncorrelated with the error component along the line. Degenerate
    replicates (undefined slope, or all residuals on one side) are
    returned as NaN.
    """
    n = ref.shape[0]
    e = cand - b0 - b1 * ref
    if variance_ratio is None:
        norm = np.sqrt(1.0 + b1**2)
        d_ref, d_cand = -b1 / norm, 1.0 / norm
        coef = e / norm  # signed perpendicular residual e_perp
    else:
        d_ref, d_cand = -variance_ratio * b1, 1.0
        coef = e / (1.0 + variance_ratio * b1**2)
    foot_ref = ref - coef * d_ref
    foot_cand = cand - coef * d_cand
    out = np.full(n_boot, np.nan)
    ws = _PBWorkspace(n)
    for b in range(n_boot):
        we = coef * (2.0 * rng.integers(0, 2, size=n) - 1.0)
        ref_b = foot_ref + we * d_ref
        cand_b = foot_cand + we * d_cand
        slope_b, intercept_b = ws.fit(ref_b, cand_b, estimator)
        if not (np.isfinite(slope_b) and np.isfinite(intercept_b)):
            continue
        out[b] = _pb_cusum_1983(ref_b, cand_b, intercept_b, slope_b)["statistic"]
    return out


def passing_bablok(
    reference: NDArray[np.float64],
    candidate: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
    *,
    estimator: PBEstimator = "passing_bablok_1983",
    ci_method: PBCIMethod = "analytic",
    confidence_level: float = 0.95,
    n_bootstrap: int = 2000,
    cusum_method: PBCusumMethod = "passing_bablok_1983",
    cusum_alpha: float = 0.05,
    n_cusum_permutations: int = 2000,
    n_cusum_bootstrap: int = 1000,
    cusum_bootstrap_variance_ratio: float | None = None,
    seed: int | None = None,
) -> PassingBablokResult:
    r"""Compute a Passing-Bablok regression of `candidate` on `reference`.

    Implements the pairwise-slope construction of manuscript eqs.
    ``pb_slopes``/``pb_slope``/``pb_intercept``: `candidate` is modeled
    as a function of `reference`, ``candidate = beta0 + beta1 *
    reference``, via the shifted median of the pairwise slopes rather
    than least squares, together with its confidence intervals and the
    CUSUM linearity test.

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
    estimator : {"passing_bablok_1983", "manuscript_literal"}, default "passing_bablok_1983"
        Slope-estimator variant; see Notes.
    ci_method : {"analytic", "bootstrap"}, default "analytic"
        ``"analytic"``: the rank-based interval of Passing & Bablok
        (1983). ``"bootstrap"``: paired percentile bootstrap over
        ``(reference, candidate)`` pairs (same `estimator` per replicate).
    confidence_level : float, default 0.95
        Nominal two-sided confidence level for `slope_ci`/`intercept_ci`.
    n_bootstrap : int, default 2000
        Number of paired bootstrap resamples (``ci_method="bootstrap"``
        only).
    cusum_method : {"passing_bablok_1983", "bootstrap", "permutation"}, default "passing_bablok_1983"
        CUSUM linearity test; see Notes. ``"passing_bablok_1983"`` uses
        the classical asymptotic Kolmogorov critical value (very
        conservative, about 0-1% rejection at nominal 5%; reproduces
        published results). ``"bootstrap"`` calibrates the same statistic
        by a wild bootstrap under the fitted line (close to nominal; see
        Notes for its one known liberal case). ``"permutation"`` is a
        different, residual-permutation CUSUM.
    cusum_alpha : float, default 0.05
        Significance level of the linearity decision
        `cusum_reject_linearity`.
    n_cusum_permutations : int, default 2000
        Number of permutations (``cusum_method="permutation"`` only).
    n_cusum_bootstrap : int, default 1000
        Number of wild-bootstrap replicates (``cusum_method="bootstrap"``
        only). Each replicate refits the line, costing one Passing-Bablok
        fit (``O(n^2)`` pairwise slopes).
    cusum_bootstrap_variance_ratio : float or None, default None
        Error-variance ratio :math:`\lambda` = var(`reference` error) /
        var(`candidate` error) (the :func:`deming` convention) that sets
        the direction along which the bootstrap flips residuals
        (``cusum_method="bootstrap"`` only). None: perpendicular to the
        fitted line (isotropic errors, :math:`\lambda = 1`). Supply it when
        the ratio is known (e.g. from replicates) and far from 1; see Notes.
    seed : int or None, default None
        Seed for ``np.random.default_rng`` driving the bootstrap and
        permutation resampling; None draws fresh entropy.

    Returns
    -------
    PassingBablokResult
        Slope, intercept, their confidence intervals, and the CUSUM
        linearity diagnostic. See :class:`PassingBablokResult` for the
        full field list.

    Notes
    -----
    **Estimator.** For pairs :math:`i < j`, :math:`S_{ij} = (x_j - x_i) /
    (y_j - y_i)` with :math:`x` = `candidate`, :math:`y` = `reference`.

    * ``"passing_bablok_1983"`` (Passing & Bablok 1983): pairs with
      :math:`\Delta y = 0` and :math:`\Delta x = 0` are discarded; pairs
      with :math:`\Delta y = 0`, :math:`\Delta x \neq 0` give
      :math:`S = \pm\infty` by the sign of :math:`\Delta x` and are kept;
      slopes exactly equal to :math:`-1` are discarded. With the
      :math:`N` remaining slopes sorted and :math:`K = \#\{S < -1\}`,
      :math:`\hat\beta_1 = S_{((N+1)/2 + K)}` for odd :math:`N` and
      :math:`\tfrac12 (S_{(N/2+K)} + S_{(N/2+K+1)})` for even :math:`N`
      (1-indexed). The method assumes a positive slope; if the shifted
      index falls outside ``1..N`` (strongly negative slopes) it is
      clamped and a warning is recorded.
    * ``"manuscript_literal"``: the manuscript's eq. ``pb_slope``,
      :math:`S_{(\lfloor M/2 \rfloor + K + 1)}`, with every equal-reference
      pair discarded and the shifted index wrapped modulo :math:`M`.

    For both, :math:`\hat\beta_0 = \mathrm{median}(x_i - \hat\beta_1 y_i)`.

    **Analytic CI** (Passing & Bablok 1983): with :math:`n` observations,
    :math:`C = z_{1-\gamma/2}\sqrt{n(n-1)(2n+5)/18}`,
    :math:`M_1 = \mathrm{round}((N - C)/2)` (half up), :math:`M_2 = N -
    M_1 + 1`; the slope CI is :math:`[S_{(M_1+K)}, S_{(M_2+K)}]` and the
    intercept CI is
    :math:`[\mathrm{median}(x - b_U y), \mathrm{median}(x - b_L y)]`
    (reported in ascending order). Indices outside ``1..N`` (very small
    samples) are clamped and a warning is recorded.

    **CUSUM linearity statistic** (Passing & Bablok 1983, Sect. 4; used
    by ``"bootstrap"`` and ``"passing_bablok_1983"``): residuals
    :math:`e_i = x_i - (\hat\beta_0 + \hat\beta_1 y_i)`; with
    :math:`n_+ = \#(e > 0)`, :math:`n_- = \#(e < 0)`, scores
    :math:`r_i = \sqrt{n_-/n_+}` for :math:`e_i > 0`,
    :math:`-\sqrt{n_+/n_-}` for :math:`e_i < 0`, and 0 otherwise (so
    :math:`\sum r_i = 0`). Points are ordered by their projection on the
    fitted line, :math:`D_i = (y_i + \hat\beta_1 (x_i - \hat\beta_0)) /
    \sqrt{1 + \hat\beta_1^2}`, and the statistic is
    :math:`H = \max_k |\sum_{i \le k} r_{(i)}| / \sqrt{n_+ + n_- + 1}`.
    Residuals within a relative tolerance of :math:`10^{-12}` of zero are
    treated as zero; if :math:`n_+` or :math:`n_-` is zero the test is
    undefined (NaN, with a warning).

    ``"passing_bablok_1983"`` (asymptotic): because :math:`\sum r_i^2 =
    n_+ + n_-` and :math:`\sum r_i = 0`, with exchangeable residual signs
    along a *known* line the partial-sum process converges to a Brownian
    bridge and :math:`H` to the Kolmogorov distribution: p-value
    :func:`scipy.stats.kstwobign` ``.sf(H)``, rejection at 5% when
    :math:`H > 1.358`. But the line is *estimated from the same data*,
    and the median-type fit balances the residual signs along the line,
    which shrinks :math:`H`; this is the estimated-parameter problem of
    goodness-of-fit tests on empirical processes (Durbin 1973), for which
    the known-parameter limit no longer applies. Under a truly linear
    model with error in both methods the 95th percentile of :math:`H` is
    about 0.98-1.06 rather than 1.358, and the nominal 5% test rejects
    well under 1% of the time, so it is very conservative and loses power
    against mild curvature.

    ``"bootstrap"`` (wild bootstrap under the fitted model, after Stute,
    Gonzalez Manteiga & Presedo Quindimil 1998): each point is split into
    its foot on the fitted line and a residual along a direction
    :math:`d`, by default the unit normal :math:`\nu = (-\hat\beta_1,
    1)/\sqrt{1 + \hat\beta_1^2}` in (`reference`, `candidate`) coordinates,
    i.e. the perpendicular residual :math:`e^\perp_i = (x_i - \hat\beta_0 -
    \hat\beta_1 y_i) / \sqrt{1 + \hat\beta_1^2}`. In each replicate the
    residuals are multiplied by independent Rademacher signs :math:`w_i =
    \pm 1` and the point is placed at :math:`\mathrm{foot}_i + w_i
    e^\perp_i \nu`: data exactly linear around the fitted line, keeping
    each point's residual magnitude (hence heteroscedasticity). The line
    is then *re-estimated* with the same `estimator` and :math:`H^*` is
    computed exactly as :math:`H` (same scores, ordering and
    normalisation), so the bootstrap reproduces the fitting effect that
    makes the asymptotic test conservative. The p-value is :math:`(1 +
    \#\{H^* \ge H\}) / (B + 1)` over the :math:`B` valid replicates and
    linearity is rejected when it is :math:`\le` `cusum_alpha`;
    `cusum_critical_value` is the :math:`1 -` `cusum_alpha` quantile of
    :math:`H^*`. Replicates with an undefined slope or all residuals on
    one side are skipped and counted (warning; the valid count is
    ``meta.configuration["n_cusum_bootstrap_valid"]``); with fewer than
    ``max(20, n_cusum_bootstrap / 2)`` valid replicates the p-value is NaN
    (warning).

    Calibration (simulation, nominal 5%, errors in both methods): 2-5%
    rejection for homo- and heteroscedastic, skewed (lognormal, gamma)
    errors and error-variance ratios 0.25-4 at n = 50-1000, versus 0-1%
    for the asymptotic test, with much higher power against mild
    curvature. It is at or below nominal because the scores are usually
    :math:`\pm 1` (:math:`n_+ = n_-`), so :math:`H` lives on a coarse
    lattice and ties :math:`H^* = H` count as exceedances; the achieved
    level matches the best non-randomized level for this discrete
    statistic. Known liberal case: error variances far from isotropic
    (variance ratio :math:`\lambda` = var(`reference` error) /
    var(`candidate` error) :math:`\ne 1`) with a bounded, e.g. uniform,
    spread of true concentrations and large :math:`n`: the points at the
    ends of the line are selected on the error component along the line,
    which for :math:`\lambda \ne 1` is correlated with the perpendicular
    one, so residual signs there are not exchangeable, while perpendicular
    flipping removes that correlation (about 7% at n = 1000 for
    :math:`\lambda` = 0.25 or 4 with small errors, 15% with errors ~10% of
    the range). Supplying `cusum_bootstrap_variance_ratio` :math:`=
    \lambda` fixes this (about 5% in the same settings): residuals are
    then taken along :math:`d = (-\lambda \hat\beta_1, 1)`, conjugate to
    the line under the error covariance :math:`\mathrm{diag}(\lambda, 1)`
    so that the flipped component is uncorrelated with the along-line
    one, i.e. point :math:`= \mathrm{foot}_i + w_i c_i d` with :math:`c_i
    = (x_i - \hat\beta_0 - \hat\beta_1 y_i) / (1 + \lambda
    \hat\beta_1^2)`. Cost: :math:`B` Passing-Bablok refits, each
    :math:`O(n^2)` time and memory for the pairwise slopes (peak about
    13 MB at n = 1000, 52 MB at n = 2000); ``B = 1000`` takes about 0.2 s
    at n = 100, 2 s at n = 500, 6 s at n = 1000 and 20-30 s at n = 2000
    (single core).

    ``"permutation"``: the maximum absolute cumulative sum of
    SD-standardized, reference-ordered residuals with a permutation
    p-value (not the classical test; permuting residuals of a fitted line
    does not account for the fit either).

    Fewer than 3 valid pairs yields a NaN-filled result with
    ``meta.status = ResultStatus.INSUFFICIENT_DATA``; no usable pairwise
    slope yields ``ResultStatus.UNDEFINED``.

    References
    ----------
    .. [1] Passing, H. & Bablok, W. (1983). A new biometrical procedure for
       testing the equality of measurements from two different analytical
       methods. *J. Clin. Chem. Clin. Biochem.* 21, 709-720.
    .. [2] Durbin, J. (1973). Weak convergence of the sample distribution
       function when parameters are estimated. *Ann. Statist.* 1, 279-290.
    .. [3] Stute, W., Gonzalez Manteiga, W. & Presedo Quindimil, M. (1998).
       Bootstrap approximations in model checks for regression. *J. Amer.
       Statist. Assoc.* 93, 141-149.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.method_comparison import passing_bablok
    >>> reference = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
    >>> candidate = np.array([10.6, 12.3, 13.7, 16.5, 18.1, 20.9])
    >>> ids = np.array(["s1", "s2", "s3", "s4", "s5", "s6"])
    >>> result = passing_bablok(reference, candidate, ids)
    >>> print(f"{result.slope:.4f} {result.intercept:.4f}")
    1.0300 -0.0200
    >>> print("{:.4f} {:.4f}".format(*result.slope_ci))
    0.7750 1.4000
    >>> result.n_pairwise_slopes, result.shift_k
    (15, 0)

    The default CUSUM test compares the PB-1983 statistic ``H`` with the
    asymptotic Kolmogorov critical value; ``cusum_method="bootstrap"``
    calibrates the same ``H`` by a wild bootstrap (here ``B = 1000``
    refits of the line):

    >>> print(f"{result.cusum_statistic:.4f} {result.cusum_critical_value:.4f} {result.cusum_p_value:.3f}")
    0.3780 1.3581 0.999
    >>> boot = passing_bablok(reference, candidate, ids, cusum_method="bootstrap", seed=0)
    >>> boot.meta.configuration["n_cusum_bootstrap"]
    1000
    >>> print(f"{boot.cusum_statistic:.4f} {boot.cusum_critical_value:.4f} {boot.cusum_p_value:.3f}")
    0.3780 0.7559 1.000

    The manuscript-literal estimator with a bootstrap interval and the
    permutation CUSUM variant:

    >>> literal = passing_bablok(
    ...     reference, candidate, ids, estimator="manuscript_literal",
    ...     ci_method="bootstrap", cusum_method="permutation",
    ...     n_bootstrap=200, n_cusum_permutations=200, seed=0,
    ... )
    >>> print(f"{literal.slope:.4f} {literal.intercept:.4f}")
    1.0300 -0.0200
    """

    _check_choice("estimator", estimator, ("passing_bablok_1983", "manuscript_literal"))
    _check_choice("ci_method", ci_method, ("analytic", "bootstrap"))
    _check_choice("cusum_method", cusum_method, ("bootstrap", "passing_bablok_1983", "permutation"))
    if cusum_method == "bootstrap" and int(n_cusum_bootstrap) < 1:
        raise ValueError(f"n_cusum_bootstrap must be >= 1, got {n_cusum_bootstrap!r}")
    if cusum_bootstrap_variance_ratio is not None and not (
        np.isfinite(cusum_bootstrap_variance_ratio) and cusum_bootstrap_variance_ratio > 0
    ):
        raise ValueError(
            f"cusum_bootstrap_variance_ratio must be positive and finite, got {cusum_bootstrap_variance_ratio!r}"
        )

    all_ids, included_ids, ref, cand = _finite_pairs(reference, candidate, sample_ids)
    n = len(ref)
    n_invalid = len(all_ids) - len(included_ids)
    status = ResultStatus.VALID if n >= 3 else ResultStatus.INSUFFICIENT_DATA
    warnings: list[str] = []

    info = _pb_estimate(ref, cand, estimator) if status is ResultStatus.VALID else None
    if info is not None and info["slopes"].shape[0] == 0:
        status = ResultStatus.UNDEFINED
        warnings.append("no usable pairwise slope (all pairs excluded)")

    configuration = {
        "estimator": estimator,
        "ci_method": ci_method,
        "ci_construction": (
            "passing_bablok_1983_rank" if ci_method == "analytic" else "paired_percentile_bootstrap"
        ),
        "confidence_level": confidence_level,
        "n_bootstrap": n_bootstrap if ci_method == "bootstrap" else None,
        "cusum_method": cusum_method,
        "cusum_p_value_method": {
            "bootstrap": "wild_bootstrap_rademacher_refit",
            "passing_bablok_1983": "asymptotic_kolmogorov_kstwobign",
            "permutation": "permutation",
        }[cusum_method],
        "cusum_alpha": cusum_alpha,
        "n_cusum_permutations": n_cusum_permutations if cusum_method == "permutation" else None,
        "n_cusum_bootstrap": int(n_cusum_bootstrap) if cusum_method == "bootstrap" else None,
    }
    if cusum_method == "bootstrap":
        configuration["cusum_bootstrap_line_reestimated_per_replicate"] = True
        configuration["cusum_bootstrap_variance_ratio"] = cusum_bootstrap_variance_ratio
        configuration["cusum_bootstrap_residual_direction"] = (
            "perpendicular"
            if cusum_bootstrap_variance_ratio is None
            else "error_covariance_conjugate (-lambda * slope, 1)"
        )

    if status is not ResultStatus.VALID:
        meta = build_meta(
            status=status,
            all_ids=all_ids,
            included_ids=included_ids,
            exclusion_reasons={"nan_or_inf": n_invalid} if n_invalid else {},
            warnings=tuple(warnings),
            method="passing_bablok",
            configuration=configuration,
            random_seed=seed,
        )
        return PassingBablokResult(
            meta=meta,
            slope=float("nan"),
            intercept=float("nan"),
            slope_ci=_NAN_PAIR,
            intercept_ci=_NAN_PAIR,
            n_pairwise_slopes=0 if info is None else int(info["slopes"].shape[0]),
            n_excluded_equal_reference_pairs=0 if info is None else info["n_excluded_equal_reference"],
            cusum_max_abs=float("nan"),
            cusum_p_value=float("nan"),
            confidence_level=confidence_level,
            ci_method=ci_method,
            estimator=estimator,
            cusum_method=cusum_method,
        )

    beta1, beta0 = info["slope"], info["intercept"]
    m = int(info["slopes"].shape[0])
    if info["index_clamped"]:
        warnings.append(
            "shifted-median index fell outside 1..N and was clamped; Passing-Bablok "
            "assumes a positive slope"
        )

    rng = np.random.default_rng(seed)
    if ci_method == "analytic":
        slope_ci, intercept_ci, ci_clamped = _pb_analytic_ci(ref, cand, info, confidence_level)
        if ci_clamped:
            warnings.append(
                "analytic confidence-interval rank fell outside 1..N and was clamped "
                "(sample too small for the nominal level)"
            )
    else:
        boot_slopes = np.full(n_bootstrap, np.nan)
        boot_intercepts = np.full(n_bootstrap, np.nan)
        ws = _PBWorkspace(n)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            boot_slopes[b], boot_intercepts[b] = ws.fit(ref[idx], cand[idx], estimator)
        slope_ci = _percentile_ci(boot_slopes, confidence_level)
        intercept_ci = _percentile_ci(boot_intercepts, confidence_level)
        configuration["n_bootstrap_failed"] = int(np.sum(~np.isfinite(boot_slopes)))

    if cusum_method in ("passing_bablok_1983", "bootstrap"):
        cus = _pb_cusum_1983(ref, cand, beta0, beta1)
        cusum_max_abs, cusum_stat = cus["max_abs"], cus["statistic"]
        configuration["cusum_n_positive"] = cus["n_pos"]
        configuration["cusum_n_negative"] = cus["n_neg"]
        configuration["cusum_normalisation"] = "sqrt(n_pos + n_neg + 1)"
        if not np.isfinite(cusum_stat):
            warnings.append(
                "CUSUM linearity test undefined: residuals are all on one side of the fitted line"
            )
        if cusum_method == "passing_bablok_1983":
            cusum_p = float(_stats.kstwobign.sf(cusum_stat)) if np.isfinite(cusum_stat) else float("nan")
            critical = float(_stats.kstwobign.isf(cusum_alpha))
            reject_lin = bool(np.isfinite(cusum_stat) and cusum_stat > critical)
        else:
            cusum_p = critical = float("nan")
            reject_lin = False
            n_boot = int(n_cusum_bootstrap)
            n_valid = 0
            if np.isfinite(cusum_stat):
                h_star = _pb_cusum_wild_bootstrap(
                    ref, cand, beta0, beta1, estimator, n_boot, rng, cusum_bootstrap_variance_ratio
                )
                h_star = h_star[np.isfinite(h_star)]
                n_valid = int(h_star.shape[0])
                n_skipped = n_boot - n_valid
                if n_skipped:
                    warnings.append(
                        f"{n_skipped} of {n_boot} CUSUM bootstrap replicates were degenerate "
                        "(undefined slope or all residuals on one side) and were skipped"
                    )
                if n_valid >= max(_PB_CUSUM_MIN_VALID, n_boot / 2):
                    # Ties are common (H is lattice-valued); count them as
                    # exceedances, with a relative tolerance for rounding.
                    n_ge = int(np.count_nonzero(h_star >= cusum_stat * (1.0 - 1e-9)))
                    cusum_p = float((1 + n_ge) / (n_valid + 1))
                    critical = float(np.quantile(h_star, 1.0 - cusum_alpha))
                    reject_lin = bool(cusum_p <= cusum_alpha)
                else:
                    warnings.append(
                        f"only {n_valid} of {n_boot} CUSUM bootstrap replicates were valid; "
                        "CUSUM p-value set to NaN"
                    )
            configuration["n_cusum_bootstrap_valid"] = n_valid
    else:
        order = np.argsort(ref, kind="stable")
        resid = (cand - (beta0 + beta1 * ref))[order]
        resid_std = np.std(resid, ddof=1) if np.std(resid, ddof=1) > 0 else 1.0
        cusum = np.cumsum(resid) / resid_std
        cusum_max_abs = float(np.max(np.abs(cusum)))
        perm_stats = np.empty(n_cusum_permutations)
        for p in range(n_cusum_permutations):
            perm_resid = rng.permutation(resid)
            perm_stats[p] = np.max(np.abs(np.cumsum(perm_resid) / resid_std))
        cusum_p = float((1 + np.sum(perm_stats >= cusum_max_abs)) / (n_cusum_permutations + 1))
        cusum_stat = cusum_max_abs
        critical = float("nan")
        reject_lin = bool(cusum_p < cusum_alpha)

    configuration.update(
        n_infinite_slopes=info["n_infinite"],
        n_excluded_minus_one_slopes=info["n_minus_one"],
        n_excluded_equal_reference_pairs=info["n_excluded_equal_reference"],
        shift_k=info["k"],
    )
    meta = build_meta(
        status=ResultStatus.VALID,
        all_ids=all_ids,
        included_ids=included_ids,
        exclusion_reasons={"nan_or_inf": n_invalid} if n_invalid else {},
        warnings=tuple(warnings),
        method="passing_bablok",
        configuration=configuration,
        random_seed=seed,
    )
    return PassingBablokResult(
        meta=meta,
        slope=beta1,
        intercept=beta0,
        slope_ci=(float(slope_ci[0]), float(slope_ci[1])),
        intercept_ci=(float(intercept_ci[0]), float(intercept_ci[1])),
        n_pairwise_slopes=m,
        n_excluded_equal_reference_pairs=info["n_excluded_equal_reference"],
        cusum_max_abs=cusum_max_abs,
        cusum_p_value=cusum_p,
        confidence_level=confidence_level,
        ci_method=ci_method,
        estimator=estimator,
        n_infinite_slopes=info["n_infinite"],
        n_excluded_minus_one_slopes=info["n_minus_one"],
        shift_k=info["k"],
        cusum_method=cusum_method,
        cusum_statistic=cusum_stat,
        cusum_critical_value=critical,
        cusum_reject_linearity=reject_lin,
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


def _ols_batch(x: NDArray[np.float64], y: NDArray[np.float64]):
    """Closed-form OLS of `y` on `x` along the last axis.

    Returns ``(intercept, slope, intercept_se, slope_se)`` (residual
    variance on ``n - 2`` degrees of freedom), NaN where ``S_xx = 0``.
    Matches :func:`scipy.stats.linregress`.
    """
    n = x.shape[-1]
    x_bar = np.mean(x, axis=-1, keepdims=True)
    y_bar = np.mean(y, axis=-1, keepdims=True)
    dx = x - x_bar
    dy = y - y_bar
    s_xx = np.sum(dx * dx, axis=-1)
    s_xy = np.sum(dx * dy, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(s_xx > 0, s_xy / np.where(s_xx > 0, s_xx, 1.0), np.nan)
        intercept = y_bar[..., 0] - slope * x_bar[..., 0]
        resid = dy - slope[..., None] * dx
        sigma2 = np.sum(resid * resid, axis=-1) / (n - 2) if n > 2 else np.full(slope.shape, np.nan)
        slope_se = np.sqrt(sigma2 / s_xx)
        intercept_se = np.sqrt(sigma2 * (1.0 / n + x_bar[..., 0] ** 2 / s_xx))
    return intercept, slope, intercept_se, slope_se


def _deming_batch(ref: NDArray[np.float64], test: NDArray[np.float64], mu: float):
    """Vectorized :func:`deming` coefficients along the last axis.

    Same formula as :func:`deming` (``reference`` = ``y``, ``test`` =
    ``x``, ``mu = test_error_variance / reference_error_variance``);
    NaN where the slope is undefined.
    """
    n = ref.shape[-1]
    y_bar = np.mean(ref, axis=-1, keepdims=True)
    x_bar = np.mean(test, axis=-1, keepdims=True)
    s_yy = np.sum((ref - y_bar) ** 2, axis=-1) / (n - 1)
    s_xx = np.sum((test - x_bar) ** 2, axis=-1) / (n - 1)
    s_xy = np.sum((ref - y_bar) * (test - x_bar), axis=-1) / (n - 1)
    disc = (s_xx - mu * s_yy) ** 2 + 4 * mu * s_xy**2
    ok = (s_xy != 0) & (disc >= 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(ok, (s_xx - mu * s_yy + np.sqrt(np.where(ok, disc, 0.0))) / (2 * np.where(ok, s_xy, 1.0)), np.nan)
    intercept = x_bar[..., 0] - slope * y_bar[..., 0]
    return intercept, slope


def _bootstrap_indices(rng: np.random.Generator, n: int, n_bootstrap: int) -> NDArray[np.int64]:
    return rng.integers(0, n, size=(n_bootstrap, n))


def _excludes(ci: tuple[float, float], null_value: float) -> bool:
    return bool(np.isfinite(ci[0]) and np.isfinite(ci[1]) and not (ci[0] <= null_value <= ci[1]))


def _concordant(ci_a: tuple[float, float], ci_b: tuple[float, float], dec_a: bool, dec_b: bool) -> bool:
    finite = all(np.isfinite(v) for v in (*ci_a, *ci_b))
    return bool(finite and dec_a == dec_b)


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
    confidence_level : float
        Nominal two-sided confidence level of both intervals; the test
        level is ``1 - confidence_level``.
    alpha_ci_analytic : tuple of (float, float)
        ``alpha +/- t_{n-2, (1+confidence_level)/2} * alpha_se``.
    alpha_ci_bootstrap : tuple of (float, float)
        Paired percentile bootstrap interval for `alpha`.
    reject_null_analytic : bool
        True if `alpha_ci_analytic` excludes 0 (equivalently
        ``alpha_p_value < 1 - confidence_level``).
    reject_null_bootstrap : bool
        True if `alpha_ci_bootstrap` excludes 0.
    analytic_bootstrap_concordant : bool
        True if both decisions are defined and agree.
    """

    alpha: float
    beta: float
    alpha_se: float
    alpha_t: float
    alpha_p_value: float
    confidence_level: float = 0.95
    alpha_ci_analytic: tuple[float, float] = _NAN_PAIR
    alpha_ci_bootstrap: tuple[float, float] = _NAN_PAIR
    reject_null_analytic: bool = False
    reject_null_bootstrap: bool = False
    analytic_bootstrap_concordant: bool = False


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
    confidence_level : float
        Nominal two-sided confidence level of both intervals.
    beta_ci_analytic : tuple of (float, float)
        ``beta +/- t_{n-2, (1+confidence_level)/2} * beta_se``.
    beta_ci_bootstrap : tuple of (float, float)
        Paired percentile bootstrap interval for `beta`.
    reject_null_analytic : bool
        True if `beta_ci_analytic` excludes 0.
    reject_null_bootstrap : bool
        True if `beta_ci_bootstrap` excludes 0.
    analytic_bootstrap_concordant : bool
        True if both decisions are defined and agree.
    """

    alpha: float
    beta: float
    beta_se: float
    beta_t: float
    beta_p_value: float
    bias_correction_applied: float | None
    confidence_level: float = 0.95
    beta_ci_analytic: tuple[float, float] = _NAN_PAIR
    beta_ci_bootstrap: tuple[float, float] = _NAN_PAIR
    reject_null_analytic: bool = False
    reject_null_bootstrap: bool = False
    analytic_bootstrap_concordant: bool = False


@dataclass(frozen=True)
class EquivalenceTest3Result(AnalysisResult):
    """Result of :func:`equivalence_test_3` (agreement).

    Joint test of H0: Deming intercept ``alpha = 0`` and slope
    ``beta = 1``, with a Bonferroni-adjusted per-parameter level (spec:
    5% global significance splits to a 97.5% confidence interval per
    parameter), by paired bootstrap (primary) and by jackknife-SE
    analytic intervals.

    Attributes
    ----------
    meta : AnalysisMeta
        Provenance and inclusion/exclusion record for this computation.
    alpha : float
        Point estimate of the Deming intercept (candidate on reference).
    beta : float
        Point estimate of the Deming slope.
    alpha_ci : tuple of (float, float)
        Paired percentile bootstrap confidence interval for `alpha`, at
        `confidence_level_per_parameter`.
    beta_ci : tuple of (float, float)
        Paired percentile bootstrap confidence interval for `beta`, at
        `confidence_level_per_parameter`.
    reject_joint_null : bool
        Bootstrap (primary) decision: True if `alpha_ci` excludes 0 or
        `beta_ci` excludes 1.
    confidence_level_per_parameter : float
        Per-parameter confidence level actually used,
        ``1 - alpha / 2`` (the Bonferroni-adjusted level).
    bias_correction_applied : float or None
        The `bias_correction` value that was subtracted from `candidate`
        before fitting, or None if none was applied.
    alpha_se_jackknife : float
        Jackknife standard error of `alpha` (Linnet 1990).
    beta_se_jackknife : float
        Jackknife standard error of `beta`.
    alpha_ci_analytic : tuple of (float, float)
        ``alpha +/- t_{n-2, 1-alpha_global/4} * alpha_se_jackknife``.
    beta_ci_analytic : tuple of (float, float)
        ``beta +/- t_{n-2, 1-alpha_global/4} * beta_se_jackknife``.
    alpha_p_value_analytic : float
        Two-sided t (``n - 2`` df) p-value for H0: ``alpha = 0`` with the
        jackknife SE; compare against ``alpha_global / 2``.
    beta_p_value_analytic : float
        Two-sided t p-value for H0: ``beta = 1`` with the jackknife SE.
    reject_joint_null_analytic : bool
        Analytic decision: True if `alpha_ci_analytic` excludes 0 or
        `beta_ci_analytic` excludes 1.
    analytic_bootstrap_concordant : bool
        True if both joint decisions are defined and agree.
    """

    alpha: float
    beta: float
    alpha_ci: tuple[float, float]
    beta_ci: tuple[float, float]
    reject_joint_null: bool
    confidence_level_per_parameter: float
    bias_correction_applied: float | None
    alpha_se_jackknife: float = float("nan")
    beta_se_jackknife: float = float("nan")
    alpha_ci_analytic: tuple[float, float] = _NAN_PAIR
    beta_ci_analytic: tuple[float, float] = _NAN_PAIR
    alpha_p_value_analytic: float = float("nan")
    beta_p_value_analytic: float = float("nan")
    reject_joint_null_analytic: bool = False
    analytic_bootstrap_concordant: bool = False


def _t_test_with_bootstrap(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    parameter: str,
    center_x: bool,
    confidence_level: float,
    n_bootstrap: int,
    seed: int | None,
) -> dict:
    """Shared analytic + bootstrap inference for Tests 1 and 2."""
    n = x.shape[0]
    x_fit = x - np.mean(x) if center_x else x
    intercept, slope, intercept_se, slope_se = (float(v) for v in _ols_batch(x_fit, y))
    est, se = (intercept, intercept_se) if parameter == "intercept" else (slope, slope_se)
    df = n - 2
    t_stat = est / se if np.isfinite(se) and se > 0 else float("nan")
    p_value = float(2 * _stats.t.sf(np.abs(t_stat), df)) if np.isfinite(t_stat) else float("nan")
    if np.isfinite(se):
        half = float(_stats.t.ppf(0.5 + confidence_level / 2.0, df)) * se
        ci_analytic = (est - half, est + half)
    else:
        ci_analytic = _NAN_PAIR

    rng = np.random.default_rng(seed)
    idx = _bootstrap_indices(rng, n, n_bootstrap)
    xb, yb = x[idx], y[idx]
    if center_x:
        xb = xb - np.mean(xb, axis=1, keepdims=True)
    b_int, b_slope, _, _ = _ols_batch(xb, yb)
    boot = b_int if parameter == "intercept" else b_slope
    ci_boot = _percentile_ci(boot, confidence_level)
    reject_a = _excludes(ci_analytic, 0.0)
    reject_b = _excludes(ci_boot, 0.0)
    return {
        "intercept": intercept,
        "slope": slope,
        "se": se,
        "t": t_stat,
        "p": p_value,
        "ci_analytic": ci_analytic,
        "ci_bootstrap": ci_boot,
        "reject_analytic": reject_a,
        "reject_bootstrap": reject_b,
        "concordant": _concordant(ci_analytic, ci_boot, reject_a, reject_b),
        "n_bootstrap_failed": int(np.sum(~np.isfinite(boot))),
    }


def equivalence_test_1(
    reference: NDArray[np.float64],
    candidate: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
    *,
    confidence_level: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int | None = None,
) -> EquivalenceTest1Result:
    """Silveira et al. equivalence Test 1: accuracy.

    Fits ``d = candidate - reference`` against the mean-centered
    reference, ``d = alpha + beta * (reference - mean(reference)) +
    nu``, and tests H0: ``alpha = 0`` (no systematic translation bias),
    reporting both the analytic t-test and a paired percentile bootstrap
    interval, as the manuscript prescribes.

    Parameters
    ----------
    reference : ndarray of float, shape (n_samples,)
        Values from the established/ground-truth method.
    candidate : ndarray of float, shape (n_samples,)
        Values from the model or method under evaluation.
    sample_ids : ndarray of str, shape (n_samples,)
        Per-sample identifiers, used for inclusion/exclusion bookkeeping.
    confidence_level : float, default 0.95
        Two-sided confidence level of the analytic and bootstrap
        intervals; the test level is ``1 - confidence_level``.
    n_bootstrap : int, default 2000
        Number of paired bootstrap resamples. Each replicate re-centres
        the resampled reference and refits the same regression.
    seed : int or None, default None
        Seed for ``np.random.default_rng`` driving the bootstrap.

    Returns
    -------
    EquivalenceTest1Result
        Fitted intercept/slope, the accuracy test statistics, both
        intervals and decisions. See :class:`EquivalenceTest1Result` for
        the full field list.

    Notes
    -----
    Closed-form OLS (numerically identical to
    :func:`scipy.stats.linregress`). The analytic interval is
    ``alpha +/- t_{n-2} * se``. The bootstrap is preferred by the
    manuscript when normality of the differences cannot be assumed;
    `analytic_bootstrap_concordant` flags disagreement between the two
    decisions, which Silveira et al. recommend interpreting jointly.
    Fewer than 3 valid pairs yields a NaN-filled result with
    ``meta.status = ResultStatus.INSUFFICIENT_DATA``; a constant
    reference yields ``ResultStatus.UNDEFINED``.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.method_comparison import equivalence_test_1
    >>> reference = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
    >>> candidate = np.array([10.6, 12.3, 13.7, 16.5, 18.1, 20.9])
    >>> ids = np.array(["s1", "s2", "s3", "s4", "s5", "s6"])
    >>> result = equivalence_test_1(reference, candidate, ids, seed=0)
    >>> print(f"{result.alpha:.4f} {result.alpha_t:.4f} {result.alpha_p_value:.4f}")
    0.3500 1.8779 0.1336
    >>> print("{:.4f} {:.4f}".format(*result.alpha_ci_analytic))
    -0.1675 0.8675
    >>> result.reject_null_analytic
    False
    """
    all_ids, included_ids, ref, cand = _finite_pairs(reference, candidate, sample_ids)
    n = len(ref)
    n_invalid = len(all_ids) - len(included_ids)
    status = ResultStatus.VALID if n >= 3 else ResultStatus.INSUFFICIENT_DATA
    if status is ResultStatus.VALID and np.ptp(ref) == 0:
        status = ResultStatus.UNDEFINED
    configuration = {
        "regression": "d ~ 1 + (reference - mean(reference))",
        "backend": "closed-form OLS (equivalent to scipy.stats.linregress)",
        "analytic_ci": "t_{n-2}",
        "bootstrap_design": "paired_percentile",
        "confidence_level": confidence_level,
        "n_bootstrap": n_bootstrap,
    }

    if status is not ResultStatus.VALID:
        meta = build_meta(
            status=status,
            all_ids=all_ids,
            included_ids=included_ids,
            exclusion_reasons={"nan_or_inf": n_invalid} if n_invalid else {},
            warnings=("reference is constant",) if status is ResultStatus.UNDEFINED else (),
            method="equivalence_test_1",
            configuration=configuration,
            random_seed=seed,
        )
        return EquivalenceTest1Result(
            meta=meta, alpha=float("nan"), beta=float("nan"),
            alpha_se=float("nan"), alpha_t=float("nan"), alpha_p_value=float("nan"),
            confidence_level=confidence_level,
        )

    res = _t_test_with_bootstrap(
        ref, cand - ref, parameter="intercept", center_x=True,
        confidence_level=confidence_level, n_bootstrap=n_bootstrap, seed=seed,
    )
    configuration["n_bootstrap_failed"] = res["n_bootstrap_failed"]
    meta = build_meta(
        status=ResultStatus.VALID,
        all_ids=all_ids,
        included_ids=included_ids,
        exclusion_reasons={"nan_or_inf": n_invalid} if n_invalid else {},
        method="equivalence_test_1",
        configuration=configuration,
        random_seed=seed,
    )
    return EquivalenceTest1Result(
        meta=meta,
        alpha=res["intercept"],
        beta=res["slope"],
        alpha_se=res["se"],
        alpha_t=res["t"],
        alpha_p_value=res["p"],
        confidence_level=confidence_level,
        alpha_ci_analytic=res["ci_analytic"],
        alpha_ci_bootstrap=res["ci_bootstrap"],
        reject_null_analytic=res["reject_analytic"],
        reject_null_bootstrap=res["reject_bootstrap"],
        analytic_bootstrap_concordant=res["concordant"],
    )


def equivalence_test_2(
    reference: NDArray[np.float64],
    candidate: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
    *,
    bias_correction: float | None = None,
    confidence_level: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int | None = None,
) -> EquivalenceTest2Result:
    """Silveira et al. equivalence Test 2: precision (Shukla 1973).

    Fits ``d = candidate - reference`` against ``candidate +
    reference``, ``d = alpha + beta * (candidate + reference) + theta``,
    and tests H0: ``beta = 0`` (equivalent to a reference/candidate
    variance ratio of 1), reporting both the analytic t-test and a paired
    percentile bootstrap interval.

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
        1's fitted ``alpha``. None applies no correction. See Notes: it
        changes only the fitted intercept, never `beta` or its p-value.
    confidence_level : float, default 0.95
        Two-sided confidence level of the analytic and bootstrap
        intervals.
    n_bootstrap : int, default 2000
        Number of paired bootstrap resamples (same regression refit per
        replicate).
    seed : int or None, default None
        Seed for ``np.random.default_rng`` driving the bootstrap.

    Returns
    -------
    EquivalenceTest2Result
        Fitted intercept/slope, the precision test statistics, both
        intervals and decisions. See :class:`EquivalenceTest2Result` for
        the full field list.

    Notes
    -----
    Closed-form OLS (numerically identical to
    :func:`scipy.stats.linregress`).

    **The bias correction cannot change the precision test.** Subtracting
    a constant :math:`c` from `candidate` shifts both the response
    :math:`d` and the regressor :math:`x + y` by :math:`-c`. An OLS slope
    (and its standard error) is invariant to adding constants to the
    regressor and the response, so `beta`, `beta_se`, `beta_t`,
    `beta_p_value` and (for the same seed) the bootstrap interval are
    unchanged; only the intercept moves, by :math:`-c(1 - \\beta)`. The
    parameter is kept for protocol fidelity and provenance.

    Fewer than 3 valid pairs yields a NaN-filled result with
    ``meta.status = ResultStatus.INSUFFICIENT_DATA``; a constant
    ``candidate + reference`` yields ``ResultStatus.UNDEFINED``.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.method_comparison import equivalence_test_2
    >>> reference = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
    >>> candidate = np.array([10.6, 12.3, 13.7, 16.5, 18.1, 20.9])
    >>> ids = np.array(["s1", "s2", "s3", "s4", "s5", "s6"])
    >>> result = equivalence_test_2(reference, candidate, ids, seed=0)
    >>> print(f"{result.beta:.4f} {result.beta_t:.4f} {result.beta_p_value:.4f}")
    0.0149 0.5596 0.6056
    >>> result.reject_null_analytic, result.reject_null_bootstrap
    (False, False)
    """
    all_ids, included_ids, ref, cand = _finite_pairs(reference, candidate, sample_ids)
    n = len(ref)
    n_invalid = len(all_ids) - len(included_ids)
    status = ResultStatus.VALID if n >= 3 else ResultStatus.INSUFFICIENT_DATA
    if bias_correction is not None:
        cand = cand - bias_correction
    if status is ResultStatus.VALID and np.ptp(cand + ref) == 0:
        status = ResultStatus.UNDEFINED
    configuration = {
        "regression": "d ~ 1 + (candidate + reference)",
        "backend": "closed-form OLS (equivalent to scipy.stats.linregress)",
        "bias_correction_applied": bias_correction,
        "analytic_ci": "t_{n-2}",
        "bootstrap_design": "paired_percentile",
        "confidence_level": confidence_level,
        "n_bootstrap": n_bootstrap,
    }

    if status is not ResultStatus.VALID:
        meta = build_meta(
            status=status,
            all_ids=all_ids,
            included_ids=included_ids,
            exclusion_reasons={"nan_or_inf": n_invalid} if n_invalid else {},
            warnings=("candidate + reference is constant",) if status is ResultStatus.UNDEFINED else (),
            method="equivalence_test_2",
            configuration=configuration,
            random_seed=seed,
        )
        return EquivalenceTest2Result(
            meta=meta, alpha=float("nan"), beta=float("nan"),
            beta_se=float("nan"), beta_t=float("nan"), beta_p_value=float("nan"),
            bias_correction_applied=bias_correction,
            confidence_level=confidence_level,
        )

    res = _t_test_with_bootstrap(
        cand + ref, cand - ref, parameter="slope", center_x=False,
        confidence_level=confidence_level, n_bootstrap=n_bootstrap, seed=seed,
    )
    configuration["n_bootstrap_failed"] = res["n_bootstrap_failed"]
    meta = build_meta(
        status=ResultStatus.VALID,
        all_ids=all_ids,
        included_ids=included_ids,
        exclusion_reasons={"nan_or_inf": n_invalid} if n_invalid else {},
        method="equivalence_test_2",
        configuration=configuration,
        random_seed=seed,
    )
    return EquivalenceTest2Result(
        meta=meta,
        alpha=res["intercept"],
        beta=res["slope"],
        beta_se=res["se"],
        beta_t=res["t"],
        beta_p_value=res["p"],
        bias_correction_applied=bias_correction,
        confidence_level=confidence_level,
        beta_ci_analytic=res["ci_analytic"],
        beta_ci_bootstrap=res["ci_bootstrap"],
        reject_null_analytic=res["reject_analytic"],
        reject_null_bootstrap=res["reject_bootstrap"],
        analytic_bootstrap_concordant=res["concordant"],
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
    r"""Silveira et al. equivalence Test 3: agreement.

    Joint test of H0: Deming intercept ``alpha = 0`` and slope ``beta =
    1`` for the :func:`deming` regression of `candidate` on `reference`,
    using a per-parameter confidence level ``1 - alpha / 2`` (Bonferroni
    correction over the two joint-null parameters). Both a paired
    bootstrap decision (primary) and an analytic decision from jackknife
    standard errors are reported.

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
        estimate, every bootstrap replicate and every jackknife fit),
        analogous to `equivalence_test_2`'s `bias_correction`.
    alpha : float, default 0.05
        Global two-sided significance level for the joint test. Split
        Bonferroni-style across the two joint-null parameters to a
        per-parameter confidence level of ``1 - alpha / 2``.
    n_bootstrap : int, default 2000
        Number of paired bootstrap resamples used to build the
        bootstrap confidence intervals.
    seed : int or None, default None
        Seed for ``np.random.default_rng`` driving the bootstrap
        resampling; None draws fresh entropy.

    Returns
    -------
    EquivalenceTest3Result
        Point estimates, bootstrap and analytic confidence intervals,
        and both joint-null decisions. See :class:`EquivalenceTest3Result`
        for the full field list.

    Notes
    -----
    The point estimate is obtained by calling :func:`deming` (with
    `reference` as ``y`` and `candidate` as ``x``; see that function's
    Notes for the regression-direction convention and the
    manuscript-consistency caveat that governs its slope formula); the
    bootstrap replicates and jackknife fits use the identical closed-form
    formula. If the point-estimate Deming fit is not
    ``ResultStatus.VALID`` (e.g. undefined slope, insufficient data),
    that status is propagated unchanged and no resampling is run.

    **Analytic intervals** follow Linnet (1990): leave-one-out Deming fits
    :math:`\theta_{(i)}` give
    :math:`\mathrm{SE} = \sqrt{\tfrac{n-1}{n}\sum_i (\theta_{(i)} -
    \bar\theta_{(\cdot)})^2}` and the interval
    :math:`\hat\theta \pm t_{n-2,\,1-\alpha/4}\,\mathrm{SE}` (the
    Bonferroni per-parameter level). They require ``n >= 3``.

    **Decision.** `reject_joint_null` is the *bootstrap* decision, since
    the manuscript treats the bootstrap as the preferred basis for
    inference; `reject_joint_null_analytic` is reported alongside and
    `analytic_bootstrap_concordant` flags disagreement.

    **Bias correction.** Subtracting a constant from `candidate` (the
    Deming ``x``) leaves all sample variances and the covariance
    unchanged, so the Deming slope (and its bootstrap/jackknife
    distribution) is invariant; only the intercept shifts, by exactly
    ``-bias_correction``.

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
    >>> result.reject_joint_null_analytic, result.analytic_bootstrap_concordant
    (False, True)
    """

    if bias_correction is not None:
        candidate = np.asarray(candidate, dtype=np.float64) - bias_correction

    per_param_level = 1.0 - alpha / 2.0
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
            alpha_ci=_NAN_PAIR,
            beta_ci=_NAN_PAIR,
            reject_joint_null=False,
            confidence_level_per_parameter=per_param_level,
            bias_correction_applied=bias_correction,
        )

    all_ids, included_ids, ref, cand = _finite_pairs(reference, candidate, sample_ids)
    n = len(ref)
    mu = test_error_variance / reference_error_variance
    warnings: list[str] = []

    rng = np.random.default_rng(seed)
    idx = _bootstrap_indices(rng, n, n_bootstrap)
    boot_alpha, boot_beta = _deming_batch(ref[idx], cand[idx], mu)
    alpha_ci = _percentile_ci(boot_alpha, per_param_level)
    beta_ci = _percentile_ci(boot_beta, per_param_level)
    reject = _excludes(alpha_ci, 0.0) or _excludes(beta_ci, 1.0)
    n_boot_failed = int(np.sum(~np.isfinite(boot_beta)))

    alpha_se = beta_se = float("nan")
    alpha_ci_an = beta_ci_an = _NAN_PAIR
    alpha_p_an = beta_p_an = float("nan")
    if n >= 3:
        loo = np.array([np.delete(np.arange(n), i) for i in range(n)])
        jk_alpha, jk_beta = _deming_batch(ref[loo], cand[loo], mu)
        if np.all(np.isfinite(jk_alpha)) and np.all(np.isfinite(jk_beta)):
            alpha_se = float(np.sqrt((n - 1) / n * np.sum((jk_alpha - jk_alpha.mean()) ** 2)))
            beta_se = float(np.sqrt((n - 1) / n * np.sum((jk_beta - jk_beta.mean()) ** 2)))
            df = n - 2
            t_crit = float(_stats.t.ppf(1.0 - alpha / 4.0, df))
            alpha_ci_an = (point.intercept - t_crit * alpha_se, point.intercept + t_crit * alpha_se)
            beta_ci_an = (point.slope - t_crit * beta_se, point.slope + t_crit * beta_se)
            with np.errstate(divide="ignore", invalid="ignore"):
                t_a = point.intercept / alpha_se if alpha_se > 0 else float("nan")
                t_b = (point.slope - 1.0) / beta_se if beta_se > 0 else float("nan")
            alpha_p_an = float(2 * _stats.t.sf(abs(t_a), df)) if np.isfinite(t_a) else float("nan")
            beta_p_an = float(2 * _stats.t.sf(abs(t_b), df)) if np.isfinite(t_b) else float("nan")
        else:
            warnings.append("a leave-one-out Deming fit was undefined; analytic intervals not computed")
    else:
        warnings.append("analytic (jackknife) intervals need at least 3 valid pairs")
    reject_an = _excludes(alpha_ci_an, 0.0) or _excludes(beta_ci_an, 1.0)
    finite_all = all(np.isfinite(v) for v in (*alpha_ci, *beta_ci, *alpha_ci_an, *beta_ci_an))
    concordant = bool(finite_all and reject == reject_an)

    meta = build_meta(
        status=ResultStatus.VALID,
        all_ids=all_ids,
        included_ids=included_ids,
        warnings=tuple(warnings),
        method="equivalence_test_3",
        configuration={
            "ci_method": "bootstrap_percentile",
            "decision_basis": "bootstrap",
            "n_bootstrap": n_bootstrap,
            "n_bootstrap_failed": n_boot_failed,
            "analytic_ci": "jackknife SE (Linnet 1990), t_{n-2, 1-alpha/4}",
            "bonferroni_global_alpha": alpha,
            "confidence_level_per_parameter": per_param_level,
            "bias_correction_applied": bias_correction,
            "reference_error_variance": reference_error_variance,
            "test_error_variance": test_error_variance,
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
        alpha_se_jackknife=alpha_se,
        beta_se_jackknife=beta_se,
        alpha_ci_analytic=alpha_ci_an,
        beta_ci_analytic=beta_ci_an,
        alpha_p_value_analytic=alpha_p_an,
        beta_p_value_analytic=beta_p_an,
        reject_joint_null_analytic=reject_an,
        analytic_bootstrap_concordant=concordant,
    )
