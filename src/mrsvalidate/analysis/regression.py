"""Regression and association metrics for reference/prediction pairs.

Implements the regression- and correlation-based metrics from the
validation spec (eqs. 33-42) and manuscript: Pearson's r, a
median-centered analogue, direct and adjusted :math:`R^2`, two
"concise score" formulations, and Theil-Sen robust regression.

Notes
-----
Every function in this module follows the same ``(reference,
prediction, sample_ids)`` argument convention as
:mod:`mrsvalidate.analysis.errors`: ``prediction`` is the model
output, ``reference`` is ground truth, and residuals (where reported)
are ``prediction - reference``.

**Theil-Sen direction convention.** The manuscript's Theil-Sen
equation (``y = m*x + b``) inherits the same-section convention
``x = prediction``, ``y = reference`` used for its :math:`R^2`
derivation immediately above it, so :func:`theil_sen` wraps
``scipy.stats.theilslopes(reference, prediction, method="joint")`` --
i.e. it estimates *reference as a function of prediction*, not the
other way around. This is recorded explicitly in the result's
``method_metadata`` because it is easy to assume the opposite (as
most "predicted vs. true" scatter plots do), and the manuscript
equation does not restate the convention locally.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats as _stats

from ..results import AnalysisResult, ResultStatus, ScalarMetricResult, build_meta
from ._shared import meta_for, valid_subset

__all__ = [
    "CorrelationResult",
    "RegressionResult",
    "TheilSenResult",
    "adjusted_r_squared",
    "concise_score_manuscript",
    "concise_score_original",
    "median_centered_correlation",
    "pearson_r",
    "r_squared_direct",
    "regression_metrics",
    "theil_sen",
]


@dataclass(frozen=True)
class CorrelationResult(AnalysisResult):
    """Result of a pairwise correlation statistic.

    Returned by :func:`pearson_r` and :func:`median_centered_correlation`.

    Attributes
    ----------
    meta : AnalysisMeta
        Provenance/inclusion record describing which samples were used
        and why any were excluded.
    r : float
        The correlation coefficient. ``nan`` when ``meta.status`` is
        not ``ResultStatus.VALID`` (e.g. insufficient data or a
        constant vector).
    p_value : float or None
        Two-sided p-value for the null hypothesis of no correlation.
        ``None`` when the statistic carries no valid parametric
        significance test, as for :func:`median_centered_correlation`.
    """

    r: float
    p_value: float | None


@dataclass(frozen=True)
class TheilSenResult(AnalysisResult):
    """Result of a Theil-Sen robust linear fit.

    Returned by :func:`theil_sen`.

    Attributes
    ----------
    meta : AnalysisMeta
        Provenance/inclusion record.
    slope : float
        Theil-Sen slope estimate. See Notes for the fit direction
        convention.
    intercept : float
        Theil-Sen intercept estimate, paired with `slope`.
    slope_low : float
        Lower bound of the slope confidence interval at the requested
        confidence level.
    slope_high : float
        Upper bound of the slope confidence interval at the requested
        confidence level.

    Notes
    -----
    The fit direction is ``reference = slope * prediction +
    intercept`` (see the module docstring): `slope` and `intercept`
    describe reference as a function of prediction, not the reverse.
    """

    slope: float
    intercept: float
    slope_low: float
    slope_high: float


@dataclass(frozen=True)
class RegressionResult(AnalysisResult):
    """Aggregate regression/association result bundling every metric in
    this module.

    Returned by :func:`regression_metrics`; plotting code should
    consume this result rather than recomputing any of its components.

    Attributes
    ----------
    meta : AnalysisMeta
        Provenance/inclusion record.
    slope : float
        Theil-Sen slope (``reference = slope * prediction +
        intercept``; see the module docstring).
    intercept : float
        Theil-Sen intercept paired with `slope`.
    r : float
        Pearson's r between reference and prediction.
    r_pvalue : float or None
        Two-sided p-value for Pearson's r.
    r_squared_direct : float
        Direct :math:`R^2`, see :func:`r_squared_direct`.
    adjusted_r_squared : float
        Adjusted :math:`R^2`, see :func:`adjusted_r_squared`.
    concise_score_original : float
        Original Rizzo et al. concise score, see
        :func:`concise_score_original`.
    concise_score_manuscript : float
        Manuscript concise score, see :func:`concise_score_manuscript`.
    x_range_for_line : tuple of float
        ``(min, max)`` of the reference values, for drawing the fitted
        line across the observed range (``x_range_for_line_axis`` in
        `method_metadata` records that this range is on the reference
        axis).
    residuals : ndarray of float
        Per-sample ``prediction - reference`` residuals, in the same
        order as ``meta.included_ids``.
    confidence_intervals : mapping of str to tuple of float
        Named confidence intervals; currently only
        ``"theil_sen_slope"``.
    method_metadata : mapping of str to Any
        Per-submetric configuration dictionaries, retained for
        provenance and debugging.
    reference : ndarray of float
        Raw (aligned, finite) reference values, in the same order as
        ``meta.included_ids`` -- retained so plotting code can render
        the scatter cloud without recomputing anything.
    prediction : ndarray of float
        Raw (aligned, finite) prediction values, in the same order as
        ``meta.included_ids``.
    """

    slope: float
    intercept: float
    r: float
    r_pvalue: float | None
    r_squared_direct: float
    adjusted_r_squared: float
    concise_score_original: float
    concise_score_manuscript: float
    x_range_for_line: tuple[float, float]
    residuals: NDArray[np.float64]
    confidence_intervals: Mapping[str, tuple[float, float]]
    method_metadata: Mapping[str, Any]
    #: Raw (aligned, finite) reference/prediction values, in the same
    #: order as ``meta.included_ids`` — retained so plotting code can
    #: render the scatter cloud without recomputing anything (spec:
    #: plots consume result objects, not raw arrays passed separately).
    reference: NDArray[np.float64]
    prediction: NDArray[np.float64]


def pearson_r(
    reference: NDArray[np.float64],
    prediction: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
) -> CorrelationResult:
    """Pearson product-moment correlation coefficient.

    Wraps ``scipy.stats.pearsonr`` rather than reproducing its
    inference code locally, so that the reported p-value is backed by
    SciPy's tested implementation.

    Parameters
    ----------
    reference : ndarray of float
        Ground-truth values, one per sample.
    prediction : ndarray of float
        Model-predicted values, aligned with `reference`.
    sample_ids : ndarray of str
        Sample identifiers aligned with `reference` and `prediction`,
        used to build the inclusion/exclusion provenance record.

    Returns
    -------
    CorrelationResult
        `r` is Pearson's correlation coefficient and `p_value` its
        two-sided significance; both are ``nan``/``None`` when
        `meta.status` is not ``VALID`` (fewer than two valid samples,
        or either input is a constant vector).

    Notes
    -----
    Pearson's r is undefined, not merely unavailable, when `reference`
    or `prediction` has zero variance; that case is reported through
    ``ResultStatus.UNDEFINED`` rather than raising or returning a
    fabricated value.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.regression import pearson_r
    >>> ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> pred = np.array([1.1, 2.0, 2.9, 4.3, 4.7])
    >>> ids = np.array(["s1", "s2", "s3", "s4", "s5"])
    >>> result = pearson_r(ref, pred, ids)
    >>> round(result.r, 4)
    0.9904
    >>> result.meta.status.value
    'valid'

    A constant vector makes r undefined rather than raising:

    >>> flat = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    >>> undefined = pearson_r(ref, flat, ids)
    >>> undefined.meta.status.value
    'undefined'
    """

    all_ids, report, (ref, pred) = valid_subset(sample_ids, reference, prediction)
    meta_base = meta_for(all_ids=all_ids, report=report, method="pearson_r", min_n=2)
    if meta_base.status is not ResultStatus.VALID:
        return CorrelationResult(meta=meta_base, r=float("nan"), p_value=None)

    if np.std(ref) == 0 or np.std(pred) == 0:
        meta = meta_for(
            all_ids=all_ids,
            report=report,
            method="pearson_r",
            status=ResultStatus.UNDEFINED,
            warnings=("Pearson r undefined: a constant vector has zero variance",),
        )
        return CorrelationResult(meta=meta, r=float("nan"), p_value=None)

    result = _stats.pearsonr(ref, pred)
    meta = meta_for(
        all_ids=all_ids,
        report=report,
        method="pearson_r",
        configuration={"backend": "scipy.stats.pearsonr"},
    )
    return CorrelationResult(meta=meta, r=float(result.statistic), p_value=float(result.pvalue))


def median_centered_correlation(
    reference: NDArray[np.float64],
    prediction: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
) -> CorrelationResult:
    """Manuscript's median-centered correlation, :math:`\\tilde{r}`.

    Both arrays are centered on their own medians (rather than their
    means, as in Pearson's r) before computing the normalized inner
    product.

    Parameters
    ----------
    reference : ndarray of float
        Ground-truth values, one per sample.
    prediction : ndarray of float
        Model-predicted values, aligned with `reference`.
    sample_ids : ndarray of str
        Sample identifiers aligned with `reference` and `prediction`,
        used to build the inclusion/exclusion provenance record.

    Returns
    -------
    CorrelationResult
        `r` is the median-centered correlation coefficient. `p_value`
        is always ``None``: Pearson's parametric significance test is
        not valid for this median-centered statistic, so no p-value is
        reported rather than one computed under an inapplicable
        assumption.

    Notes
    -----
    Deliberately carries no p-value, for the reason given above under
    Returns.

    .. math::

        \\tilde{r} = \\frac{\\sum_i (x_i - \\mathrm{med}(x))(y_i -
        \\mathrm{med}(y))}
        {\\sqrt{\\sum_i (x_i - \\mathrm{med}(x))^2 \\sum_i (y_i -
        \\mathrm{med}(y))^2}}

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.regression import median_centered_correlation
    >>> ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> pred = np.array([1.1, 2.0, 2.9, 4.3, 4.7])
    >>> ids = np.array(["s1", "s2", "s3", "s4", "s5"])
    >>> result = median_centered_correlation(ref, pred, ids)
    >>> round(result.r, 4)
    0.9878
    >>> result.p_value is None
    True
    """

    all_ids, report, (ref, pred) = valid_subset(sample_ids, reference, prediction)
    meta_base = meta_for(
        all_ids=all_ids, report=report, method="median_centered_correlation", min_n=2
    )
    if meta_base.status is not ResultStatus.VALID:
        return CorrelationResult(meta=meta_base, r=float("nan"), p_value=None)

    ref_c = ref - np.median(ref)
    pred_c = pred - np.median(pred)
    denom = np.sqrt(np.sum(ref_c**2) * np.sum(pred_c**2))
    if denom == 0:
        meta = meta_for(
            all_ids=all_ids,
            report=report,
            method="median_centered_correlation",
            status=ResultStatus.UNDEFINED,
            warnings=("median-centered correlation undefined: zero denominator",),
        )
        return CorrelationResult(meta=meta, r=float("nan"), p_value=None)

    value = float(np.sum(ref_c * pred_c) / denom)
    meta = meta_for(
        all_ids=all_ids, report=report, method="median_centered_correlation"
    )
    return CorrelationResult(meta=meta, r=value, p_value=None)


def r_squared_direct(
    reference: NDArray[np.float64],
    prediction: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
) -> ScalarMetricResult:
    """Direct coefficient of determination, :math:`R^2 = 1 -
    SS_{res}/SS_{tot}`.

    Parameters
    ----------
    reference : ndarray of float
        Ground-truth values, one per sample; used for both `SS_tot`
        and `SS_res`.
    prediction : ndarray of float
        Model-predicted values, aligned with `reference`.
    sample_ids : ndarray of str
        Sample identifiers aligned with `reference` and `prediction`,
        used to build the inclusion/exclusion provenance record.

    Returns
    -------
    ScalarMetricResult
        `value` is :math:`R^2`; may legitimately be negative (when
        predictions are worse than the reference mean) and is never
        clipped to zero here.

    Notes
    -----
    :math:`R^2` is undefined, not merely unavailable, when `reference`
    is constant (``SS_tot == 0``); that case is reported through
    ``ResultStatus.UNDEFINED``.

    .. math::

        R^2 = 1 - \\frac{\\sum_i (\\mathrm{reference}_i -
        \\mathrm{prediction}_i)^2}
        {\\sum_i (\\mathrm{reference}_i - \\overline{\\mathrm{reference}})^2}

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.regression import r_squared_direct
    >>> ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> pred = np.array([1.1, 2.0, 2.9, 4.3, 4.7])
    >>> ids = np.array(["s1", "s2", "s3", "s4", "s5"])
    >>> result = r_squared_direct(ref, pred, ids)
    >>> round(result.value, 4)
    0.98

    A constant reference makes :math:`R^2` undefined rather than a
    division-by-zero value:

    >>> flat_ref = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
    >>> undefined = r_squared_direct(flat_ref, pred, ids)
    >>> undefined.meta.status.value
    'undefined'
    """

    all_ids, report, (ref, pred) = valid_subset(sample_ids, reference, prediction)
    meta_base = meta_for(all_ids=all_ids, report=report, method="r_squared_direct", min_n=2)
    if meta_base.status is not ResultStatus.VALID:
        return ScalarMetricResult(meta=meta_base, value=float("nan"))

    ss_tot = np.sum((ref - np.mean(ref)) ** 2)
    if ss_tot == 0:
        meta = meta_for(
            all_ids=all_ids,
            report=report,
            method="r_squared_direct",
            status=ResultStatus.UNDEFINED,
            warnings=("R^2 undefined: reference is constant (SS_tot == 0)",),
        )
        return ScalarMetricResult(meta=meta, value=float("nan"))

    ss_res = np.sum((ref - pred) ** 2)
    value = float(1.0 - ss_res / ss_tot)
    meta = meta_for(all_ids=all_ids, report=report, method="r_squared_direct")
    return ScalarMetricResult(meta=meta, value=value)


def adjusted_r_squared(
    reference: NDArray[np.float64],
    prediction: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
    *,
    n_predictors: int = 1,
) -> ScalarMetricResult:
    """Adjusted coefficient of determination.

    Rescales :func:`r_squared_direct` by the degrees of freedom of an
    auxiliary regression, :math:`R^2_{adj} = 1 - (1 - R^2)(n-1)/(n-p-1)`.

    Parameters
    ----------
    reference : ndarray of float
        Ground-truth values, one per sample.
    prediction : ndarray of float
        Model-predicted values, aligned with `reference`.
    sample_ids : ndarray of str
        Sample identifiers aligned with `reference` and `prediction`,
        used to build the inclusion/exclusion provenance record.
    n_predictors : int, optional
        Number of predictors (``p``) in the auxiliary regression used
        to evaluate model outputs -- typically 1 for a per-metabolite
        regression. This is **not** the number of trainable
        neural-network parameters. Default is 1.

    Returns
    -------
    ScalarMetricResult
        `value` is :math:`R^2_{adj}`; ``nan`` when `meta.status` is
        not ``VALID``, including when too few degrees of freedom
        remain (``n - n_predictors - 1 <= 0``), in which case the
        status is ``ResultStatus.INSUFFICIENT_DATA``.

    Notes
    -----
    .. math::

        R^2_{adj} = 1 - (1 - R^2) \\frac{n - 1}{n - p - 1}

    where :math:`R^2` is the direct :math:`R^2` from
    :func:`r_squared_direct`, :math:`n` is the number of valid samples,
    and :math:`p` is `n_predictors`.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.regression import adjusted_r_squared
    >>> ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> pred = np.array([1.1, 2.0, 2.9, 4.3, 4.7])
    >>> ids = np.array(["s1", "s2", "s3", "s4", "s5"])
    >>> result = adjusted_r_squared(ref, pred, ids, n_predictors=1)
    >>> round(result.value, 4)
    0.9733
    """

    r2_result = r_squared_direct(reference, prediction, sample_ids)
    if r2_result.meta.status is not ResultStatus.VALID:
        return r2_result

    n = r2_result.meta.n_used
    df_res = n - n_predictors - 1
    if df_res <= 0:
        meta = build_meta(
            status=ResultStatus.INSUFFICIENT_DATA,
            all_ids=r2_result.meta.included_ids + r2_result.meta.excluded_ids,
            included_ids=r2_result.meta.included_ids,
            exclusion_reasons=r2_result.meta.exclusion_reasons,
            method="adjusted_r_squared",
            configuration={"n_predictors": n_predictors},
            warnings=(f"insufficient degrees of freedom: n={n}, p={n_predictors}",),
        )
        return ScalarMetricResult(meta=meta, value=float("nan"))

    value = float(1.0 - (1.0 - r2_result.value) * (n - 1) / df_res)
    meta = build_meta(
        status=ResultStatus.VALID,
        all_ids=r2_result.meta.included_ids + r2_result.meta.excluded_ids,
        included_ids=r2_result.meta.included_ids,
        exclusion_reasons=r2_result.meta.exclusion_reasons,
        method="adjusted_r_squared",
        configuration={"n_predictors": n_predictors},
    )
    return ScalarMetricResult(meta=meta, value=value)


def concise_score_original(
    reference: NDArray[np.float64],
    prediction: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
) -> ScalarMetricResult:
    """Original Rizzo et al. concise score.

    Computes ``a * R^2_direct``, where ``a`` is the ordinary
    least-squares slope of prediction regressed on reference.

    Parameters
    ----------
    reference : ndarray of float
        Ground-truth values, one per sample; the independent variable
        of the internal OLS fit.
    prediction : ndarray of float
        Model-predicted values, aligned with `reference`; the
        dependent variable of the internal OLS fit.
    sample_ids : ndarray of str
        Sample identifiers aligned with `reference` and `prediction`,
        used to build the inclusion/exclusion provenance record.

    Returns
    -------
    ScalarMetricResult
        `value` is ``a * R^2_direct``. Unbounded and scale-dependent
        -- see :func:`concise_score_manuscript` for the bounded
        replacement.

    Notes
    -----
    ``a`` is the OLS slope from ``scipy.stats.linregress(reference,
    prediction)``, recorded in the result's
    ``meta.configuration["slope_source"]``. Because ``a`` is
    unbounded, this score is unbounded and not comparable across
    metabolites with different scales.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.regression import concise_score_original
    >>> ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> pred = np.array([1.1, 2.0, 2.9, 4.3, 4.7])
    >>> ids = np.array(["s1", "s2", "s3", "s4", "s5"])
    >>> result = concise_score_original(ref, pred, ids)
    >>> round(result.value, 4)
    0.931
    """

    all_ids, report, (ref, pred) = valid_subset(sample_ids, reference, prediction)
    meta_base = meta_for(
        all_ids=all_ids, report=report, method="concise_score_original", min_n=2
    )
    if meta_base.status is not ResultStatus.VALID:
        return ScalarMetricResult(meta=meta_base, value=float("nan"))

    r2 = r_squared_direct(reference, prediction, sample_ids)
    if r2.meta.status is not ResultStatus.VALID:
        return ScalarMetricResult(meta=r2.meta, value=float("nan"))

    ols = _stats.linregress(ref, pred)
    meta = meta_for(
        all_ids=all_ids,
        report=report,
        method="concise_score_original",
        configuration={"slope_source": "ols(prediction ~ reference)"},
    )
    return ScalarMetricResult(meta=meta, value=float(ols.slope * r2.value))


def concise_score_manuscript(
    reference: NDArray[np.float64],
    prediction: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
) -> ScalarMetricResult:
    """Manuscript's new concise score, :math:`r^3 := r \\cdot \\max(0,
    R^2_{direct})`.

    Parameters
    ----------
    reference : ndarray of float
        Ground-truth values, one per sample.
    prediction : ndarray of float
        Model-predicted values, aligned with `reference`.
    sample_ids : ndarray of str
        Sample identifiers aligned with `reference` and `prediction`,
        used to build the inclusion/exclusion provenance record.

    Returns
    -------
    ScalarMetricResult
        `value` is ``r * max(0, R2_direct)``, where ``r`` is Pearson's
        r (:func:`pearson_r`) and ``R2_direct`` is the direct
        :math:`R^2` (:func:`r_squared_direct`).

    Notes
    -----
    Both ``r`` and :math:`R^2` are computed independently via
    :func:`pearson_r` and :func:`r_squared_direct` -- this is **not**
    the algebraic ``r**3``, and must never be replaced with it.

    .. math::

        r^3 := r \\cdot \\max(0, R^2_{direct})

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.regression import concise_score_manuscript
    >>> ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> pred = np.array([1.1, 2.0, 2.9, 4.3, 4.7])
    >>> ids = np.array(["s1", "s2", "s3", "s4", "s5"])
    >>> result = concise_score_manuscript(ref, pred, ids)
    >>> round(result.value, 4)
    0.9706
    """

    r_result = pearson_r(reference, prediction, sample_ids)
    r2_result = r_squared_direct(reference, prediction, sample_ids)

    if r_result.meta.status is not ResultStatus.VALID or (
        r2_result.meta.status is not ResultStatus.VALID
    ):
        status = (
            r_result.meta.status
            if r_result.meta.status is not ResultStatus.VALID
            else r2_result.meta.status
        )
        meta = build_meta(
            status=status,
            all_ids=r_result.meta.included_ids + r_result.meta.excluded_ids,
            included_ids=(),
            method="concise_score_manuscript",
        )
        return ScalarMetricResult(meta=meta, value=float("nan"))

    value = float(r_result.r * max(0.0, r2_result.value))
    meta = build_meta(
        status=ResultStatus.VALID,
        all_ids=r_result.meta.included_ids + r_result.meta.excluded_ids,
        included_ids=r_result.meta.included_ids,
        exclusion_reasons=r_result.meta.exclusion_reasons,
        method="concise_score_manuscript",
        configuration={"formula": "r * max(0, R2_direct)", "not_algebraic_r_cubed": True},
    )
    return ScalarMetricResult(meta=meta, value=value)


def theil_sen(
    reference: NDArray[np.float64],
    prediction: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
    *,
    alpha: float = 0.95,
) -> TheilSenResult:
    """Theil-Sen robust slope and intercept.

    Wraps ``scipy.stats.theilslopes(reference, prediction,
    method="joint")`` -- see the module docstring for the fit
    direction convention.

    Parameters
    ----------
    reference : ndarray of float
        Ground-truth values, one per sample; fit as the dependent
        variable (``y``) of the Theil-Sen line.
    prediction : ndarray of float
        Model-predicted values, aligned with `reference`; fit as the
        independent variable (``x``) of the Theil-Sen line.
    sample_ids : ndarray of str
        Sample identifiers aligned with `reference` and `prediction`,
        used to build the inclusion/exclusion provenance record.
    alpha : float, optional
        Confidence level for the slope confidence interval, passed
        through to ``scipy.stats.theilslopes``. Default is 0.95.

    Returns
    -------
    TheilSenResult
        `slope` and `intercept` describe the fitted line
        ``reference = slope * prediction + intercept``; `slope_low`
        and `slope_high` bound the slope at the requested `alpha`.
        `meta.configuration["direction"]` records this same
        convention explicitly.

    Notes
    -----
    The fit direction is ``reference = slope * prediction +
    intercept``, i.e. it estimates *reference as a function of
    prediction* -- see the module docstring for why this is easy to
    get backwards.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.regression import theil_sen
    >>> ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> pred = np.array([1.1, 2.0, 2.9, 4.3, 4.7])
    >>> ids = np.array(["s1", "s2", "s3", "s4", "s5"])
    >>> result = theil_sen(ref, pred, ids)
    >>> round(result.slope, 4)
    1.1111
    >>> round(result.intercept, 4)
    -0.2222
    >>> result.meta.configuration["direction"]
    'reference = slope * prediction + intercept'
    """

    all_ids, report, (ref, pred) = valid_subset(sample_ids, reference, prediction)
    meta_base = meta_for(all_ids=all_ids, report=report, method="theil_sen", min_n=2)
    if meta_base.status is not ResultStatus.VALID:
        return TheilSenResult(
            meta=meta_base,
            slope=float("nan"),
            intercept=float("nan"),
            slope_low=float("nan"),
            slope_high=float("nan"),
        )

    slope, intercept, low, high = _stats.theilslopes(ref, pred, alpha=alpha, method="joint")
    meta = meta_for(
        all_ids=all_ids,
        report=report,
        method="theil_sen",
        configuration={
            "backend": "scipy.stats.theilslopes",
            "intercept_method": "joint",
            "alpha": alpha,
            "direction": "reference = slope * prediction + intercept",
        },
    )
    return TheilSenResult(
        meta=meta,
        slope=float(slope),
        intercept=float(intercept),
        slope_low=float(low),
        slope_high=float(high),
    )


def regression_metrics(
    reference: NDArray[np.float64],
    prediction: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
    *,
    n_predictors: int = 1,
    theilsen_alpha: float = 0.95,
) -> RegressionResult:
    """Assemble the full regression/association result for one metabolite.

    Computes every metric in this module -- Pearson's r, direct and
    adjusted :math:`R^2`, both concise scores, and Theil-Sen -- and
    bundles them into a single :class:`RegressionResult`.

    Parameters
    ----------
    reference : ndarray of float
        Ground-truth values, one per sample.
    prediction : ndarray of float
        Model-predicted values, aligned with `reference`.
    sample_ids : ndarray of str
        Sample identifiers aligned with `reference` and `prediction`,
        used to build the inclusion/exclusion provenance record.
    n_predictors : int, optional
        Number of predictors passed through to
        :func:`adjusted_r_squared`. Default is 1.
    theilsen_alpha : float, optional
        Confidence level passed through to :func:`theil_sen`. Default
        is 0.95.

    Returns
    -------
    RegressionResult
        Bundles `slope`/`intercept` (Theil-Sen), `r`/`r_pvalue`
        (Pearson), `r_squared_direct`, `adjusted_r_squared`, both
        concise scores, `residuals`, and the aligned `reference`/
        `prediction` arrays plus per-submetric `method_metadata`.
        ``meta.status`` reflects the worst status among the
        constituent metrics; when data are insufficient, every numeric
        field is ``nan``/empty.

    Notes
    -----
    Plotting code should consume this result rather than recomputing
    any of its components (see ``mrsvalidate.viz``).

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.regression import regression_metrics
    >>> ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> pred = np.array([1.1, 2.0, 2.9, 4.3, 4.7])
    >>> ids = np.array(["s1", "s2", "s3", "s4", "s5"])
    >>> result = regression_metrics(ref, pred, ids)
    >>> round(result.r, 4)
    0.9904
    >>> round(result.r_squared_direct, 4)
    0.98
    >>> result.x_range_for_line
    (1.0, 5.0)
    >>> result.method_metadata["x_range_for_line_axis"]
    'reference'
    """

    all_ids, report, (ref, pred) = valid_subset(sample_ids, reference, prediction)
    meta_base = meta_for(all_ids=all_ids, report=report, method="regression_metrics", min_n=2)
    if meta_base.status is not ResultStatus.VALID:
        empty = np.asarray([], dtype=np.float64)
        return RegressionResult(
            meta=meta_base,
            slope=float("nan"),
            intercept=float("nan"),
            r=float("nan"),
            r_pvalue=None,
            r_squared_direct=float("nan"),
            adjusted_r_squared=float("nan"),
            concise_score_original=float("nan"),
            concise_score_manuscript=float("nan"),
            x_range_for_line=(float("nan"), float("nan")),
            residuals=empty,
            confidence_intervals={},
            method_metadata={},
            reference=empty,
            prediction=empty,
        )

    r_result = pearson_r(reference, prediction, sample_ids)
    r2_result = r_squared_direct(reference, prediction, sample_ids)
    adj_r2_result = adjusted_r_squared(
        reference, prediction, sample_ids, n_predictors=n_predictors
    )
    concise_orig = concise_score_original(reference, prediction, sample_ids)
    concise_manu = concise_score_manuscript(reference, prediction, sample_ids)
    ts = theil_sen(reference, prediction, sample_ids, alpha=theilsen_alpha)

    warnings = tuple(
        w
        for res in (r_result, r2_result, adj_r2_result, concise_orig, concise_manu, ts)
        for w in res.meta.warnings
    )
    meta = meta_for(
        all_ids=all_ids,
        report=report,
        method="regression_metrics",
        warnings=warnings,
        configuration={"n_predictors": n_predictors, "theilsen_alpha": theilsen_alpha},
    )

    return RegressionResult(
        meta=meta,
        slope=ts.slope,
        intercept=ts.intercept,
        r=r_result.r,
        r_pvalue=r_result.p_value,
        r_squared_direct=r2_result.value,
        adjusted_r_squared=adj_r2_result.value,
        concise_score_original=concise_orig.value,
        concise_score_manuscript=concise_manu.value,
        x_range_for_line=(float(np.min(ref)), float(np.max(ref))),
        residuals=pred - ref,
        confidence_intervals={"theil_sen_slope": (ts.slope_low, ts.slope_high)},
        method_metadata={
            "theil_sen": dict(ts.meta.configuration),
            "pearson_r": dict(r_result.meta.configuration),
            "x_range_for_line_axis": "reference",
        },
        reference=ref,
        prediction=pred,
    )
