"""Manuscript-defined percent-error and squared-error metrics.

Implements MPE, MePE, MAPE/MeAPE (each with a manuscript-literal
definition mode and one or more corrected/conventional definition
modes), bias-corrected MAPE, RMSE and RMeSE.

Sign and variable convention (shared with
:mod:`mrsvalidate.analysis.regression`): ``prediction`` is the model
output (manuscript's ``x``), ``reference`` is ground truth
(manuscript's ``y``). Percent errors are ``100 * (prediction -
reference) / reference`` (spec/eq. MPE), so ``prediction > reference``
is a positive/overestimation error.

Notes
-----
Percentage-error metrics have an explicit, scale-aware zero-reference
policy (spec: "no epsilon substitution"), controlled by
``zero_reference_policy``:

``"omit_and_report"`` (default)
    Rows whose reference value is near zero (near-zero policy, scale =
    SD of the valid reference values) are excluded and counted under
    the ``"near_zero_reference"`` exclusion reason.
``"error"``
    Any near-zero reference makes the whole result
    ``ResultStatus.INVALID_INPUT`` rather than silently omitting rows.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .._numerics import DEFAULT_ABS_TOL, DEFAULT_REL_TOL, near_zero_threshold
from ..results import ResultStatus, ScalarMetricResult, build_meta
from ._shared import meta_for, valid_subset

__all__ = [
    "bias_corrected_mape",
    "mape",
    "meape",
    "mepe",
    "mpe",
    "rmese",
    "rmse",
]

ZeroReferencePolicy = Literal["omit_and_report", "error"]
MapeDefinition = Literal["manuscript_literal", "absolute_reference"]
BiasCorrectedMapeDefinition = Literal["manuscript_literal", "absolute_centered"]
MeapeDefinition = Literal[
    "manuscript_literal", "manuscript_literal_bias_corrected", "conventional"
]


def _prepare(
    reference: NDArray[np.float64],
    prediction: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
    *,
    method: str,
    require_nonzero_reference: bool,
    zero_reference_policy: ZeroReferencePolicy,
    abs_tol: float,
    rel_tol: float,
    extra_configuration: dict | None = None,
):
    """Shared setup for every percent-error / plain-difference metric.

    Excludes non-finite rows, then (when `require_nonzero_reference`
    is true) applies the zero-reference policy described in the
    module docstring, before handing the caller a ready-to-compute
    subset.

    Parameters
    ----------
    reference : ndarray of float
        Ground-truth values (manuscript's ``y``), one per sample.
    prediction : ndarray of float
        Model-output values (manuscript's ``x``), one per sample,
        aligned with `reference`.
    sample_ids : ndarray of str
        Sample identifier aligned with `reference`/`prediction`.
    method : str
        Name recorded as ``meta.method`` on the returned/eventual
        result.
    require_nonzero_reference : bool
        If ``True``, apply the near-zero reference-value policy; if
        ``False``, skip it (used by :func:`rmse`/:func:`rmese`, which
        have no reference-denominator to protect).
    zero_reference_policy : {"omit_and_report", "error"}
        How to handle rows whose reference value is near zero; see the
        module docstring.
    abs_tol : float
        Absolute near-zero tolerance for the reference value.
    rel_tol : float
        Relative near-zero tolerance for the reference value.
    extra_configuration : dict or None, optional
        Additional key/value pairs (e.g. ``{"definition": ...}``)
        merged into ``meta.configuration``.

    Returns
    -------
    early_result : ScalarMetricResult or None
        A fully-formed non-``VALID`` result (``INVALID_INPUT`` or
        ``INSUFFICIENT_DATA``) that the caller should return
        immediately, or ``None`` if the caller should proceed to
        compute the statistic.
    meta : AnalysisMeta or None
        The metadata to attach to the caller's result; only set when
        `early_result` is ``None``.
    arrays : tuple of ndarray or None
        ``(ref, pred)``, the finite, (optionally) nonzero-reference
        subset ready to compute on; only set when `early_result` is
        ``None``.
    unused : None
        Always ``None``; kept so the tuple has a stable four-element
        shape at every call site.
    """

    all_ids, report, (ref, pred) = valid_subset(sample_ids, reference, prediction)

    exclusion_reasons = dict(report.exclusion_reasons)
    warnings: list[str] = []
    included_ids = list(report.included_ids)
    threshold = None

    if require_nonzero_reference and len(ref) > 0:
        scale = float(np.std(ref, ddof=1)) if len(ref) > 1 else float(np.abs(ref[0]))
        threshold = near_zero_threshold(scale, abs_tol=abs_tol, rel_tol=rel_tol)
        near_zero_mask = ref**2 <= threshold
        n_near_zero = int(np.sum(near_zero_mask))

        if n_near_zero > 0:
            if zero_reference_policy == "error":
                meta = meta_for(
                    all_ids=all_ids,
                    report=report,
                    method=method,
                    status=ResultStatus.INVALID_INPUT,
                    configuration={
                        "zero_reference_policy": zero_reference_policy,
                        "near_zero_threshold": threshold,
                        **(extra_configuration or {}),
                    },
                    warnings=(
                        f'{n_near_zero} reference value(s) are effectively zero and zero_reference_policy="error"',
                    ),
                )
                return ScalarMetricResult(meta=meta, value=float("nan")), None, None, None

            # "omit_and_report": drop those rows and record the reason.
            keep = ~near_zero_mask
            included_ids = [included_ids[i] for i in np.flatnonzero(keep)]
            exclusion_reasons["near_zero_reference"] = n_near_zero
            ref = ref[keep]
            pred = pred[keep]
            warnings.append(
                f"{n_near_zero} row(s) omitted: reference value effectively zero"
            )

    config = {"zero_reference_policy": zero_reference_policy, **(extra_configuration or {})}
    if threshold is not None:
        config["near_zero_threshold"] = threshold

    status = ResultStatus.VALID if len(ref) >= 1 else ResultStatus.INSUFFICIENT_DATA
    meta = build_meta(
        status=status,
        all_ids=all_ids,
        included_ids=tuple(included_ids),
        exclusion_reasons=exclusion_reasons,
        warnings=tuple(warnings),
        method=method,
        configuration=config,
    )
    if status is not ResultStatus.VALID:
        return ScalarMetricResult(meta=meta, value=float("nan")), None, None, None

    return None, meta, (ref, pred), None


def mpe(
    reference: NDArray[np.float64],
    prediction: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
    *,
    zero_reference_policy: ZeroReferencePolicy = "omit_and_report",
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
) -> ScalarMetricResult:
    """Mean percent error.

    Signed error, so systematic over- and under-estimation do not
    cancel out only by coincidence: a positive value indicates
    `prediction` is on average above `reference`.

    Parameters
    ----------
    reference : ndarray of float
        Ground-truth values (manuscript's ``y``), one per sample.
    prediction : ndarray of float
        Model-output values (manuscript's ``x``), one per sample,
        aligned with `reference`.
    sample_ids : ndarray of str
        Sample identifier aligned with `reference`/`prediction`.
    zero_reference_policy : {"omit_and_report", "error"}, optional
        How to handle reference values that are effectively zero; see
        the module docstring. Default is ``"omit_and_report"``.
    abs_tol : float, optional
        Absolute near-zero tolerance for the reference value. Default
        is ``mrsvalidate._numerics.DEFAULT_ABS_TOL`` (``0.0``).
    rel_tol : float, optional
        Relative near-zero tolerance for the reference value, scaled
        by the reference SD. Default is
        ``mrsvalidate._numerics.DEFAULT_REL_TOL`` (``1e-12``).

    Returns
    -------
    result : ScalarMetricResult
        ``result.value`` is the mean percent error over the included
        rows. ``result.meta.status`` is ``ResultStatus.INVALID_INPUT``
        when a near-zero reference is present and
        `zero_reference_policy` is ``"error"``, or
        ``ResultStatus.INSUFFICIENT_DATA`` if no rows remain.

    Notes
    -----
    .. math::

        \\text{MPE} = \\frac{100}{N}\\sum_{n=1}^N
            \\frac{\\hat{y}_n - y_n}{y_n}

    where :math:`\\hat{y}_n` is `prediction` and :math:`y_n` is
    `reference`.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.errors import mpe
    >>> ref = np.array([10.0, 10.0, 10.0])
    >>> pred = np.array([11.0, 11.0, 11.0])  # +10% each
    >>> ids = np.array(["a", "b", "c"])
    >>> round(mpe(ref, pred, ids).value, 4)
    10.0
    """

    early, meta, arrays, _ = _prepare(
        reference,
        prediction,
        sample_ids,
        method="mpe",
        require_nonzero_reference=True,
        zero_reference_policy=zero_reference_policy,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )
    if early is not None:
        return early
    ref, pred = arrays
    value = float(100.0 * np.mean((pred - ref) / ref))
    return ScalarMetricResult(meta=meta, value=value)


def mepe(
    reference: NDArray[np.float64],
    prediction: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
    *,
    zero_reference_policy: ZeroReferencePolicy = "omit_and_report",
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
) -> ScalarMetricResult:
    """Median percent error.

    Like :func:`mpe` but robust to outlying percent errors, since the
    median rather than the mean is taken over the signed per-sample
    percent errors.

    Parameters
    ----------
    reference : ndarray of float
        Ground-truth values (manuscript's ``y``), one per sample.
    prediction : ndarray of float
        Model-output values (manuscript's ``x``), one per sample,
        aligned with `reference`.
    sample_ids : ndarray of str
        Sample identifier aligned with `reference`/`prediction`.
    zero_reference_policy : {"omit_and_report", "error"}, optional
        How to handle reference values that are effectively zero; see
        the module docstring. Default is ``"omit_and_report"``.
    abs_tol : float, optional
        Absolute near-zero tolerance for the reference value. Default
        is ``mrsvalidate._numerics.DEFAULT_ABS_TOL`` (``0.0``).
    rel_tol : float, optional
        Relative near-zero tolerance for the reference value, scaled
        by the reference SD. Default is
        ``mrsvalidate._numerics.DEFAULT_REL_TOL`` (``1e-12``).

    Returns
    -------
    result : ScalarMetricResult
        ``result.value`` is the median percent error over the included
        rows. ``result.meta.status`` is ``ResultStatus.INVALID_INPUT``
        when a near-zero reference is present and
        `zero_reference_policy` is ``"error"``, or
        ``ResultStatus.INSUFFICIENT_DATA`` if no rows remain.

    Notes
    -----
    .. math::

        \\text{MePE} = 100 \\cdot \\operatorname{median}_n\\left(
            \\frac{\\hat{y}_n - y_n}{y_n}\\right)

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.errors import mepe
    >>> ref = np.array([10.0, 10.0, 100.0])
    >>> pred = np.array([11.0, 9.0, 90.0])
    >>> ids = np.array(["a", "b", "c"])
    >>> mepe(ref, pred, ids).value
    -10.0
    """

    early, meta, arrays, _ = _prepare(
        reference,
        prediction,
        sample_ids,
        method="mepe",
        require_nonzero_reference=True,
        zero_reference_policy=zero_reference_policy,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )
    if early is not None:
        return early
    ref, pred = arrays
    value = float(100.0 * np.median((pred - ref) / ref))
    return ScalarMetricResult(meta=meta, value=value)


def mape(
    reference: NDArray[np.float64],
    prediction: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
    *,
    definition: MapeDefinition = "manuscript_literal",
    zero_reference_policy: ZeroReferencePolicy = "omit_and_report",
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
) -> ScalarMetricResult:
    """Mean absolute percent error.

    Two definitions are supported and must not be conflated:
    `definition` = ``"manuscript_literal"`` keeps the manuscript's
    signed denominator exactly as displayed, so a negative `reference`
    can flip the sign of individual terms before they are summed
    (each term is no longer guaranteed non-negative even though the
    numerator is an absolute value); `definition` =
    ``"absolute_reference"`` uses ``|reference|`` in the denominator
    instead, which avoids that sign flip and is the recommended choice
    whenever `reference` may be negative.

    Parameters
    ----------
    reference : ndarray of float
        Ground-truth values (manuscript's ``y``), one per sample.
    prediction : ndarray of float
        Model-output values (manuscript's ``x``), one per sample,
        aligned with `reference`.
    sample_ids : ndarray of str
        Sample identifier aligned with `reference`/`prediction`.
    definition : {"manuscript_literal", "absolute_reference"}, optional
        Which denominator convention to use; see Notes. Default is
        ``"manuscript_literal"``.
    zero_reference_policy : {"omit_and_report", "error"}, optional
        How to handle reference values that are effectively zero; see
        the module docstring. Default is ``"omit_and_report"``.
    abs_tol : float, optional
        Absolute near-zero tolerance for the reference value. Default
        is ``mrsvalidate._numerics.DEFAULT_ABS_TOL`` (``0.0``).
    rel_tol : float, optional
        Relative near-zero tolerance for the reference value, scaled
        by the reference SD. Default is
        ``mrsvalidate._numerics.DEFAULT_REL_TOL`` (``1e-12``).

    Returns
    -------
    result : ScalarMetricResult
        ``result.value`` is the mean absolute percent error over the
        included rows, under the selected `definition`.
        ``result.meta.configuration["definition"]`` records which
        convention was used. ``result.meta.status`` is
        ``ResultStatus.INVALID_INPUT`` when a near-zero reference is
        present and `zero_reference_policy` is ``"error"``, or
        ``ResultStatus.INSUFFICIENT_DATA`` if no rows remain.

    Notes
    -----
    ``definition="manuscript_literal"`` (signed denominator, exactly as
    displayed in the manuscript):

    .. math::

        \\text{MAPE}_{\\text{literal}} = \\frac{100}{N}\\sum_{n=1}^N
            \\frac{|\\hat{y}_n - y_n|}{y_n}

    ``definition="absolute_reference"`` (RECOMMENDED EXTENSION — avoids
    sign flips from a negative reference):

    .. math::

        \\text{MAPE}_{\\text{abs\\_ref}} = \\frac{100}{N}\\sum_{n=1}^N
            \\frac{|\\hat{y}_n - y_n|}{|y_n|}

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.errors import mape
    >>> ref = np.array([-10.0, 10.0])
    >>> pred = np.array([-9.0, 11.0])  # errors: +1 and +1
    >>> ids = np.array(["a", "b"])
    >>> mape(ref, pred, ids, definition="manuscript_literal").value
    0.0
    >>> mape(ref, pred, ids, definition="absolute_reference").value
    10.0
    """

    early, meta, arrays, _ = _prepare(
        reference,
        prediction,
        sample_ids,
        method="mape",
        require_nonzero_reference=True,
        zero_reference_policy=zero_reference_policy,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        extra_configuration={"definition": definition},
    )
    if early is not None:
        return early
    ref, pred = arrays
    denom = ref if definition == "manuscript_literal" else np.abs(ref)
    value = float(100.0 * np.mean(np.abs(pred - ref) / denom))
    return ScalarMetricResult(meta=meta, value=value)


def bias_corrected_mape(
    reference: NDArray[np.float64],
    prediction: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
    *,
    definition: BiasCorrectedMapeDefinition = "manuscript_literal",
    zero_reference_policy: ZeroReferencePolicy = "omit_and_report",
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
) -> ScalarMetricResult:
    """Bias-corrected mean absolute percent error.

    Centers each sample's manuscript-literal absolute percent error on
    :func:`mpe` before averaging, so that a systematic bias already
    captured by MPE is not double-counted in the dispersion measure.
    Two definitions are supported: `definition` =
    ``"manuscript_literal"`` reproduces the manuscript's displayed
    equation exactly, which has *no* outer absolute value and is
    algebraically ``MAPE_literal - MPE`` (it can be negative);
    `definition` = ``"absolute_centered"`` takes the absolute value of
    the centered term before averaging, giving a genuine (always
    non-negative) dispersion measure.

    Parameters
    ----------
    reference : ndarray of float
        Ground-truth values (manuscript's ``y``), one per sample.
    prediction : ndarray of float
        Model-output values (manuscript's ``x``), one per sample,
        aligned with `reference`.
    sample_ids : ndarray of str
        Sample identifier aligned with `reference`/`prediction`.
    definition : {"manuscript_literal", "absolute_centered"}, optional
        Which convention to use; see Notes. Default is
        ``"manuscript_literal"``.
    zero_reference_policy : {"omit_and_report", "error"}, optional
        How to handle reference values that are effectively zero; see
        the module docstring. Default is ``"omit_and_report"``.
    abs_tol : float, optional
        Absolute near-zero tolerance for the reference value. Default
        is ``mrsvalidate._numerics.DEFAULT_ABS_TOL`` (``0.0``).
    rel_tol : float, optional
        Relative near-zero tolerance for the reference value, scaled
        by the reference SD. Default is
        ``mrsvalidate._numerics.DEFAULT_REL_TOL`` (``1e-12``).

    Returns
    -------
    result : ScalarMetricResult
        ``result.value`` is the bias-corrected MAPE over the included
        rows, under the selected `definition`.
        ``result.meta.configuration["definition"]`` records which
        convention was used. ``result.meta.status`` is
        ``ResultStatus.INVALID_INPUT`` when a near-zero reference is
        present and `zero_reference_policy` is ``"error"``, or
        ``ResultStatus.INSUFFICIENT_DATA`` if no rows remain.

    Notes
    -----
    ``definition="manuscript_literal"`` (the displayed equation; no
    outer absolute value, so the result can be negative):

    .. math::

        \\text{MAPE}_{\\text{bc,literal}} = \\frac{100}{N}\\sum_{n=1}^N
            \\left(\\frac{|\\hat{y}_n - y_n|}{y_n} - \\frac{\\text{MPE}}{100}\\right)

    ``definition="absolute_centered"`` (RECOMMENDED EXTENSION — a
    genuine absolute centered-percent-error dispersion measure):

    .. math::

        \\text{MAPE}_{\\text{bc,abs}} = \\frac{100}{N}\\sum_{n=1}^N
            \\left|\\,\\frac{|\\hat{y}_n - y_n|}{y_n} - \\frac{\\text{MPE}}{100}\\,\\right|

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.errors import bias_corrected_mape
    >>> ref = np.array([1.0, 1.0, 1.0, 1.0])
    >>> pred = np.array([4.0, 1.0, 1.0, -1.0])
    >>> ids = np.array(["a", "b", "c", "d"])
    >>> bias_corrected_mape(ref, pred, ids, definition="manuscript_literal").value
    100.0
    >>> bias_corrected_mape(ref, pred, ids, definition="absolute_centered").value
    125.0
    """

    early, meta, arrays, _ = _prepare(
        reference,
        prediction,
        sample_ids,
        method="bias_corrected_mape",
        require_nonzero_reference=True,
        zero_reference_policy=zero_reference_policy,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        extra_configuration={"definition": definition},
    )
    if early is not None:
        return early
    ref, pred = arrays
    mpe_value = 100.0 * np.mean((pred - ref) / ref)
    centered = np.abs(pred - ref) / ref - mpe_value / 100.0
    if definition == "manuscript_literal":
        value = float(100.0 * np.mean(centered))
    else:
        value = float(100.0 * np.mean(np.abs(centered)))
    return ScalarMetricResult(meta=meta, value=value)


def meape(
    reference: NDArray[np.float64],
    prediction: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
    *,
    definition: MeapeDefinition = "manuscript_literal",
    zero_reference_policy: ZeroReferencePolicy = "omit_and_report",
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
) -> ScalarMetricResult:
    """Median absolute percent error.

    Three definitions are supported. `definition` =
    ``"manuscript_literal"`` reproduces the manuscript's displayed
    ``1/N`` factor exactly, even though a ``1/N`` prefactor in front of
    a median is not standard practice — it is preserved here for
    faithful reproduction rather than silently "fixed". `definition` =
    ``"manuscript_literal_bias_corrected"`` reproduces the
    manuscript's displayed bias-corrected line, using
    :func:`mepe`-centered terms with the same ``1/N`` prefactor.
    `definition` = ``"conventional"`` drops the ``1/N`` factor and uses
    an absolute-reference denominator, giving the ordinary statistical
    definition of a median absolute percent error.

    Parameters
    ----------
    reference : ndarray of float
        Ground-truth values (manuscript's ``y``), one per sample.
    prediction : ndarray of float
        Model-output values (manuscript's ``x``), one per sample,
        aligned with `reference`.
    sample_ids : ndarray of str
        Sample identifier aligned with `reference`/`prediction`.
    definition : {"manuscript_literal", \
"manuscript_literal_bias_corrected", "conventional"}, optional
        Which convention to use; see Notes. Default is
        ``"manuscript_literal"``.
    zero_reference_policy : {"omit_and_report", "error"}, optional
        How to handle reference values that are effectively zero; see
        the module docstring. Default is ``"omit_and_report"``.
    abs_tol : float, optional
        Absolute near-zero tolerance for the reference value. Default
        is ``mrsvalidate._numerics.DEFAULT_ABS_TOL`` (``0.0``).
    rel_tol : float, optional
        Relative near-zero tolerance for the reference value, scaled
        by the reference SD. Default is
        ``mrsvalidate._numerics.DEFAULT_REL_TOL`` (``1e-12``).

    Returns
    -------
    result : ScalarMetricResult
        ``result.value`` is the median absolute percent error over the
        included rows, under the selected `definition`.
        ``result.meta.configuration["definition"]`` records which
        convention was used. ``result.meta.status`` is
        ``ResultStatus.INVALID_INPUT`` when a near-zero reference is
        present and `zero_reference_policy` is ``"error"``, or
        ``ResultStatus.INSUFFICIENT_DATA`` if no rows remain.

    Notes
    -----
    ``definition="manuscript_literal"`` (reproduces the manuscript's
    ``1/N`` factor in front of the median, inconsistent with a median
    but preserved for reproduction):

    .. math::

        \\text{MeAPE}_{\\text{literal}} = \\frac{100}{N} \\cdot
            \\operatorname{median}_n\\left(\\frac{|\\hat{y}_n - y_n|}{y_n}\\right)

    ``definition="manuscript_literal_bias_corrected"`` (the
    manuscript's displayed bias-corrected line):

    .. math::

        \\text{MeAPE}_{\\text{bc}} = \\frac{100}{N} \\cdot
            \\operatorname{median}_n\\left(\\frac{|\\hat{y}_n - y_n|}{y_n}
            - \\frac{\\text{MePE}}{100}\\right)

    ``definition="conventional"`` (RECOMMENDED EXTENSION — no ``1/N``
    factor, absolute-reference denominator):

    .. math::

        \\text{MeAPE}_{\\text{conventional}} = 100 \\cdot
            \\operatorname{median}_n\\left(\\frac{|\\hat{y}_n - y_n|}{|y_n|}\\right)

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.errors import meape
    >>> ref = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    >>> pred = np.array([2.0, 2.0, 5.0, 4.0, 90.0])
    >>> ids = np.array(["a", "b", "c", "d", "e"])
    >>> meape(ref, pred, ids, definition="manuscript_literal").value
    2.0
    >>> meape(ref, pred, ids, definition="conventional").value
    10.0
    """

    early, meta, arrays, _ = _prepare(
        reference,
        prediction,
        sample_ids,
        method="meape",
        require_nonzero_reference=True,
        zero_reference_policy=zero_reference_policy,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        extra_configuration={"definition": definition},
    )
    if early is not None:
        return early
    ref, pred = arrays
    n = len(ref)

    if definition == "manuscript_literal":
        value = float((100.0 / n) * np.median(np.abs(pred - ref) / ref))
    elif definition == "manuscript_literal_bias_corrected":
        mepe_value = 100.0 * np.median((pred - ref) / ref)
        value = float(
            (100.0 / n) * np.median(np.abs(pred - ref) / ref - mepe_value / 100.0)
        )
    else:  # conventional
        value = float(100.0 * np.median(np.abs(pred - ref) / np.abs(ref)))
    return ScalarMetricResult(meta=meta, value=value)


def rmse(
    reference: NDArray[np.float64],
    prediction: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
) -> ScalarMetricResult:
    """Root mean squared error between `prediction` and `reference`.

    Unlike the percent-error metrics in this module, RMSE has no
    reference-denominator and therefore no zero-reference policy: it
    is always defined once at least one finite-valued row remains.

    Parameters
    ----------
    reference : ndarray of float
        Ground-truth values (manuscript's ``y``), one per sample.
    prediction : ndarray of float
        Model-output values (manuscript's ``x``), one per sample,
        aligned with `reference`.
    sample_ids : ndarray of str
        Sample identifier aligned with `reference`/`prediction`.

    Returns
    -------
    result : ScalarMetricResult
        ``result.value`` is the root mean squared error over the
        included rows. ``result.meta.status`` is
        ``ResultStatus.INSUFFICIENT_DATA`` (``value = nan``) if no
        rows remain.

    Notes
    -----
    .. math::

        \\text{RMSE} = \\sqrt{\\frac{1}{N}\\sum_{n=1}^N (\\hat{y}_n - y_n)^2}

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.errors import rmse
    >>> ref = np.array([1.0, 2.0, 3.0])
    >>> pred = np.array([2.0, 4.0, 6.0])
    >>> ids = np.array(["a", "b", "c"])
    >>> rmse(ref, pred, ids).value
    2.160246899469287
    """

    all_ids, report, (ref, pred) = valid_subset(sample_ids, reference, prediction)
    meta = meta_for(all_ids=all_ids, report=report, method="rmse")
    value = (
        float(np.sqrt(np.mean((pred - ref) ** 2)))
        if meta.status is ResultStatus.VALID
        else float("nan")
    )
    return ScalarMetricResult(meta=meta, value=value)


def rmese(
    reference: NDArray[np.float64],
    prediction: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
) -> ScalarMetricResult:
    """Root median squared error between `prediction` and `reference`.

    Like :func:`rmse` but robust to a small number of large errors,
    since the median rather than the mean is taken over the squared
    per-sample errors. Has no reference-denominator and therefore no
    zero-reference policy.

    Parameters
    ----------
    reference : ndarray of float
        Ground-truth values (manuscript's ``y``), one per sample.
    prediction : ndarray of float
        Model-output values (manuscript's ``x``), one per sample,
        aligned with `reference`.
    sample_ids : ndarray of str
        Sample identifier aligned with `reference`/`prediction`.

    Returns
    -------
    result : ScalarMetricResult
        ``result.value`` is the root median squared error over the
        included rows. ``result.meta.status`` is
        ``ResultStatus.INSUFFICIENT_DATA`` (``value = nan``) if no
        rows remain.

    Notes
    -----
    .. math::

        \\text{RMeSE} = \\sqrt{\\operatorname{median}_n\\left(
            (\\hat{y}_n - y_n)^2\\right)}

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.errors import rmese
    >>> ref = np.array([1.0, 2.0, 3.0])
    >>> pred = np.array([2.0, 4.0, 6.0])
    >>> ids = np.array(["a", "b", "c"])
    >>> rmese(ref, pred, ids).value
    2.0
    """

    all_ids, report, (ref, pred) = valid_subset(sample_ids, reference, prediction)
    meta = meta_for(all_ids=all_ids, report=report, method="rmese")
    value = (
        float(np.sqrt(np.median((pred - ref) ** 2)))
        if meta.status is ResultStatus.VALID
        else float("nan")
    )
    return ScalarMetricResult(meta=meta, value=value)
