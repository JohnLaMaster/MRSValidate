"""Agreement metrics: Lin's concordance correlation coefficient and the
manuscript's experimental robust analogue.

Implements the scalar agreement metrics from the spec/manuscript
(``lin's rho`` / ``lin's rho np``): Lin's concordance correlation
coefficient (:func:`lin_ccc`) and a median/robust-SD-based analogue
(:func:`robust_ccc`).

Notes
-----
Lin's :math:`\\rho_c` is implemented from first principles (not
derived from Pearson r's implementation) because it is an agreement
measure relative to the 45-degree equality line, not merely a
correlation coefficient -- two perfectly correlated but biased samples
must not report :math:`\\rho_c` == 1.

**Variance normalization.** Variances here use the **1/N**
(population) normalization, consistent with Lin's original
formulation, not Bessel's correction used elsewhere in this package
for plain sample variance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .._numerics import robust_sd as _robust_sd
from ..results import AnalysisResult, ResultStatus
from ._shared import meta_for, valid_subset
from .regression import pearson_r

__all__ = ["ConcordanceResult", "lin_ccc", "robust_ccc"]


@dataclass(frozen=True)
class ConcordanceResult(AnalysisResult):
    """Result of a concordance (agreement) computation.

    Returned by both :func:`lin_ccc` and :func:`robust_ccc`; the two
    are distinguished by `experimental`/`method` rather than by
    separate result types, so downstream code can handle either
    uniformly.

    Attributes
    ----------
    meta : AnalysisMeta
        Provenance/inclusion record describing which samples were used
        and why any were excluded.
    rho_c : float
        The concordance correlation coefficient. ``nan`` when
        `meta.status` is not ``ResultStatus.VALID`` (e.g. insufficient
        data or zero total variance/scale).
    r : float
        Pearson's r between reference and prediction, the correlation
        component that `rho_c` rescales by `bias_correction_factor`.
    bias_correction_factor : float
        Lin's :math:`C_b`, the location/scale penalty applied to `r`
        to obtain `rho_c` (``rho_c = r * bias_correction_factor``).
        Equal to 1 only when the two variables share the same
        location and scale.
    location_x : float
        Location (mean for :func:`lin_ccc`, median for
        :func:`robust_ccc`) of the **prediction** array.
    location_y : float
        Location of the **reference** array.
    scale_x : float
        Scale (1/N-normalized SD for :func:`lin_ccc`, robust SD for
        :func:`robust_ccc`) of the **prediction** array.
    scale_y : float
        Scale of the **reference** array.
    experimental : bool
        ``True`` for :func:`robust_ccc` (a manuscript-specific,
        not-yet-statistically-validated statistic), ``False`` for
        :func:`lin_ccc`.
    method : str
        Name of the computing function (``"lin_ccc"`` or
        ``"manuscript_robust_ccc"``), duplicated from `meta.method` for
        convenient access without unpacking `meta`.

    Notes
    -----
    Despite the ``_x``/``_y`` suffixes -- chosen to mirror Lin's
    original notation -- `location_x`/`scale_x` describe `prediction`
    and `location_y`/`scale_y` describe `reference`; they are not named
    after the module's usual ``reference``/``prediction`` argument
    order.
    """

    rho_c: float
    r: float
    bias_correction_factor: float
    location_x: float
    location_y: float
    scale_x: float
    scale_y: float
    experimental: bool
    method: str


def lin_ccc(
    reference: NDArray[np.float64],
    prediction: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
) -> ConcordanceResult:
    """Lin's concordance correlation coefficient.

    Measures agreement with the 45-degree equality line, penalizing
    both loss of correlation and departures in location/scale between
    `reference` and `prediction`.

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
    ConcordanceResult
        `rho_c` is Lin's concordance correlation coefficient,
        `r` is Pearson's r, and `bias_correction_factor` is Lin's
        :math:`C_b`. `experimental` is ``False`` and `method` is
        ``"lin_ccc"``. All numeric fields are ``nan`` when
        `meta.status` is not ``VALID`` (insufficient data, an
        undefined Pearson r, or zero total variance).

    Notes
    -----
    Variances use the **1/N** (population) normalization, consistent
    with Lin's original formulation, not Bessel's correction used
    elsewhere in this package -- see the module docstring and
    ``meta.configuration["variance_normalization"]``.

    .. math::

        \\rho_c = r \\cdot \\frac{2 \\sigma_x \\sigma_y}
        {\\sigma_x^2 + \\sigma_y^2 + (\\mu_x - \\mu_y)^2}

    where :math:`r` is Pearson's r, :math:`\\mu_x, \\sigma_x` are the
    1/N-normalized mean and standard deviation of `prediction`, and
    :math:`\\mu_y, \\sigma_y` are those of `reference`.

    Examples
    --------
    Perfect agreement (identical arrays) gives :math:`\\rho_c = 1`:

    >>> import numpy as np
    >>> from mrsvalidate.analysis.agreement import lin_ccc
    >>> ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> ids = np.array(["s1", "s2", "s3", "s4", "s5"])
    >>> perfect = lin_ccc(ref, ref, ids)
    >>> round(perfect.rho_c, 4)
    1.0

    A perfectly *correlated* but *biased* prediction (shifted by a
    constant offset) is not perfect agreement -- :math:`\\rho_c < 1`
    even though :math:`r = 1`, which is exactly the distinction Lin's
    CCC is designed to capture:

    >>> biased = ref + 2.0
    >>> result = lin_ccc(ref, biased, ids)
    >>> round(result.r, 4)
    1.0
    >>> round(result.rho_c, 4)
    0.5
    """

    all_ids, report, (ref, pred) = valid_subset(sample_ids, reference, prediction)
    meta_base = meta_for(all_ids=all_ids, report=report, method="lin_ccc", min_n=2)
    if meta_base.status is not ResultStatus.VALID:
        return ConcordanceResult(
            meta=meta_base,
            rho_c=float("nan"),
            r=float("nan"),
            bias_correction_factor=float("nan"),
            location_x=float("nan"),
            location_y=float("nan"),
            scale_x=float("nan"),
            scale_y=float("nan"),
            experimental=False,
            method="lin_ccc",
        )

    r_result = pearson_r(reference, prediction, sample_ids)
    if r_result.meta.status is not ResultStatus.VALID:
        return ConcordanceResult(
            meta=r_result.meta,
            rho_c=float("nan"),
            r=float("nan"),
            bias_correction_factor=float("nan"),
            location_x=float("nan"),
            location_y=float("nan"),
            scale_x=float("nan"),
            scale_y=float("nan"),
            experimental=False,
            method="lin_ccc",
        )

    mu_x, mu_y = float(np.mean(pred)), float(np.mean(ref))
    var_x, var_y = float(np.var(pred, ddof=0)), float(np.var(ref, ddof=0))
    sigma_x, sigma_y = np.sqrt(var_x), np.sqrt(var_y)
    denom = (mu_x - mu_y) ** 2 + var_x + var_y
    c_b = (2 * sigma_x * sigma_y / denom) if denom > 0 else float("nan")
    rho_c = r_result.r * c_b if denom > 0 else float("nan")

    status = ResultStatus.UNDEFINED if denom == 0 else ResultStatus.VALID
    meta = meta_for(
        all_ids=all_ids,
        report=report,
        method="lin_ccc",
        status=status,
        configuration={"variance_normalization": "1/N (population, not Bessel-corrected)"},
        warnings=() if denom > 0 else ("Lin CCC undefined: zero total variance",),
    )
    return ConcordanceResult(
        meta=meta,
        rho_c=float(rho_c),
        r=r_result.r,
        bias_correction_factor=float(c_b),
        location_x=mu_x,
        location_y=mu_y,
        scale_x=sigma_x,
        scale_y=sigma_y,
        experimental=False,
        method="lin_ccc",
    )


def robust_ccc(
    reference: NDArray[np.float64],
    prediction: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
) -> ConcordanceResult:
    """Manuscript's robust (median/robust-SD-based) concordance analogue.

    Structurally mirrors :func:`lin_ccc` -- same Pearson-r-times-bias-
    correction form -- but replaces mean/SD with median/robust SD
    throughout, to reduce sensitivity to outliers.

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
    ConcordanceResult
        `rho_c` is the robust concordance analogue, `r` is Pearson's
        r, and `bias_correction_factor` is the robust bias-correction
        term. `location_x`/`location_y` are medians and
        `scale_x`/`scale_y` are raw robust SDs (see
        :func:`mrsvalidate._numerics.robust_sd`) of `prediction`/
        `reference` respectively. `experimental` is ``True`` and
        `method` is ``"manuscript_robust_ccc"``. All numeric fields
        are ``nan`` when `meta.status` is not ``VALID``.

    Notes
    -----
    **Experimental**: this is a manuscript-specific statistic, not a
    drop-in, statistically validated replacement for Lin's CCC -- see
    `experimental`/`method` on the returned result.

    .. math::

        \\rho_{c,robust} = r \\cdot \\frac{2 s_x s_y}
        {s_x^2 + s_y^2 + (\\mathrm{med}(x) - \\mathrm{med}(y))^2}

    where :math:`r` is Pearson's r and :math:`s_x, s_y` are the raw
    (not normal-consistency-rescaled) robust standard deviations of
    `prediction` and `reference`.

    Examples
    --------
    Perfect agreement (identical arrays) gives :math:`\\rho_{c,robust}
    = 1`:

    >>> import numpy as np
    >>> from mrsvalidate.analysis.agreement import robust_ccc
    >>> ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> ids = np.array(["s1", "s2", "s3", "s4", "s5"])
    >>> perfect = robust_ccc(ref, ref, ids)
    >>> round(perfect.rho_c, 4)
    1.0

    As with :func:`lin_ccc`, a perfectly correlated but biased
    prediction is penalized for the offset in location:

    >>> biased = ref + 2.0
    >>> result = robust_ccc(ref, biased, ids)
    >>> round(result.r, 4)
    1.0
    >>> round(result.rho_c, 4)
    0.3333
    >>> result.experimental
    True
    """

    all_ids, report, (ref, pred) = valid_subset(sample_ids, reference, prediction)
    meta_base = meta_for(all_ids=all_ids, report=report, method="manuscript_robust_ccc", min_n=2)
    if meta_base.status is not ResultStatus.VALID:
        return ConcordanceResult(
            meta=meta_base,
            rho_c=float("nan"),
            r=float("nan"),
            bias_correction_factor=float("nan"),
            location_x=float("nan"),
            location_y=float("nan"),
            scale_x=float("nan"),
            scale_y=float("nan"),
            experimental=True,
            method="manuscript_robust_ccc",
        )

    r_result = pearson_r(reference, prediction, sample_ids)
    if r_result.meta.status is not ResultStatus.VALID:
        return ConcordanceResult(
            meta=r_result.meta,
            rho_c=float("nan"),
            r=float("nan"),
            bias_correction_factor=float("nan"),
            location_x=float("nan"),
            location_y=float("nan"),
            scale_x=float("nan"),
            scale_y=float("nan"),
            experimental=True,
            method="manuscript_robust_ccc",
        )

    med_x, med_y = float(np.median(pred)), float(np.median(ref))
    rsd_x, rsd_y = _robust_sd(pred), _robust_sd(ref)
    denom = (med_x - med_y) ** 2 + rsd_x**2 + rsd_y**2
    c_b = (2 * rsd_x * rsd_y / denom) if denom > 0 else float("nan")
    rho_c = r_result.r * c_b if denom > 0 else float("nan")

    status = ResultStatus.UNDEFINED if denom == 0 else ResultStatus.VALID
    meta = meta_for(
        all_ids=all_ids,
        report=report,
        method="manuscript_robust_ccc",
        status=status,
        configuration={
            "experimental": True,
            "location": "median",
            "scale": "raw_robust_sd",
        },
        warnings=() if denom > 0 else ("robust CCC undefined: zero total robust scale",),
    )
    return ConcordanceResult(
        meta=meta,
        rho_c=float(rho_c),
        r=r_result.r,
        bias_correction_factor=float(c_b),
        location_x=med_x,
        location_y=med_y,
        scale_x=rsd_x,
        scale_y=rsd_y,
        experimental=True,
        method="manuscript_robust_ccc",
    )
