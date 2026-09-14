"""Effective sample size and clustering diagnostics.

This module implements the manuscript's one-way and cross-classified
random-effects intraclass correlation (ICC), design effect, and
effective-sample-size computations (spec eqs. 3-10).

Notes
-----
Variance components are estimated via REML using
``statsmodels.regression.mixed_linear_model.MixedLM`` (spec: Statsmodels
provides appropriate infrastructure for mixed-effects fitting, but the
manuscript's specific models are not assumed to be solved automatically
— the one-way and cross-classified designs are wired up explicitly
here).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from statsmodels.regression.mixed_linear_model import MixedLM

from ..results import AnalysisResult, ResultStatus, build_meta
from .sample_size import MINIMUM_INDEPENDENT_CLUSTERS, check_minimum_clusters

__all__ = [
    "CrossClassifiedEffectiveSampleSizeResult",
    "EffectiveSampleSizeResult",
    "cross_classified_icc",
    "one_way_icc",
]


@dataclass(frozen=True)
class EffectiveSampleSizeResult(AnalysisResult):
    """Result of a one-way (single clustering dimension) ICC computation.

    Returned by :func:`one_way_icc`. The clustering dimension is a
    single grouping variable, e.g. subject, with repeated observations
    ("spectra") nested within it. Per the manuscript, this should be
    computed per metabolite, not pooled across metabolites.

    Attributes
    ----------
    meta : AnalysisMeta
        Provenance and inclusion/exclusion record (status, sample
        counts, resolved configuration, warnings).
    rho : float
        Intraclass correlation coefficient (ICC), the fraction of
        total variance attributable to between-subject differences.
    between_variance : float
        Estimated between-subject variance component
        (:math:`\\sigma^2_{between}`).
    within_variance : float
        Estimated within-subject (residual) variance component
        (:math:`\\sigma^2_{within}`).
    design_effect : float
        Multiplicative inflation of sampling variance caused by
        within-subject correlation, relative to an equal number of
        fully independent observations.
    n_eff : float
        Effective sample size: the number of independent observations
        that would carry the same statistical information as
        `n_total` correlated ones.
    n_total : int
        Total number of observations (spectra) used in the fit.
    n_subjects : int
        Number of distinct subjects (clusters) used in the fit.
    mean_cluster_size : float
        Mean number of observations per subject, ``n_total / n_subjects``.
    """

    rho: float
    between_variance: float
    within_variance: float
    design_effect: float
    n_eff: float
    n_total: int
    n_subjects: int
    mean_cluster_size: float


@dataclass(frozen=True)
class CrossClassifiedEffectiveSampleSizeResult(AnalysisResult):
    """Result of a cross-classified ICC computation.

    Returned by :func:`cross_classified_icc` (manuscript eqs.
    ``deff_cc`` / ``neff_cc`` / ``icc_cc``). Two clustering dimensions
    (metabolite and subject/spectrum) are cross-classified rather than
    nested: every metabolite is observed for every subject/spectrum,
    and neither grouping is a subset of the other.

    Attributes
    ----------
    meta : AnalysisMeta
        Provenance and inclusion/exclusion record (status, sample
        counts, resolved configuration, warnings).
    rho_metabolite : float
        Intraclass correlation attributable to the metabolite
        clustering dimension.
    rho_subject : float
        Intraclass correlation attributable to the subject/spectrum
        clustering dimension.
    metabolite_variance : float
        Estimated variance component for the metabolite dimension.
    subject_variance : float
        Estimated variance component for the subject/spectrum
        dimension.
    residual_variance : float
        Estimated residual (unexplained) variance component.
    design_effect : float
        Cross-classified design effect combining both clustering
        dimensions' contributions to variance inflation.
    n_eff : float
        Effective sample size accounting for both clustering
        dimensions.
    n_total : int
        Total number of observations used in the fit.
    n_metabolites : int
        Number of distinct metabolites used in the fit.
    n_subjects : int
        Number of distinct subjects used in the fit.
    mean_metabolites_per_spectrum : float
        Mean number of metabolites observed per spectrum.
    mean_spectra_per_subject : float
        Mean number of spectra observed per subject.
    """

    rho_metabolite: float
    rho_subject: float
    metabolite_variance: float
    subject_variance: float
    residual_variance: float
    design_effect: float
    n_eff: float
    n_total: int
    n_metabolites: int
    n_subjects: int
    mean_metabolites_per_spectrum: float
    mean_spectra_per_subject: float


def _insufficient(
    method: str, all_ids: tuple[str, ...], included_ids: tuple[str, ...], exclusion_reasons: dict
) -> ResultStatus:
    return build_meta(
        status=ResultStatus.INSUFFICIENT_DATA,
        all_ids=all_ids,
        included_ids=included_ids,
        exclusion_reasons=exclusion_reasons,
        method=method,
    )


def one_way_icc(
    values: NDArray[np.float64],
    subject_ids: NDArray[np.str_],
    sample_ids: NDArray[np.str_],
    *,
    min_subjects: int = 2,
) -> EffectiveSampleSizeResult:
    """Compute the one-way ICC, design effect, and effective sample size.

    Fits the random-intercept model :math:`d_{s,n} = \\mu + w_s +
    \\nu_{s,n}` by REML for a single metabolite, where :math:`s` indexes
    subjects and :math:`n` indexes repeated observations (spectra)
    within a subject, :math:`w_s \\sim N(0, \\sigma^2_{between})` is the
    subject random effect, and :math:`\\nu_{s,n} \\sim N(0,
    \\sigma^2_{within})` is residual noise. This should be called once
    per metabolite, not on pooled data across metabolites.

    Parameters
    ----------
    values : ndarray of float
        Observed values (one metabolite), one entry per row.
    subject_ids : ndarray of str
        Subject label for each row; rows sharing a label are treated as
        repeated observations on the same subject.
    sample_ids : ndarray of str
        Identifier for each row, used for provenance
        (``meta.included_ids``/``meta.excluded_ids``); non-finite
        `values` are excluded before fitting.
    min_subjects : int, optional
        Minimum number of distinct subjects required to attempt the
        fit. Default 2.

    Returns
    -------
    EffectiveSampleSizeResult
        The fitted variance components, ICC, design effect, and
        effective sample size. ``meta.status`` is
        ``ResultStatus.INSUFFICIENT_DATA`` if fewer than 2 observations
        or fewer than `min_subjects` subjects remain after excluding
        non-finite values, or ``ResultStatus.FAILED`` if the REML fit
        raises (e.g. failure to converge); in both cases the numeric
        fields are ``NaN``.

    Notes
    -----
    The intraclass correlation is the fraction of total variance
    attributable to between-subject differences:

    .. math::

        \\rho = \\frac{\\sigma^2_{between}}
        {\\sigma^2_{between} + \\sigma^2_{within}}

    The design effect inflates the sampling variance of statistics
    computed on the pooled, correlated observations relative to the
    same count of fully independent ones, as a function of the mean
    number of observations per subject :math:`\\bar{n}` (``n_total /
    n_subjects``):

    .. math::

        \\text{deff} = 1 + \\rho \\, (\\bar{n} - 1)

    and the effective sample size is the corresponding deflation of the
    raw observation count:

    .. math::

        n_{eff} = \\frac{n_{total}}{\\text{deff}}

    Variance components are estimated via REML using
    ``statsmodels.regression.mixed_linear_model.MixedLM``. When fewer
    than :data:`~mrsvalidate.analysis.sample_size.MINIMUM_INDEPENDENT_CLUSTERS`
    subjects are available, a warning is attached to ``meta.warnings``
    noting that cluster-robust inference may be unreliable, but the
    result is still returned as ``VALID``.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.clustering import one_way_icc
    >>> rng = np.random.default_rng(0)
    >>> subject_effect = np.repeat([0.0, 2.0, -1.0], 4)
    >>> noise = rng.normal(0, 0.5, size=12)
    >>> values = 10.0 + subject_effect + noise
    >>> subject_ids = np.repeat(["subj1", "subj2", "subj3"], 4)
    >>> sample_ids = np.array([f"r{i}" for i in range(12)])
    >>> result = one_way_icc(values, subject_ids, sample_ids)
    >>> result.meta.status.value
    'valid'
    >>> result.n_subjects
    3
    >>> round(result.rho, 3)
    0.974
    >>> round(result.n_eff, 3)
    3.06
    """

    values_arr = np.asarray(values, dtype=np.float64)
    subject_arr = np.asarray(subject_ids, dtype=str)
    sample_arr = np.asarray(sample_ids, dtype=str)
    all_ids = tuple(sample_arr.tolist())

    finite = np.isfinite(values_arr)
    included_ids = tuple(sample_arr[finite].tolist())
    exclusion_reasons = {"nan_or_inf": int(np.sum(~finite))} if not np.all(finite) else {}

    v = values_arr[finite]
    subj = subject_arr[finite]
    n_total = v.shape[0]
    unique_subjects = np.unique(subj)
    s = int(unique_subjects.shape[0])

    nan_result = {
        "rho": float("nan"),
        "between_variance": float("nan"),
        "within_variance": float("nan"),
        "design_effect": float("nan"),
        "n_eff": float("nan"),
        "n_total": n_total,
        "n_subjects": s,
        "mean_cluster_size": (n_total / s) if s else float("nan"),
    }

    if n_total < 2 or s < min_subjects:
        meta = _insufficient("one_way_icc", all_ids, included_ids, exclusion_reasons)
        return EffectiveSampleSizeResult(meta=meta, **nan_result)

    df = pd.DataFrame({"value": v, "subject": subj})
    try:
        model = MixedLM(df["value"], np.ones((n_total, 1)), groups=df["subject"])
        fit = model.fit(reml=True)
        between = float(np.asarray(fit.cov_re)[0, 0])
        within = float(fit.scale)
    except Exception as exc:  # noqa: BLE001 -- statsmodels can raise many different exception types on convergence failure; caught broadly and deliberately so it is reported as ResultStatus.FAILED rather than propagating
        meta = build_meta(
            status=ResultStatus.FAILED,
            all_ids=all_ids,
            included_ids=included_ids,
            exclusion_reasons=exclusion_reasons,
            method="one_way_icc",
            warnings=(f"MixedLM failed to converge: {exc}",),
        )
        return EffectiveSampleSizeResult(meta=meta, **nan_result)

    denom = between + within
    rho = between / denom if denom > 0 else float("nan")
    mean_n = n_total / s
    deff = 1.0 + rho * (mean_n - 1.0) if np.isfinite(rho) else float("nan")
    n_eff = n_total / deff if deff and np.isfinite(deff) else float("nan")

    warnings = ()
    if not check_minimum_clusters(s):
        warnings = (
            f"only {s} independent subjects; reliable cluster-robust inference "
            f"recommends >={MINIMUM_INDEPENDENT_CLUSTERS}"
        ),
    meta = build_meta(
        status=ResultStatus.VALID,
        all_ids=all_ids,
        included_ids=included_ids,
        exclusion_reasons=exclusion_reasons,
        method="one_way_icc",
        configuration={"fit_method": "statsmodels.MixedLM(REML)", "n_subjects": s},
        warnings=warnings,
    )
    return EffectiveSampleSizeResult(
        meta=meta,
        rho=rho,
        between_variance=between,
        within_variance=within,
        design_effect=deff,
        n_eff=n_eff,
        n_total=n_total,
        n_subjects=s,
        mean_cluster_size=mean_n,
    )


def cross_classified_icc(
    values: NDArray[np.float64],
    metabolite_ids: NDArray[np.str_],
    subject_ids: NDArray[np.str_],
    sample_ids: NDArray[np.str_],
    *,
    min_levels: int = 2,
) -> CrossClassifiedEffectiveSampleSizeResult:
    """Compute the cross-classified ICC, design effect, and effective size.

    Fits a cross-classified random-effects model with two clustering
    dimensions that are not nested — metabolite and subject/spectrum —
    via statsmodels' crossed-random-effects (variance components)
    machinery (``MixedLM.from_formula`` with a ``vc_formula`` for each
    dimension).

    Parameters
    ----------
    values : ndarray of float
        Observed values, one entry per row.
    metabolite_ids : ndarray of str
        Metabolite label for each row.
    subject_ids : ndarray of str
        Subject label for each row.
    sample_ids : ndarray of str
        Spectrum identifier for each row (rows sharing a `sample_ids`
        value are different metabolites measured on the same
        spectrum); also used for provenance
        (``meta.included_ids``/``meta.excluded_ids``).
    min_levels : int, optional
        Minimum number of distinct metabolites and minimum number of
        distinct subjects required to attempt the fit. Default 2.

    Returns
    -------
    CrossClassifiedEffectiveSampleSizeResult
        The fitted variance components, both ICCs, the cross-classified
        design effect, and the effective sample size. ``meta.status``
        is ``ResultStatus.INSUFFICIENT_DATA`` if fewer than 2
        observations remain, or fewer than `min_levels` metabolites or
        subjects remain, after excluding non-finite values; or
        ``ResultStatus.FAILED`` if the crossed REML fit raises. In both
        cases the numeric fields are ``NaN``.

    Notes
    -----
    Each clustering dimension contributes its own intraclass
    correlation, both expressed over the same total variance
    (metabolite variance + subject variance + residual variance):

    .. math::

        \\rho_{metabolite} = \\frac{\\sigma^2_{metabolite}}
        {\\sigma^2_{metabolite} + \\sigma^2_{subject} + \\sigma^2_{residual}}
        \\qquad
        \\rho_{subject} = \\frac{\\sigma^2_{subject}}
        {\\sigma^2_{metabolite} + \\sigma^2_{subject} + \\sigma^2_{residual}}

    The cross-classified design effect combines both dimensions'
    contributions, using the mean metabolites per spectrum
    :math:`\\bar{k}` and mean spectra per subject :math:`\\bar{n}`
    (manuscript's unbalanced-design approximation, computed as
    ``total / number of clusters`` for each dimension):

    .. math::

        \\text{deff}_{cc} = 1 + \\rho_{metabolite} \\,(\\bar{k} - 1)
        + \\rho_{subject} \\,(\\bar{n} - 1)

    and the effective sample size scales the total number of
    metabolite x subject combinations by that design effect:

    .. math::

        n_{eff} = \\frac{k \\, s \\, \\bar{n}}{\\text{deff}_{cc}}

    where :math:`k` is the number of metabolites and :math:`s` the
    number of subjects.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.clustering import cross_classified_icc
    >>> rng = np.random.default_rng(1)
    >>> sample_ids = np.array(
    ...     ["sp1", "sp1", "sp1", "sp2", "sp2", "sp2",
    ...      "sp3", "sp3", "sp3", "sp4", "sp4", "sp4"]
    ... )
    >>> subject_ids = np.array(["sub1"] * 6 + ["sub2"] * 6)
    >>> metabolite_ids = np.array(["mA", "mB", "mC"] * 4)
    >>> metabolite_effect = np.tile([0.0, 1.5, -1.0], 4)
    >>> subject_effect = np.repeat([0.0, 2.0], 6)
    >>> noise = rng.normal(0, 0.3, size=12)
    >>> values = 5.0 + metabolite_effect + subject_effect + noise
    >>> result = cross_classified_icc(
    ...     values, metabolite_ids, subject_ids, sample_ids
    ... )
    >>> result.meta.status.value
    'valid'
    >>> result.n_metabolites, result.n_subjects
    (3, 2)
    >>> round(result.rho_metabolite, 3), round(result.rho_subject, 3)
    (0.46, 0.533)
    >>> round(result.n_eff, 3)
    4.893
    """

    values_arr = np.asarray(values, dtype=np.float64)
    metabolite_arr = np.asarray(metabolite_ids, dtype=str)
    subject_arr = np.asarray(subject_ids, dtype=str)
    sample_arr = np.asarray(sample_ids, dtype=str)
    all_ids = tuple(sample_arr.tolist())

    finite = np.isfinite(values_arr)
    included_ids = tuple(sample_arr[finite].tolist())
    exclusion_reasons = {"nan_or_inf": int(np.sum(~finite))} if not np.all(finite) else {}

    v = values_arr[finite]
    metab = metabolite_arr[finite]
    subj = subject_arr[finite]
    n_total = v.shape[0]
    unique_metab = np.unique(metab)
    unique_subj = np.unique(subj)
    k = int(unique_metab.shape[0])
    s = int(unique_subj.shape[0])

    nan_result = {
        "rho_metabolite": float("nan"),
        "rho_subject": float("nan"),
        "metabolite_variance": float("nan"),
        "subject_variance": float("nan"),
        "residual_variance": float("nan"),
        "design_effect": float("nan"),
        "n_eff": float("nan"),
        "n_total": n_total,
        "n_metabolites": k,
        "n_subjects": s,
        "mean_metabolites_per_spectrum": float("nan"),
        "mean_spectra_per_subject": float("nan"),
    }

    if n_total < 2 or k < min_levels or s < min_levels:
        meta = _insufficient(
            "cross_classified_icc", all_ids, included_ids, exclusion_reasons
        )
        return CrossClassifiedEffectiveSampleSizeResult(meta=meta, **nan_result)

    df = pd.DataFrame({"value": v, "metabolite": metab, "subject": subj})
    vc = {"metabolite": "0 + C(metabolite)", "subject": "0 + C(subject)"}
    try:
        model = MixedLM.from_formula(
            "value ~ 1", groups=np.ones(n_total), vc_formula=vc, data=df
        )
        fit = model.fit(reml=True)
        var_metabolite = float(fit.vcomp[0])
        var_subject = float(fit.vcomp[1])
        residual = float(fit.scale)
    except Exception as exc:  # noqa: BLE001 -- statsmodels can raise many different exception types on convergence failure; caught broadly and deliberately so it is reported as ResultStatus.FAILED rather than propagating
        meta = build_meta(
            status=ResultStatus.FAILED,
            all_ids=all_ids,
            included_ids=included_ids,
            exclusion_reasons=exclusion_reasons,
            method="cross_classified_icc",
            warnings=(f"crossed MixedLM failed to converge: {exc}",),
        )
        return CrossClassifiedEffectiveSampleSizeResult(meta=meta, **nan_result)

    denom = var_metabolite + var_subject + residual
    rho_i = var_metabolite / denom if denom > 0 else float("nan")
    rho_n = var_subject / denom if denom > 0 else float("nan")

    # Number of distinct spectra ("N") equals the number of unique
    # sample_ids among included rows; mean metabolites per spectrum and
    # mean spectra per subject follow the manuscript's unbalanced-design
    # approximation (total / number of clusters).
    n_spectra = int(np.unique(sample_arr[finite]).shape[0])
    mean_k = n_total / n_spectra if n_spectra else float("nan")
    mean_n = n_spectra / s if s else float("nan")

    deff_cc = 1.0 + rho_i * (mean_k - 1.0) + rho_n * (mean_n - 1.0)
    n_eff = (k * s * mean_n) / deff_cc if deff_cc else float("nan")

    meta = build_meta(
        status=ResultStatus.VALID,
        all_ids=all_ids,
        included_ids=included_ids,
        exclusion_reasons=exclusion_reasons,
        method="cross_classified_icc",
        configuration={
            "fit_method": "statsmodels.MixedLM(REML, crossed vc_formula)",
            "n_metabolites": k,
            "n_subjects": s,
        },
    )
    return CrossClassifiedEffectiveSampleSizeResult(
        meta=meta,
        rho_metabolite=rho_i,
        rho_subject=rho_n,
        metabolite_variance=var_metabolite,
        subject_variance=var_subject,
        residual_variance=residual,
        design_effect=deff_cc,
        n_eff=n_eff,
        n_total=n_total,
        n_metabolites=k,
        n_subjects=s,
        mean_metabolites_per_spectrum=mean_k,
        mean_spectra_per_subject=mean_n,
    )
