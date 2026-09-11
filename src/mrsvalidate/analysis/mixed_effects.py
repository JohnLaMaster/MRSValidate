"""Model-level (cross-metabolite) mixed-effects extension of the Silveira
et al. three-part equivalence protocol.

Implements spec eqs. 67-72 (manuscript ``me_test1``/``me_test2``/
``me_test3``, cluster-robust sandwich eqs. ``sandwich estimator``/
``sandwich mean``): the same three equivalence tests as
:mod:`mrsvalidate.analysis.method_comparison`, but fit jointly across
all metabolites of a panel, with metabolite treated as a random-effects
grouping factor.

Notes
-----
Tests 1-2 map onto standard mixed-effects likelihoods (random
intercept / random intercept+slope by metabolite) and are fit with
``statsmodels.MixedLM``, per spec guidance that Statsmodels is
appropriate reusable infrastructure *where its likelihood matches the
intended model*. Fixed-effect standard errors are cluster-robust,
clustered on spectrum (in-silico) or subject (in-vivo), via the
manuscript's own sandwich formula (implemented directly here — not a
Statsmodels built-in for a mixed model).

Test 3 combines errors-in-variables (Deming-type) regression with
random effects across metabolites, which is **not** a stock
``MixedLM`` likelihood (spec: implement as its own model/cluster-bootstrap
procedure). This implementation uses a cluster bootstrap: independent
clusters (spectra/subjects) are resampled with replacement, carrying
every metabolite for that cluster, and a pooled Deming fit is
recomputed per replicate to characterize the sampling distribution of
the joint (alpha, beta) null.

**Simplifications, stated explicitly (this is the spec's own
highest-complexity item, flagged as needing Statistical Reviewer
sign-off):** the cluster-robust "bread" matrix for Tests 1-2 uses
``(X'X)^-1`` on the fixed-effects design (not the mixed model's full
GLS information matrix), and Test 3's bootstrap recomputes a *pooled*
Deming fit per replicate rather than an explicit per-replicate
metabolite-random-effects decomposition. Both are documented, reduced
approximations to the full model described in the manuscript, not a
claim of an exact, published solution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats as _stats
from statsmodels.regression.mixed_linear_model import MixedLM

from ..results import AnalysisResult, ResultStatus, build_meta
from .method_comparison import deming

__all__ = [
    "mixed_equivalence_test_1",
    "mixed_equivalence_test_2",
    "mixed_equivalence_test_3",
    "MixedTest1Result",
    "MixedTest2Result",
    "MixedTest3Result",
]


def _cluster_robust_se(
    design: NDArray[np.float64], residuals: NDArray[np.float64], cluster_ids: NDArray[np.str_]
) -> NDArray[np.float64]:
    """Compute cluster-robust (CR0) sandwich standard errors.

    Parameters
    ----------
    design : ndarray of float, shape (n_obs, n_params)
        Fixed-effects design matrix.
    residuals : ndarray of float, shape (n_obs,)
        Marginal residuals ``y - design @ fixed_effect_estimate``.
    cluster_ids : ndarray of str, shape (n_obs,)
        Cluster membership (spectrum or subject) for each observation.

    Returns
    -------
    ndarray of float, shape (n_params,)
        Cluster-robust standard error for each fixed-effect coefficient,
        the square root of the diagonal of the sandwich covariance
        ``(X'X)^-1 (sum_c X_c' e_c e_c' X_c) (X'X)^-1``.

    Notes
    -----
    Uses the Moore-Penrose pseudoinverse of ``X'X`` (the "bread") rather
    than a design-specific closed form, and is the reduced ``(X'X)^-1``
    bread — not the mixed model's full GLS information matrix — flagged
    in the module Notes as a documented simplification.
    """

    xtx = design.T @ design
    xtx_inv = np.linalg.pinv(xtx)
    p = design.shape[1]
    b_hat = np.zeros((p, p))
    for cluster in np.unique(cluster_ids):
        mask = cluster_ids == cluster
        score = design[mask].T @ residuals[mask]
        b_hat += np.outer(score, score)
    cov = xtx_inv @ b_hat @ xtx_inv
    return np.sqrt(np.diag(cov))


@dataclass(frozen=True)
class MixedTest1Result(AnalysisResult):
    """Result of :func:`mixed_equivalence_test_1` (model-level accuracy).

    Attributes
    ----------
    meta : AnalysisMeta
        Provenance and inclusion/exclusion record for this computation.
    alpha : float
        Fixed-effect intercept (population-average accuracy bias across
        metabolites).
    beta : float
        Fixed-effect slope on the metabolite-mean-centered reference.
    tau_alpha_sq : float
        Estimated variance of the random intercept across metabolites,
        ``Var(u_i)``.
    alpha_se_cluster_robust : float
        Cluster-robust standard error of `alpha` (see
        :func:`_cluster_robust_se`).
    alpha_z : float
        z-statistic for H0: ``alpha = 0``, ``alpha / alpha_se_cluster_robust``.
    alpha_p_value : float
        Two-sided p-value for `alpha_z` against a standard normal
        distribution.
    n_metabolites : int
        Number of distinct metabolites included in the fit.
    n_clusters : int
        Number of distinct clusters (spectra/subjects) included in the
        fit.
    """

    alpha: float
    beta: float
    tau_alpha_sq: float
    alpha_se_cluster_robust: float
    alpha_z: float
    alpha_p_value: float
    n_metabolites: int
    n_clusters: int


def mixed_equivalence_test_1(
    differences: NDArray[np.float64],
    reference_values: NDArray[np.float64],
    metabolite_ids: NDArray[np.str_],
    cluster_ids: NDArray[np.str_],
    sample_ids: NDArray[np.str_],
) -> MixedTest1Result:
    r"""Model-level equivalence Test 1: accuracy, pooled across metabolites.

    Fits the mixed model

    .. math::

        x_{i,n} - y_{i,n} = (\alpha + u_i)
            + \beta \, (y_{i,n} - \bar{y}_i) + \nu_{i,n}

    where :math:`i` indexes metabolite, :math:`n` indexes sample, and
    :math:`u_i` is a random intercept by metabolite, then tests H0:
    ``alpha = 0`` using a cluster-robust standard error.

    Parameters
    ----------
    differences : ndarray of float, shape (n_obs,)
        Per-observation differences ``x_{i,n} - y_{i,n}``
        (candidate - reference), one row per (metabolite, sample) pair.
    reference_values : ndarray of float, shape (n_obs,)
        Per-observation reference values ``y_{i,n}``, aligned with
        `differences`.
    metabolite_ids : ndarray of str, shape (n_obs,)
        Metabolite label for each observation; the random-effects
        grouping factor.
    cluster_ids : ndarray of str, shape (n_obs,)
        Cluster label for each observation (spectrum id for *in silico*
        data, subject id for *in vivo* data); the unit resampled/
        clustered for the sandwich standard error.
    sample_ids : ndarray of str, shape (n_obs,)
        Per-observation identifiers, used for inclusion/exclusion
        bookkeeping.

    Returns
    -------
    MixedTest1Result
        Fixed-effect estimates, the random-intercept variance, and the
        accuracy test statistics. See :class:`MixedTest1Result` for the
        full field list.

    Notes
    -----
    Fit via ``statsmodels.MixedLM`` with REML and a random intercept
    grouped by `metabolite_ids`; the reference is centered per
    metabolite (``y_{i,n} - bar{y}_i``) before fitting. The reported
    standard error is the cluster-robust sandwich estimate on the
    fixed-effects design (see module Notes on the "bread" matrix
    simplification), not ``MixedLM``'s own model-based standard error.

    Fewer than 4 valid observations, or fewer than 2 distinct
    metabolites, yields a NaN-filled result with
    ``meta.status = ResultStatus.INSUFFICIENT_DATA``. A non-converging
    or erroring fit yields ``meta.status = ResultStatus.FAILED`` with
    the underlying exception message recorded in ``meta.warnings``.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.mixed_effects import mixed_equivalence_test_1
    >>> rng = np.random.default_rng(0)
    >>> n_clusters, n_metabolites = 6, 3
    >>> clusters, metabolites, samples, y, x = [], [], [], [], []
    >>> for c in range(n_clusters):
    ...     for m in range(n_metabolites):
    ...         yv = 5.0 + m * 2.0 + rng.normal(0, 0.5)
    ...         xv = yv + 0.1 + rng.normal(0, 0.3)
    ...         clusters.append(f"cluster{c}")
    ...         metabolites.append(f"metab{m}")
    ...         samples.append(f"sample{c}_{m}")
    ...         y.append(yv)
    ...         x.append(xv)
    >>> y = np.array(y)
    >>> x = np.array(x)
    >>> d = x - y
    >>> result = mixed_equivalence_test_1(
    ...     d, y, np.array(metabolites), np.array(clusters), np.array(samples),
    ... )
    >>> result.meta.status.value
    'valid'
    >>> print(f"{result.alpha:.4f} {result.alpha_z:.4f} {result.alpha_p_value:.4f}")
    0.1272 2.7540 0.0059
    >>> result.n_metabolites, result.n_clusters
    (3, 6)
    """

    all_ids = tuple(np.asarray(sample_ids, dtype=str).tolist())
    d = np.asarray(differences, dtype=np.float64)
    y = np.asarray(reference_values, dtype=np.float64)
    metab = np.asarray(metabolite_ids, dtype=str)
    clusters = np.asarray(cluster_ids, dtype=str)

    finite = np.isfinite(d) & np.isfinite(y)
    included_ids = tuple(np.asarray(all_ids)[finite].tolist())
    n_metabolites = int(np.unique(metab[finite]).shape[0]) if np.any(finite) else 0
    n_clusters = int(np.unique(clusters[finite]).shape[0]) if np.any(finite) else 0

    if np.sum(finite) < 4 or n_metabolites < 2:
        meta = build_meta(
            status=ResultStatus.INSUFFICIENT_DATA,
            all_ids=all_ids,
            included_ids=included_ids,
            method="mixed_equivalence_test_1",
        )
        return MixedTest1Result(
            meta=meta, alpha=float("nan"), beta=float("nan"), tau_alpha_sq=float("nan"),
            alpha_se_cluster_robust=float("nan"), alpha_z=float("nan"), alpha_p_value=float("nan"),
            n_metabolites=n_metabolites, n_clusters=n_clusters,
        )

    dv, yv, metabv, clusterv = d[finite], y[finite], metab[finite], clusters[finite]
    df = pd.DataFrame({"d": dv, "y": yv, "metabolite": metabv})
    df["y_centered"] = df.groupby("metabolite")["y"].transform(lambda s: s - s.mean())

    try:
        model = MixedLM(df["d"], np.column_stack([np.ones(len(df)), df["y_centered"]]), groups=df["metabolite"])
        fit = model.fit(reml=True)
        alpha, beta = float(fit.fe_params.iloc[0]), float(fit.fe_params.iloc[1])
        tau_alpha_sq = float(np.asarray(fit.cov_re)[0, 0])
    except Exception as exc:
        meta = build_meta(
            status=ResultStatus.FAILED,
            all_ids=all_ids,
            included_ids=included_ids,
            method="mixed_equivalence_test_1",
            warnings=(f"MixedLM failed to converge: {exc}",),
        )
        return MixedTest1Result(
            meta=meta, alpha=float("nan"), beta=float("nan"), tau_alpha_sq=float("nan"),
            alpha_se_cluster_robust=float("nan"), alpha_z=float("nan"), alpha_p_value=float("nan"),
            n_metabolites=n_metabolites, n_clusters=n_clusters,
        )

    design = np.column_stack([np.ones(len(df)), df["y_centered"].to_numpy()])
    residuals = dv - design @ np.array([alpha, beta])
    se = _cluster_robust_se(design, residuals, clusterv)
    alpha_se = float(se[0])
    alpha_z = alpha / alpha_se if alpha_se > 0 else float("nan")
    alpha_p = float(2 * _stats.norm.sf(np.abs(alpha_z))) if np.isfinite(alpha_z) else float("nan")

    meta = build_meta(
        status=ResultStatus.VALID,
        all_ids=all_ids,
        included_ids=included_ids,
        method="mixed_equivalence_test_1",
        configuration={
            "fit_method": "statsmodels.MixedLM(REML), random intercept by metabolite",
            "se_method": "cluster_robust_sandwich_on_fixed_effects_design",
            "n_metabolites": n_metabolites,
            "n_clusters": n_clusters,
        },
    )
    return MixedTest1Result(
        meta=meta, alpha=alpha, beta=beta, tau_alpha_sq=tau_alpha_sq,
        alpha_se_cluster_robust=alpha_se, alpha_z=alpha_z, alpha_p_value=alpha_p,
        n_metabolites=n_metabolites, n_clusters=n_clusters,
    )


@dataclass(frozen=True)
class MixedTest2Result(AnalysisResult):
    """Result of :func:`mixed_equivalence_test_2` (model-level precision).

    Attributes
    ----------
    meta : AnalysisMeta
        Provenance and inclusion/exclusion record for this computation.
    alpha : float
        Fixed-effect intercept.
    beta : float
        Fixed-effect slope on ``candidate + reference``
        (population-average precision term across metabolites).
    tau_beta_sq : float
        Estimated variance of the random slope across metabolites,
        ``Var(v_i)``.
    beta_se_cluster_robust : float
        Cluster-robust standard error of `beta` (see
        :func:`_cluster_robust_se`).
    beta_z : float
        z-statistic for H0: ``beta = 0``, ``beta / beta_se_cluster_robust``.
    beta_p_value : float
        Two-sided p-value for `beta_z` against a standard normal
        distribution.
    n_metabolites : int
        Number of distinct metabolites included in the fit.
    n_clusters : int
        Number of distinct clusters (spectra/subjects) included in the
        fit.
    """

    alpha: float
    beta: float
    tau_beta_sq: float
    beta_se_cluster_robust: float
    beta_z: float
    beta_p_value: float
    n_metabolites: int
    n_clusters: int


def mixed_equivalence_test_2(
    differences: NDArray[np.float64],
    reference_values: NDArray[np.float64],
    candidate_values: NDArray[np.float64],
    metabolite_ids: NDArray[np.str_],
    cluster_ids: NDArray[np.str_],
    sample_ids: NDArray[np.str_],
) -> MixedTest2Result:
    r"""Model-level equivalence Test 2: precision, pooled across metabolites.

    Fits the mixed model

    .. math::

        x_{i,n} - y_{i,n} = (\alpha + u_i)
            + (\beta + v_i) \, (y_{i,n} + x_{i,n}) + \theta_{i,n}

    where :math:`i` indexes metabolite, :math:`n` indexes sample, and
    :math:`u_i`/:math:`v_i` are random intercept/slope terms by
    metabolite, then tests H0: ``beta = 0`` using a cluster-robust
    standard error.

    Parameters
    ----------
    differences : ndarray of float, shape (n_obs,)
        Per-observation differences ``x_{i,n} - y_{i,n}``
        (candidate - reference).
    reference_values : ndarray of float, shape (n_obs,)
        Per-observation reference values ``y_{i,n}``, aligned with
        `differences`.
    candidate_values : ndarray of float, shape (n_obs,)
        Per-observation candidate values ``x_{i,n}``, aligned with
        `differences`.
    metabolite_ids : ndarray of str, shape (n_obs,)
        Metabolite label for each observation; the random-effects
        grouping factor for both the intercept and the slope.
    cluster_ids : ndarray of str, shape (n_obs,)
        Cluster label for each observation (spectrum id for *in silico*
        data, subject id for *in vivo* data); the unit resampled/
        clustered for the sandwich standard error.
    sample_ids : ndarray of str, shape (n_obs,)
        Per-observation identifiers, used for inclusion/exclusion
        bookkeeping.

    Returns
    -------
    MixedTest2Result
        Fixed-effect estimates, the random-slope variance, and the
        precision test statistics. See :class:`MixedTest2Result` for the
        full field list.

    Notes
    -----
    Fit via ``statsmodels.MixedLM`` with REML, a random intercept and
    random slope (on ``candidate + reference``) grouped by
    `metabolite_ids`. The reported standard error is the cluster-robust
    sandwich estimate on the fixed-effects design (see module Notes on
    the "bread" matrix simplification), not ``MixedLM``'s own
    model-based standard error.

    Fewer than 4 valid observations, or fewer than 2 distinct
    metabolites, yields a NaN-filled result with
    ``meta.status = ResultStatus.INSUFFICIENT_DATA``. A non-converging
    or erroring fit yields ``meta.status = ResultStatus.FAILED`` with
    the underlying exception message recorded in ``meta.warnings``.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.mixed_effects import mixed_equivalence_test_2
    >>> rng = np.random.default_rng(0)
    >>> n_clusters, n_metabolites = 6, 3
    >>> clusters, metabolites, samples, y, x = [], [], [], [], []
    >>> for c in range(n_clusters):
    ...     for m in range(n_metabolites):
    ...         yv = 5.0 + m * 2.0 + rng.normal(0, 0.5)
    ...         xv = yv + 0.1 + rng.normal(0, 0.3)
    ...         clusters.append(f"cluster{c}")
    ...         metabolites.append(f"metab{m}")
    ...         samples.append(f"sample{c}_{m}")
    ...         y.append(yv)
    ...         x.append(xv)
    >>> y = np.array(y)
    >>> x = np.array(x)
    >>> d = x - y
    >>> result = mixed_equivalence_test_2(
    ...     d, y, x, np.array(metabolites), np.array(clusters), np.array(samples),
    ... )
    >>> result.meta.status.value
    'valid'
    >>> print(f"{result.beta:.4f} {result.beta_z:.4f} {result.beta_p_value:.4f}")
    0.0882 2.5698 0.0102
    """

    all_ids = tuple(np.asarray(sample_ids, dtype=str).tolist())
    d = np.asarray(differences, dtype=np.float64)
    y = np.asarray(reference_values, dtype=np.float64)
    x = np.asarray(candidate_values, dtype=np.float64)
    metab = np.asarray(metabolite_ids, dtype=str)
    clusters = np.asarray(cluster_ids, dtype=str)

    finite = np.isfinite(d) & np.isfinite(y) & np.isfinite(x)
    included_ids = tuple(np.asarray(all_ids)[finite].tolist())
    n_metabolites = int(np.unique(metab[finite]).shape[0]) if np.any(finite) else 0
    n_clusters = int(np.unique(clusters[finite]).shape[0]) if np.any(finite) else 0

    if np.sum(finite) < 4 or n_metabolites < 2:
        meta = build_meta(
            status=ResultStatus.INSUFFICIENT_DATA,
            all_ids=all_ids,
            included_ids=included_ids,
            method="mixed_equivalence_test_2",
        )
        return MixedTest2Result(
            meta=meta, alpha=float("nan"), beta=float("nan"), tau_beta_sq=float("nan"),
            beta_se_cluster_robust=float("nan"), beta_z=float("nan"), beta_p_value=float("nan"),
            n_metabolites=n_metabolites, n_clusters=n_clusters,
        )

    dv, sumv, metabv, clusterv = d[finite], (x + y)[finite], metab[finite], clusters[finite]
    df = pd.DataFrame({"d": dv, "sum_xy": sumv, "metabolite": metabv})
    exog = np.column_stack([np.ones(len(df)), df["sum_xy"]])

    try:
        model = MixedLM(df["d"], exog, groups=df["metabolite"], exog_re=exog)
        fit = model.fit(reml=True)
        alpha, beta = float(fit.fe_params.iloc[0]), float(fit.fe_params.iloc[1])
        tau_beta_sq = float(np.asarray(fit.cov_re)[1, 1])
    except Exception as exc:
        meta = build_meta(
            status=ResultStatus.FAILED,
            all_ids=all_ids,
            included_ids=included_ids,
            method="mixed_equivalence_test_2",
            warnings=(f"MixedLM failed to converge: {exc}",),
        )
        return MixedTest2Result(
            meta=meta, alpha=float("nan"), beta=float("nan"), tau_beta_sq=float("nan"),
            beta_se_cluster_robust=float("nan"), beta_z=float("nan"), beta_p_value=float("nan"),
            n_metabolites=n_metabolites, n_clusters=n_clusters,
        )

    residuals = dv - exog @ np.array([alpha, beta])
    se = _cluster_robust_se(exog, residuals, clusterv)
    beta_se = float(se[1])
    beta_z = beta / beta_se if beta_se > 0 else float("nan")
    beta_p = float(2 * _stats.norm.sf(np.abs(beta_z))) if np.isfinite(beta_z) else float("nan")

    meta = build_meta(
        status=ResultStatus.VALID,
        all_ids=all_ids,
        included_ids=included_ids,
        method="mixed_equivalence_test_2",
        configuration={
            "fit_method": "statsmodels.MixedLM(REML), random intercept+slope by metabolite",
            "se_method": "cluster_robust_sandwich_on_fixed_effects_design",
            "n_metabolites": n_metabolites,
            "n_clusters": n_clusters,
        },
    )
    return MixedTest2Result(
        meta=meta, alpha=alpha, beta=beta, tau_beta_sq=tau_beta_sq,
        beta_se_cluster_robust=beta_se, beta_z=beta_z, beta_p_value=beta_p,
        n_metabolites=n_metabolites, n_clusters=n_clusters,
    )


@dataclass(frozen=True)
class MixedTest3Result(AnalysisResult):
    """Result of :func:`mixed_equivalence_test_3` (model-level agreement).

    Attributes
    ----------
    meta : AnalysisMeta
        Provenance and inclusion/exclusion record for this computation.
    alpha : float
        Point estimate of the pooled Deming intercept, across all
        metabolites and clusters.
    beta : float
        Point estimate of the pooled Deming slope.
    alpha_ci : tuple of (float, float)
        Cluster-bootstrap confidence interval for `alpha`, at
        `confidence_level_per_parameter`.
    beta_ci : tuple of (float, float)
        Cluster-bootstrap confidence interval for `beta`, at
        `confidence_level_per_parameter`.
    reject_joint_null : bool
        True if either `alpha_ci` excludes 0 or `beta_ci` excludes 1
        (i.e. the joint null of perfect model-level agreement is
        rejected).
    confidence_level_per_parameter : float
        Per-parameter confidence level actually used,
        ``1 - global_alpha / 2`` (the Bonferroni-adjusted level).
    n_clusters : int
        Number of distinct clusters (spectra/subjects) included in the
        fit.
    """

    alpha: float
    beta: float
    alpha_ci: tuple[float, float]
    beta_ci: tuple[float, float]
    reject_joint_null: bool
    confidence_level_per_parameter: float
    n_clusters: int


def mixed_equivalence_test_3(
    reference_values: NDArray[np.float64],
    candidate_values: NDArray[np.float64],
    cluster_ids: NDArray[np.str_],
    sample_ids: NDArray[np.str_],
    *,
    reference_error_variance: float,
    test_error_variance: float,
    global_alpha: float = 0.05,
    n_bootstrap: int = 2000,
    seed: int | None = None,
) -> MixedTest3Result:
    """Model-level equivalence Test 3: agreement, pooled across metabolites.

    Fits a single pooled :func:`~mrsvalidate.analysis.method_comparison.deming`
    regression of `candidate_values` on `reference_values` across *all*
    metabolites and clusters at once, and characterizes the joint-null
    (``alpha = 0``, ``beta = 1``) sampling distribution with a cluster
    bootstrap: whole clusters (spectra/subjects) are resampled with
    replacement, carrying every metabolite that belongs to the chosen
    cluster.

    Parameters
    ----------
    reference_values : ndarray of float, shape (n_obs,)
        Per-observation reference values, pooled across metabolites.
    candidate_values : ndarray of float, shape (n_obs,)
        Per-observation candidate values, pooled across metabolites,
        aligned with `reference_values`.
    cluster_ids : ndarray of str, shape (n_obs,)
        Cluster label for each observation (spectrum id for *in silico*
        data, subject id for *in vivo* data); the resampling unit for the
        cluster bootstrap.
    sample_ids : ndarray of str, shape (n_obs,)
        Per-observation identifiers, used for inclusion/exclusion
        bookkeeping.
    reference_error_variance : float
        Assumed error variance of `reference_values`, passed through to
        the pooled :func:`deming` fit. Must be finite and strictly
        positive.
    test_error_variance : float
        Assumed error variance of `candidate_values`, passed through to
        the pooled :func:`deming` fit. Must be finite and strictly
        positive.
    global_alpha : float, default 0.05
        Global two-sided significance level for the joint test. Split
        Bonferroni-style across the two joint-null parameters to a
        per-parameter confidence level of ``1 - global_alpha / 2``.
    n_bootstrap : int, default 2000
        Number of cluster-bootstrap resamples used to build the
        confidence intervals.
    seed : int or None, default None
        Seed for the random number generator driving the cluster
        resampling; None draws fresh entropy.

    Returns
    -------
    MixedTest3Result
        Pooled point estimates, their cluster-bootstrap confidence
        intervals, and the joint-null rejection decision. See
        :class:`MixedTest3Result` for the full field list.

    Notes
    -----
    As stated in the module Notes, this is a documented simplification
    of the manuscript's model-level agreement test: each bootstrap
    replicate recomputes a *pooled* Deming fit rather than an explicit
    per-replicate metabolite-random-effects decomposition. The point
    estimate and every replicate delegate to
    :func:`~mrsvalidate.analysis.method_comparison.deming` (`reference_values`
    as ``y``, `candidate_values` as ``x``; see that function's own Notes
    for the regression-direction convention and the
    manuscript-consistency caveat governing its slope formula).

    Fewer than 4 valid observations, or fewer than 2 distinct clusters,
    yields a NaN-filled result with
    ``meta.status = ResultStatus.INSUFFICIENT_DATA``.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.mixed_effects import mixed_equivalence_test_3
    >>> rng = np.random.default_rng(0)
    >>> n_clusters, n_metabolites = 6, 3
    >>> clusters, samples, y, x = [], [], [], []
    >>> for c in range(n_clusters):
    ...     for m in range(n_metabolites):
    ...         yv = 5.0 + m * 2.0 + rng.normal(0, 0.5)
    ...         xv = yv + 0.1 + rng.normal(0, 0.3)
    ...         clusters.append(f"cluster{c}")
    ...         samples.append(f"sample{c}_{m}")
    ...         y.append(yv)
    ...         x.append(xv)
    >>> y = np.array(y)
    >>> x = np.array(x)
    >>> result = mixed_equivalence_test_3(
    ...     y, x, np.array(clusters), np.array(samples),
    ...     reference_error_variance=1.0, test_error_variance=1.0,
    ...     n_bootstrap=200, seed=0,
    ... )
    >>> result.meta.status.value
    'valid'
    >>> print(f"{result.alpha:.4f} {result.beta:.4f}")
    0.0137 1.0166
    >>> result.reject_joint_null, result.n_clusters
    (False, 6)
    """

    all_ids = tuple(np.asarray(sample_ids, dtype=str).tolist())
    ref = np.asarray(reference_values, dtype=np.float64)
    cand = np.asarray(candidate_values, dtype=np.float64)
    clusters = np.asarray(cluster_ids, dtype=str)
    finite = np.isfinite(ref) & np.isfinite(cand)
    included_ids = tuple(np.asarray(all_ids)[finite].tolist())

    unique_clusters = np.unique(clusters[finite])
    n_clusters = unique_clusters.shape[0]

    if np.sum(finite) < 4 or n_clusters < 2:
        meta = build_meta(
            status=ResultStatus.INSUFFICIENT_DATA,
            all_ids=all_ids,
            included_ids=included_ids,
            method="mixed_equivalence_test_3",
        )
        return MixedTest3Result(
            meta=meta, alpha=float("nan"), beta=float("nan"),
            alpha_ci=(float("nan"), float("nan")), beta_ci=(float("nan"), float("nan")),
            reject_joint_null=False, confidence_level_per_parameter=1.0 - global_alpha / 2.0,
            n_clusters=n_clusters,
        )

    refv, candv, clusterv, idv = ref[finite], cand[finite], clusters[finite], np.asarray(all_ids)[finite]
    point = deming(
        refv, candv, idv,
        reference_error_variance=reference_error_variance, test_error_variance=test_error_variance,
    )

    cluster_to_idx = {c: np.flatnonzero(clusterv == c) for c in unique_clusters}
    rng = np.random.default_rng(seed)
    boot_alpha = np.empty(n_bootstrap)
    boot_beta = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        chosen = rng.choice(unique_clusters, size=n_clusters, replace=True)
        idx = np.concatenate([cluster_to_idx[c] for c in chosen])
        rep = deming(
            refv[idx], candv[idx], idv[idx],
            reference_error_variance=reference_error_variance, test_error_variance=test_error_variance,
        )
        boot_alpha[b] = rep.intercept
        boot_beta[b] = rep.slope

    per_param_level = 1.0 - global_alpha / 2.0
    tail = (1.0 - per_param_level) / 2.0
    alpha_ci = tuple(np.quantile(boot_alpha, [tail, 1 - tail]))
    beta_ci = tuple(np.quantile(boot_beta, [tail, 1 - tail]))
    reject = not (alpha_ci[0] <= 0.0 <= alpha_ci[1] and beta_ci[0] <= 1.0 <= beta_ci[1])

    meta = build_meta(
        status=ResultStatus.VALID,
        all_ids=all_ids,
        included_ids=included_ids,
        method="mixed_equivalence_test_3",
        configuration={
            "ci_method": "cluster_bootstrap_percentile_pooled_deming",
            "n_bootstrap": n_bootstrap,
            "bonferroni_global_alpha": global_alpha,
            "confidence_level_per_parameter": per_param_level,
            "n_clusters": n_clusters,
        },
        random_seed=seed,
    )
    return MixedTest3Result(
        meta=meta, alpha=point.intercept, beta=point.slope,
        alpha_ci=alpha_ci, beta_ci=beta_ci, reject_joint_null=reject,
        confidence_level_per_parameter=per_param_level, n_clusters=n_clusters,
    )
