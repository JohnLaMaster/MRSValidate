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
Notation follows the manuscript: ``x`` is the candidate (DL) value,
``y`` the reference value, ``i`` indexes metabolite and ``n`` indexes
spectrum.

Tests 1-2 map onto standard mixed-effects likelihoods (random
intercept / random intercept+slope by metabolite) and are fit with
``statsmodels.MixedLM``, per spec guidance that Statsmodels is
appropriate reusable infrastructure *where its likelihood matches the
intended model*. Fixed-effect standard errors are cluster-robust,
clustered on spectrum (in-silico) or subject (in-vivo), via the
manuscript's own sandwich formula (implemented directly here — not a
Statsmodels built-in for a mixed model), with a small-sample correction
chosen by the number of clusters ``G``: by default CR2 (Bell & McCaffrey
2002) with Bell-McCaffrey/Satterthwaite degrees of freedom (Imbens &
Kolesár 2016) when ``G < 50``, otherwise CR1 with a ``t(G - 1)``
reference (Cameron, Gelbach & Miller 2008).

Test 3 combines errors-in-variables (Deming-type) regression with
random effects across metabolites, which is **not** a stock
``MixedLM`` likelihood (spec: implement as its own model/cluster-bootstrap
procedure). It is implemented as a *two-stage* model. Stage 1 fits, for
every metabolite ``i`` separately, a York et al. (2004) errors-in-
variables regression of candidate on reference with per-observation
error variances (the per-observation ``lambda_{i,n}`` of ``me_test3``).
Stage 2 synthesizes the per-metabolite intercepts ``a_i`` and slopes
``b_i`` with a random-effects (REML) model, giving the fixed effects
``alpha``/``beta`` and the heterogeneity variances ``tau_alpha^2``/
``tau_beta^2``. Inference is a cluster bootstrap: independent clusters
(spectra/subjects) are resampled with replacement, carrying every
metabolite for that cluster, and both stages are rerun per replicate.

**Estimand and inference scope.** The tests compare two *fitting
methods* (candidate vs. reference) over the evaluated joint metabolite
set, which is fixed by the basis set rather than sampled from a
population of metabolites. Inference is therefore *panel-conditional*
by default: the fixed effects (``alpha``, ``beta``) summarise agreement
between the two fitting methods over the evaluated joint metabolite set;
the cluster-robust sandwich (Tests 1-2) and the cluster bootstrap
(Test 3) capture sampling variability across spectra (in silico) or
subjects (in vivo); and the random-effect variances ``tau^2`` are
descriptive measures of heterogeneity across the joint set, reported
alongside the fixed effects but not propagated into their uncertainty.
Tests 1-2 (``MixedLM`` + cluster-robust sandwich) are panel-conditional
only. For users who want to generalise beyond the evaluated panel,
Test 3 additionally offers ``inference="population"``: a two-way
pigeonhole bootstrap (Owen 2007) that resamples metabolites and clusters
independently, adding between-metabolite variation to the uncertainty
of ``alpha`` and ``beta``. The effective number of units is then the
number of metabolites, so intervals are wide for typical panels of
20-30 metabolites.

For Test 3 the panel-conditional estimand is the REML precision-weighted
average of the per-metabolite parameters, with weights
``1 / (s_i^2 + tau^2)`` (``s_i^2`` the stage-1 sampling variance). It
equals the unweighted panel average when the stage-1 precisions are
equal and approaches it as the number of clusters grows (``s_i^2 -> 0``);
with few clusters, large ``tau^2`` and very unequal ``s_i^2`` the two
can differ appreciably.

**Simplifications, stated explicitly (this is the spec's own
highest-complexity item, flagged as needing Statistical Reviewer
sign-off):**

* The cluster-robust "bread" matrix for Tests 1-2 uses ``(X'X)^-1`` on
  the fixed-effects design (not the mixed model's full GLS information
  matrix), and the CR2 leverage adjustment ``(I - H_gg)^(-1/2)`` uses
  the matching OLS hat matrix ``H = X (X'X)^-1 X'``; the residuals are
  the marginal residuals of the ``MixedLM`` fixed-effect fit. The
  Bell-McCaffrey degrees of freedom assume a working homoskedastic,
  independent error covariance (``Omega = I``).
* Test 3 is a two-stage approximation to the joint mixed-effects
  Deming model of ``me_test3``, not a joint likelihood: the stage-2
  synthesis is univariate per parameter (the within-metabolite
  covariance of ``a_i`` and ``b_i`` from stage 1 is not used), and the
  stage-1 York standard errors are the error-variance-based ("expected")
  ones, i.e. they trust the supplied error variances; any misspecification
  is absorbed into ``tau^2`` and, for inference, handled by the cluster
  bootstrap.
* The random-effects formulation is used to model heterogeneity across
  the joint metabolite set (and, for Tests 1-2, to obtain the
  ``MixedLM`` fixed-effect fit), not to define a population of
  metabolites: by default, all inference is panel-conditional (see
  *Estimand and inference scope* above). The ``inference="population"``
  option of Test 3 is the only route to population-level inference; the
  pigeonhole bootstrap is known to be conservative for crossed designs
  (Owen 2007), since the stage-1 sampling error is counted both through
  the metabolite resampling and through the cluster resampling.

These are documented, reduced approximations to the full model described
in the manuscript, not a claim of an exact, published solution.

References
----------
Bell, R. M. & McCaffrey, D. F. (2002). Bias reduction in standard errors
for linear regression with multi-stage samples. *Survey Methodology*
28(2), 169-181.

Imbens, G. W. & Kolesár, M. (2016). Robust standard errors in small
samples: some practical advice. *Rev. Econ. Stat.* 98(4), 701-712.

Owen, A. B. (2007). The pigeonhole bootstrap. *Ann. Appl. Stat.* 1(2),
502-525.

York, D., Evensen, N. M., López Martínez, M. & De Basabe Delgado, J.
(2004). Unified equations for the slope, intercept, and standard errors
of the best straight line. *Am. J. Phys.* 72(3), 367-375.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats as _stats
from statsmodels.regression.mixed_linear_model import MixedLM

from ..results import AnalysisResult, ResultStatus, build_meta

__all__ = [
    "MixedTest1Result",
    "MixedTest2Result",
    "MixedTest3Result",
    "mixed_equivalence_test_1",
    "mixed_equivalence_test_2",
    "mixed_equivalence_test_3",
]

_SMALL_SAMPLE_CHOICES = ("auto", "CR0", "CR1", "CR2")
_AUTO_CR2_CLUSTER_THRESHOLD = 50
_CR2_EIGEN_FLOOR = 1e-12


# ---------------------------------------------------------------------------
# Cluster-robust inference (Tests 1-2)
# ---------------------------------------------------------------------------


def _resolve_small_sample(small_sample: str, n_clusters: int) -> str:
    """Resolve the ``small_sample`` option to a concrete CR method.

    Parameters
    ----------
    small_sample : {"auto", "CR0", "CR1", "CR2"}
        Requested correction.
    n_clusters : int
        Number of distinct clusters ``G``.

    Returns
    -------
    str
        ``"CR0"``, ``"CR1"`` or ``"CR2"``; ``"auto"`` resolves to
        ``"CR2"`` when ``G < 50`` and ``"CR1"`` otherwise.

    Raises
    ------
    ValueError
        If `small_sample` is not one of the supported choices.
    """

    if small_sample not in _SMALL_SAMPLE_CHOICES:
        raise ValueError(f"small_sample must be one of {_SMALL_SAMPLE_CHOICES}, got {small_sample!r}")
    if small_sample == "auto":
        return "CR2" if n_clusters < _AUTO_CR2_CLUSTER_THRESHOLD else "CR1"
    return small_sample


def _cluster_robust_inference(
    design: NDArray[np.float64],
    residuals: NDArray[np.float64],
    cluster_ids: NDArray[np.str_],
    method: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[str, ...]]:
    r"""Cluster-robust sandwich standard errors with small-sample corrections.

    Parameters
    ----------
    design : ndarray of float, shape (n_obs, n_params)
        Fixed-effects design matrix ``X``.
    residuals : ndarray of float, shape (n_obs,)
        Marginal residuals ``y - X @ fixed_effect_estimate``.
    cluster_ids : ndarray of str, shape (n_obs,)
        Cluster membership (spectrum or subject) for each observation.
    method : {"CR0", "CR1", "CR2"}
        Resolved small-sample correction (see
        :func:`_resolve_small_sample`).

    Returns
    -------
    se : ndarray of float, shape (n_params,)
        Cluster-robust standard error of each coefficient.
    df : ndarray of float, shape (n_params,)
        Reference degrees of freedom per coefficient: ``inf`` (normal
        reference) for CR0, ``G - 1`` for CR1, and the Bell-McCaffrey
        Satterthwaite degrees of freedom for CR2.
    warnings : tuple of str
        Numerical warnings (e.g. near-unit cluster leverage under CR2).

    Notes
    -----
    With bread :math:`B = (X^\top X)^{-1}` (Moore-Penrose pseudoinverse),

    * CR0: :math:`V = B \left(\sum_g X_g^\top e_g e_g^\top X_g\right) B`.
    * CR1: :math:`V_{CR0} \cdot \frac{G}{G-1} \frac{N-1}{N-p}`.
    * CR2: the meat uses :math:`A_g e_g` with
      :math:`A_g = (I - H_{gg})^{-1/2}`,
      :math:`H_{gg} = X_g B X_g^\top`, computed by symmetric
      eigendecomposition; eigenvalues of :math:`I - H_{gg}` at or below
      ``1e-12`` are treated by the pseudo-inverse square root (and a
      warning is returned).

    The CR2 degrees of freedom for coefficient ``j`` (contrast
    :math:`c = e_j`) are :math:`(\operatorname{tr} M)^2 /
    \operatorname{tr}(M^2)` with :math:`M = G^\top G`, where column
    ``g`` of the ``N x G`` matrix :math:`G` is
    :math:`(I - H)_{\cdot g} A_g X_g B c` (Imbens & Kolesár 2016). The
    ``N x N`` matrix is never formed: writing :math:`v_g = A_g X_g B c`,
    :math:`w_g = X_g^\top v_g`, :math:`u_g = B w_g` and
    :math:`s_g = v_g^\top v_g`,
    :math:`M = D_s - W U^\top - U W^\top + U (X^\top X) U^\top`, a
    diagonal plus rank-``2p`` matrix whose trace and squared Frobenius
    norm are evaluated in ``O(G p^2)``.
    """

    x = np.asarray(design, dtype=np.float64)
    e = np.asarray(residuals, dtype=np.float64)
    n_obs, p = x.shape
    _, codes = np.unique(np.asarray(cluster_ids, dtype=str), return_inverse=True)
    n_clusters = int(codes.max()) + 1
    order = np.argsort(codes, kind="stable")
    bounds = np.flatnonzero(np.diff(codes[order])) + 1
    groups = np.split(order, bounds)

    xtx = x.T @ x
    bread = np.linalg.pinv(xtx)
    warnings: list[str] = []

    if method in ("CR0", "CR1"):
        scores = np.zeros((n_clusters, p))
        np.add.at(scores, codes, x * e[:, None])
        cov = bread @ (scores.T @ scores) @ bread
        if method == "CR1":
            if n_clusters < 2 or n_obs <= p:
                return np.full(p, np.nan), np.full(p, np.nan), ("CR1 requires G >= 2 and N > p",)
            cov = cov * (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - p))
            df = np.full(p, float(n_clusters - 1))
        else:
            df = np.full(p, np.inf)
        return np.sqrt(np.clip(np.diag(cov), 0.0, None)), df, ()

    if method != "CR2":
        raise ValueError(f"unknown cluster-robust method {method!r}")

    meat = np.zeros((p, p))
    w_all = np.empty((n_clusters, p, p))  # w_all[g, :, j] = X_g' v_g(e_j)
    s_all = np.empty((n_clusters, p))  # s_all[g, j] = v_g(e_j)' v_g(e_j)
    n_floored = 0
    for g, idx in enumerate(groups):
        xg = x[idx]
        h_gg = xg @ bread @ xg.T
        evals, evecs = np.linalg.eigh(np.eye(len(idx)) - 0.5 * (h_gg + h_gg.T))
        keep = evals > _CR2_EIGEN_FLOOR
        n_floored += int(np.sum(~keep))
        inv_sqrt = np.zeros_like(evals)
        inv_sqrt[keep] = 1.0 / np.sqrt(evals[keep])
        a_g = (evecs * inv_sqrt) @ evecs.T
        score = xg.T @ (a_g @ e[idx])
        meat += np.outer(score, score)
        v_g = a_g @ xg @ bread  # (n_g, p): column j is v_g for c = e_j
        s_all[g] = np.sum(v_g**2, axis=0)
        w_all[g] = xg.T @ v_g
    if n_floored:
        warnings.append(
            f"CR2: {n_floored} eigenvalue(s) of I - H_gg were <= {_CR2_EIGEN_FLOOR:g} "
            "(cluster leverage ~1); used the pseudo-inverse square root for those directions"
        )
    cov = bread @ meat @ bread
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    k_mat = np.zeros((2 * p, 2 * p))
    k_mat[:p, p:] = -np.eye(p)
    k_mat[p:, :p] = -np.eye(p)
    k_mat[p:, p:] = xtx
    df = np.empty(p)
    for j in range(p):
        w_j = w_all[:, :, j]  # (G, p)
        u_j = w_j @ bread  # (G, p); bread is symmetric
        z = np.hstack([w_j, u_j])  # (G, 2p)
        s_j = s_all[:, j]
        low_rank_diag = np.einsum("gi,ij,gj->g", z, k_mat, z)
        trace_m = np.sum(s_j) + np.sum(low_rank_diag)
        kztz = k_mat @ (z.T @ z)
        trace_m2 = np.sum(s_j**2) + 2.0 * np.sum(s_j * low_rank_diag) + np.trace(kztz @ kztz)
        df[j] = trace_m**2 / trace_m2 if trace_m2 > 0 else np.nan
    return se, df, tuple(warnings)


def _t_test(estimate: float, se: float, df: float) -> tuple[float, float]:
    """Return the Wald statistic ``estimate / se`` and its two-sided p-value.

    The p-value uses a standard normal reference when ``df`` is infinite
    and Student's t with ``df`` degrees of freedom otherwise.
    """

    if not (np.isfinite(se) and se > 0 and (np.isfinite(df) or df == np.inf)):
        return float("nan"), float("nan")
    stat = estimate / se
    if np.isinf(df):
        p = 2.0 * _stats.norm.sf(abs(stat))
    elif df > 0:
        p = 2.0 * _stats.t.sf(abs(stat), df)
    else:
        p = float("nan")
    return float(stat), float(p)


def _se_method_description(method: str) -> str:
    return {
        "CR0": "CR0 cluster-robust sandwich, (X'X)^-1 bread, normal reference",
        "CR1": "CR1 cluster-robust sandwich (G/(G-1)*(N-1)/(N-p)), (X'X)^-1 bread, t(G-1) reference",
        "CR2": "CR2 (Bell-McCaffrey) cluster-robust sandwich, (X'X)^-1 bread and OLS hat matrix, "
        "t reference with Bell-McCaffrey/Satterthwaite df (Imbens-Kolesar 2016, Omega = I)",
    }[method]


# ---------------------------------------------------------------------------
# Test 1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MixedTest1Result(AnalysisResult):
    """Result of :func:`mixed_equivalence_test_1` (model-level accuracy).

    Attributes
    ----------
    meta : AnalysisMeta
        Provenance and inclusion/exclusion record for this computation.
    alpha : float
        Fixed-effect intercept: model-level accuracy bias of the candidate
        relative to the reference fitting method, summarised over the
        evaluated joint metabolite set.
    beta : float
        Fixed-effect slope on the metabolite-mean-centered reference.
    tau_alpha_sq : float
        Estimated variance of the random intercept, ``Var(u_i)``: a
        descriptive measure of heterogeneity in constant bias across the
        joint metabolite set (not propagated into `alpha_se_cluster_robust`).
    alpha_se_cluster_robust : float
        Cluster-robust standard error of `alpha` under `se_method`.
    alpha_t : float
        Wald statistic for H0: ``alpha = 0``,
        ``alpha / alpha_se_cluster_robust``. Referred to Student's t with
        `df` degrees of freedom (a z statistic when ``df = inf``, i.e.
        under CR0).
    alpha_p_value : float
        Two-sided p-value for `alpha_t` against ``t(df)`` (standard
        normal when ``df = inf``).
    se_method : str
        Resolved small-sample correction actually used: ``"CR0"``,
        ``"CR1"`` or ``"CR2"``.
    df : float
        Reference degrees of freedom for the `alpha` test: ``inf`` for
        CR0, ``G - 1`` for CR1, Bell-McCaffrey degrees of freedom for
        CR2.
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
    alpha_t: float
    alpha_p_value: float
    se_method: str
    df: float
    n_metabolites: int
    n_clusters: int


def mixed_equivalence_test_1(
    differences: NDArray[np.float64],
    reference_values: NDArray[np.float64],
    metabolite_ids: NDArray[np.str_],
    cluster_ids: NDArray[np.str_],
    sample_ids: NDArray[np.str_],
    *,
    small_sample: str = "auto",
) -> MixedTest1Result:
    r"""Model-level equivalence Test 1: accuracy, pooled across metabolites.

    Fits the mixed model

    .. math::

        x_{i,n} - y_{i,n} = (\alpha + u_i)
            + \beta \, (y_{i,n} - \bar{y}_i) + \nu_{i,n}

    where :math:`i` indexes metabolite, :math:`n` indexes spectrum, and
    :math:`u_i` is a random intercept by metabolite, then tests H0:
    ``alpha = 0`` using a small-sample-corrected cluster-robust standard
    error.

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
        data, subject id for *in vivo* data); the unit clustered for the
        sandwich standard error.
    sample_ids : ndarray of str, shape (n_obs,)
        Per-observation identifiers, used for inclusion/exclusion
        bookkeeping.
    small_sample : {"auto", "CR0", "CR1", "CR2"}, default "auto"
        Small-sample correction of the cluster-robust standard error.
        ``"auto"`` uses CR2 with Bell-McCaffrey degrees of freedom when
        the number of clusters ``G < 50`` and CR1 with a ``t(G - 1)``
        reference otherwise. ``"CR0"`` is the uncorrected sandwich with a
        standard normal reference.

    Returns
    -------
    MixedTest1Result
        Fixed-effect estimates, the random-intercept variance, and the
        accuracy test statistics. See :class:`MixedTest1Result` for the
        full field list.

    Raises
    ------
    ValueError
        If `small_sample` is not a supported choice.

    Notes
    -----
    Fit via ``statsmodels.MixedLM`` with REML and a random intercept
    grouped by `metabolite_ids`; the reference is centered per
    metabolite (``y_{i,n} - bar{y}_i``) before fitting. The reported
    standard error is the cluster-robust sandwich estimate on the
    fixed-effects design with the documented ``(X'X)^-1`` bread (see
    module Notes), not ``MixedLM``'s own model-based standard error.

    **Inference scope (panel-conditional).** ``alpha`` summarises the
    accuracy of the candidate relative to the reference fitting method
    over the evaluated joint metabolite set; the cluster-robust standard
    error captures sampling variability across spectra (or subjects)
    only, and `tau_alpha_sq` is reported as descriptive heterogeneity
    across the joint set. No population-of-metabolites option exists for
    this test (see :func:`mixed_equivalence_test_3`, ``inference=``).

    Fewer than 4 valid observations, fewer than 2 distinct metabolites,
    or fewer than 2 distinct clusters yields a NaN-filled result with
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
    >>> result.se_method
    'CR2'
    >>> print(f"{result.alpha:.4f} {result.alpha_t:.4f} {result.df:.2f} {result.alpha_p_value:.4f}")
    0.1272 2.5175 4.97 0.0536
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
    se_method = _resolve_small_sample(small_sample, n_clusters)

    def _nan_result(meta: Any) -> MixedTest1Result:
        return MixedTest1Result(
            meta=meta, alpha=float("nan"), beta=float("nan"), tau_alpha_sq=float("nan"),
            alpha_se_cluster_robust=float("nan"), alpha_t=float("nan"), alpha_p_value=float("nan"),
            se_method=se_method, df=float("nan"), n_metabolites=n_metabolites, n_clusters=n_clusters,
        )

    if np.sum(finite) < 4 or n_metabolites < 2 or n_clusters < 2:
        return _nan_result(build_meta(
            status=ResultStatus.INSUFFICIENT_DATA,
            all_ids=all_ids,
            included_ids=included_ids,
            method="mixed_equivalence_test_1",
            configuration={"small_sample": small_sample, "se_method": se_method},
        ))

    dv, yv, metabv, clusterv = d[finite], y[finite], metab[finite], clusters[finite]
    df = pd.DataFrame({"d": dv, "y": yv, "metabolite": metabv})
    df["y_centered"] = df.groupby("metabolite")["y"].transform(lambda s: s - s.mean())

    try:
        model = MixedLM(df["d"], np.column_stack([np.ones(len(df)), df["y_centered"]]), groups=df["metabolite"])
        fit = model.fit(reml=True)
        alpha, beta = float(fit.fe_params.iloc[0]), float(fit.fe_params.iloc[1])
        tau_alpha_sq = float(np.asarray(fit.cov_re)[0, 0])
    except Exception as exc:  # noqa: BLE001 -- statsmodels can raise many different exception types on convergence failure; caught broadly and deliberately so it is reported as ResultStatus.FAILED rather than propagating
        return _nan_result(build_meta(
            status=ResultStatus.FAILED,
            all_ids=all_ids,
            included_ids=included_ids,
            method="mixed_equivalence_test_1",
            warnings=(f"MixedLM failed to converge: {exc}",),
            configuration={"small_sample": small_sample, "se_method": se_method},
        ))

    design = np.column_stack([np.ones(len(df)), df["y_centered"].to_numpy()])
    residuals = dv - design @ np.array([alpha, beta])
    se, dof, se_warnings = _cluster_robust_inference(design, residuals, clusterv, se_method)
    alpha_se, alpha_df = float(se[0]), float(dof[0])
    alpha_t, alpha_p = _t_test(alpha, alpha_se, alpha_df)

    meta = build_meta(
        status=ResultStatus.VALID,
        all_ids=all_ids,
        included_ids=included_ids,
        method="mixed_equivalence_test_1",
        warnings=se_warnings,
        configuration={
            "fit_method": "statsmodels.MixedLM(REML), random intercept by metabolite",
            "small_sample": small_sample,
            "se_method": se_method,
            "se_method_description": _se_method_description(se_method),
            "auto_cr2_cluster_threshold": _AUTO_CR2_CLUSTER_THRESHOLD,
            "df": alpha_df,
            "bread": "(X'X)^-1 on fixed-effects design",
            "n_metabolites": n_metabolites,
            "n_clusters": n_clusters,
        },
    )
    return MixedTest1Result(
        meta=meta, alpha=alpha, beta=beta, tau_alpha_sq=tau_alpha_sq,
        alpha_se_cluster_robust=alpha_se, alpha_t=alpha_t, alpha_p_value=alpha_p,
        se_method=se_method, df=alpha_df, n_metabolites=n_metabolites, n_clusters=n_clusters,
    )


# ---------------------------------------------------------------------------
# Test 2
# ---------------------------------------------------------------------------


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
        Fixed-effect slope on ``candidate + reference``: model-level
        precision term of the two fitting methods, summarised over the
        evaluated joint metabolite set.
    tau_alpha_sq : float
        Estimated variance of the random intercept, ``Var(u_i)``
        (descriptive heterogeneity across the joint metabolite set).
    tau_beta_sq : float
        Estimated variance of the random slope, ``Var(v_i)``
        (descriptive heterogeneity across the joint metabolite set; not
        propagated into `beta_se_cluster_robust`).
    beta_se_cluster_robust : float
        Cluster-robust standard error of `beta` under `se_method`.
    beta_t : float
        Wald statistic for H0: ``beta = 0``,
        ``beta / beta_se_cluster_robust``. Referred to Student's t with
        `df` degrees of freedom (a z statistic when ``df = inf``, i.e.
        under CR0).
    beta_p_value : float
        Two-sided p-value for `beta_t` against ``t(df)`` (standard normal
        when ``df = inf``).
    se_method : str
        Resolved small-sample correction actually used: ``"CR0"``,
        ``"CR1"`` or ``"CR2"``.
    df : float
        Reference degrees of freedom for the `beta` test: ``inf`` for
        CR0, ``G - 1`` for CR1, Bell-McCaffrey degrees of freedom for
        CR2.
    n_metabolites : int
        Number of distinct metabolites included in the fit.
    n_clusters : int
        Number of distinct clusters (spectra/subjects) included in the
        fit.
    """

    alpha: float
    beta: float
    tau_alpha_sq: float
    tau_beta_sq: float
    beta_se_cluster_robust: float
    beta_t: float
    beta_p_value: float
    se_method: str
    df: float
    n_metabolites: int
    n_clusters: int


def mixed_equivalence_test_2(
    differences: NDArray[np.float64],
    reference_values: NDArray[np.float64],
    candidate_values: NDArray[np.float64],
    metabolite_ids: NDArray[np.str_],
    cluster_ids: NDArray[np.str_],
    sample_ids: NDArray[np.str_],
    *,
    small_sample: str = "auto",
) -> MixedTest2Result:
    r"""Model-level equivalence Test 2: precision, pooled across metabolites.

    Fits the mixed model

    .. math::

        x_{i,n} - y_{i,n} = (\alpha + u_i)
            + (\beta + v_i) \, (y_{i,n} + x_{i,n}) + \theta_{i,n}

    where :math:`i` indexes metabolite, :math:`n` indexes spectrum, and
    :math:`u_i`/:math:`v_i` are random intercept/slope terms by
    metabolite, then tests H0: ``beta = 0`` using a
    small-sample-corrected cluster-robust standard error.

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
        data, subject id for *in vivo* data); the unit clustered for the
        sandwich standard error.
    sample_ids : ndarray of str, shape (n_obs,)
        Per-observation identifiers, used for inclusion/exclusion
        bookkeeping.
    small_sample : {"auto", "CR0", "CR1", "CR2"}, default "auto"
        Small-sample correction of the cluster-robust standard error; see
        :func:`mixed_equivalence_test_1`.

    Returns
    -------
    MixedTest2Result
        Fixed-effect estimates, the random-effect variances, and the
        precision test statistics. See :class:`MixedTest2Result` for the
        full field list.

    Raises
    ------
    ValueError
        If `small_sample` is not a supported choice.

    Notes
    -----
    Fit via ``statsmodels.MixedLM`` with REML, a random intercept and
    random slope (on ``candidate + reference``) grouped by
    `metabolite_ids`. The reported standard error is the cluster-robust
    sandwich estimate on the fixed-effects design with the documented
    ``(X'X)^-1`` bread (see module Notes), not ``MixedLM``'s own
    model-based standard error.

    **Inference scope (panel-conditional).** ``beta`` summarises the
    relative precision of the candidate and reference fitting methods
    over the evaluated joint metabolite set; the cluster-robust standard
    error captures sampling variability across spectra (or subjects)
    only, and `tau_alpha_sq`/`tau_beta_sq` are reported as descriptive
    heterogeneity across the joint set. No population-of-metabolites
    option exists for
    this test (see :func:`mixed_equivalence_test_3`, ``inference=``).

    Fewer than 4 valid observations, fewer than 2 distinct metabolites,
    or fewer than 2 distinct clusters yields a NaN-filled result with
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
    >>> print(f"{result.beta:.4f} {result.beta_t:.4f} {result.df:.2f} {result.beta_p_value:.4f}")
    0.0882 2.2862 4.27 0.0800
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
    se_method = _resolve_small_sample(small_sample, n_clusters)

    def _nan_result(meta: Any) -> MixedTest2Result:
        return MixedTest2Result(
            meta=meta, alpha=float("nan"), beta=float("nan"), tau_alpha_sq=float("nan"),
            tau_beta_sq=float("nan"), beta_se_cluster_robust=float("nan"), beta_t=float("nan"),
            beta_p_value=float("nan"), se_method=se_method, df=float("nan"),
            n_metabolites=n_metabolites, n_clusters=n_clusters,
        )

    if np.sum(finite) < 4 or n_metabolites < 2 or n_clusters < 2:
        return _nan_result(build_meta(
            status=ResultStatus.INSUFFICIENT_DATA,
            all_ids=all_ids,
            included_ids=included_ids,
            method="mixed_equivalence_test_2",
            configuration={"small_sample": small_sample, "se_method": se_method},
        ))

    dv, sumv, metabv, clusterv = d[finite], (x + y)[finite], metab[finite], clusters[finite]
    df = pd.DataFrame({"d": dv, "sum_xy": sumv, "metabolite": metabv})
    exog = np.column_stack([np.ones(len(df)), df["sum_xy"]])

    try:
        model = MixedLM(df["d"], exog, groups=df["metabolite"], exog_re=exog)
        fit = model.fit(reml=True)
        alpha, beta = float(fit.fe_params.iloc[0]), float(fit.fe_params.iloc[1])
        cov_re = np.asarray(fit.cov_re)
        tau_alpha_sq, tau_beta_sq = float(cov_re[0, 0]), float(cov_re[1, 1])
    except Exception as exc:  # noqa: BLE001 -- statsmodels can raise many different exception types on convergence failure; caught broadly and deliberately so it is reported as ResultStatus.FAILED rather than propagating
        return _nan_result(build_meta(
            status=ResultStatus.FAILED,
            all_ids=all_ids,
            included_ids=included_ids,
            method="mixed_equivalence_test_2",
            warnings=(f"MixedLM failed to converge: {exc}",),
            configuration={"small_sample": small_sample, "se_method": se_method},
        ))

    residuals = dv - exog @ np.array([alpha, beta])
    se, dof, se_warnings = _cluster_robust_inference(exog, residuals, clusterv, se_method)
    beta_se, beta_df = float(se[1]), float(dof[1])
    beta_t, beta_p = _t_test(beta, beta_se, beta_df)

    meta = build_meta(
        status=ResultStatus.VALID,
        all_ids=all_ids,
        included_ids=included_ids,
        method="mixed_equivalence_test_2",
        warnings=se_warnings,
        configuration={
            "fit_method": "statsmodels.MixedLM(REML), random intercept+slope by metabolite",
            "small_sample": small_sample,
            "se_method": se_method,
            "se_method_description": _se_method_description(se_method),
            "auto_cr2_cluster_threshold": _AUTO_CR2_CLUSTER_THRESHOLD,
            "df": beta_df,
            "bread": "(X'X)^-1 on fixed-effects design",
            "n_metabolites": n_metabolites,
            "n_clusters": n_clusters,
        },
    )
    return MixedTest2Result(
        meta=meta, alpha=alpha, beta=beta, tau_alpha_sq=tau_alpha_sq, tau_beta_sq=tau_beta_sq,
        beta_se_cluster_robust=beta_se, beta_t=beta_t, beta_p_value=beta_p,
        se_method=se_method, df=beta_df, n_metabolites=n_metabolites, n_clusters=n_clusters,
    )


# ---------------------------------------------------------------------------
# Test 3: two-stage York + random-effects synthesis, cluster bootstrap
# ---------------------------------------------------------------------------

_YORK_TOL = 1e-12
_YORK_MAX_ITER = 500
_REML_TOL = 1e-10
_REML_MAX_ITER = 1000
_MIN_OBS_PER_METABOLITE = 3
_INFERENCE_CHOICES = ("panel", "population")
_INFERENCE_DESCRIPTIONS = {
    "panel": "panel-conditional: cluster bootstrap over spectra/subjects only; the evaluated joint metabolite "
    "set is fixed and tau^2 is descriptive heterogeneity across it",
    "population": "population (generalise beyond the panel): two-way pigeonhole bootstrap (Owen 2007) "
    "resampling metabolites and clusters independently with replacement; between-metabolite variation "
    "enters the uncertainty of alpha and beta, effective number of units ~ number of metabolites",
}


@dataclass(frozen=True)
class _YorkFit:
    """Grouped York fits; every array has shape (n_replicates, n_groups)."""

    intercept: NDArray[np.float64]
    slope: NDArray[np.float64]
    var_intercept: NDArray[np.float64]
    var_slope: NDArray[np.float64]
    cov_intercept_slope: NDArray[np.float64]
    converged: NDArray[np.bool_]
    n_distinct: NDArray[np.float64]


def _york_grouped(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    var_x: NDArray[np.float64],
    var_y: NDArray[np.float64],
    starts: NDArray[np.intp],
    freq: NDArray[np.float64],
    slope_start: NDArray[np.float64] | None = None,
) -> _YorkFit:
    r"""Vectorised York (2004) regression of ``y`` on ``x``, per group and replicate.

    Parameters
    ----------
    x, y : ndarray of float, shape (n_obs,)
        York's ``X`` (reference) and ``Y`` (candidate), **sorted by
        group**, so that group ``k`` occupies ``starts[k]:starts[k+1]``.
    var_x, var_y : ndarray of float, shape (n_obs,)
        Per-point error variances of `x` and `y` (strictly positive).
        Error correlation is taken as zero.
    starts : ndarray of int, shape (n_groups,)
        Start index of each (non-empty) group.
    freq : ndarray of float, shape (n_replicates, n_obs)
        Frequency weights (bootstrap multiplicities); a weight ``f``
        is exactly equivalent to duplicating the point ``f`` times.
    slope_start : ndarray of float, shape (n_replicates, n_groups), optional
        Starting slopes. Default: weighted OLS slope of ``y`` on ``x``.

    Returns
    -------
    _YorkFit
        Intercepts, slopes, their York standard-error variances and
        covariance, a convergence flag and the number of distinct points
        with positive weight, per (replicate, group).

    Notes
    -----
    With :math:`\omega_X = 1/\sigma^2_X`, :math:`\omega_Y = 1/\sigma^2_Y`,
    iterate :math:`W = \omega_X\omega_Y / (\omega_X + b^2\omega_Y)`,
    :math:`\bar X = \sum WX / \sum W`, :math:`\bar Y = \sum WY/\sum W`,
    :math:`U = X - \bar X`, :math:`V = Y - \bar Y`,
    :math:`\beta = W(U/\omega_Y + bV/\omega_X)`,
    :math:`b \leftarrow \sum W\beta V / \sum W\beta U` until
    :math:`|\Delta b| < 10^{-12}\max(1, |b|)`; then
    :math:`a = \bar Y - b\bar X`. Standard errors use the adjusted points
    :math:`x = \bar X + \beta`, :math:`u = x - \bar x`:
    :math:`\sigma^2_b = 1/\sum W u^2`,
    :math:`\sigma^2_a = 1/\sum W + \bar x^2 \sigma^2_b`,
    :math:`\operatorname{cov}(a, b) = -\bar x \sigma^2_b`.
    """

    def gsum(arr: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.add.reduceat(arr, starts, axis=1)

    wx, wy = 1.0 / var_x, 1.0 / var_y
    f = np.atleast_2d(freq)
    sizes = np.diff(np.append(starts, x.shape[0]))

    def expand(arr: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.repeat(arr, sizes, axis=1)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if slope_start is None:
            sf = gsum(f)
            dx = x - expand(gsum(f * x) / sf)
            dy = y - expand(gsum(f * y) / sf)
            b = gsum(f * dx * dy) / gsum(f * dx**2)
        else:
            b = np.array(slope_start, dtype=np.float64, copy=True)

        def moments(b_cur: NDArray[np.float64]):
            bb = expand(b_cur)
            w = wx * wy / (wx + bb**2 * wy)
            fw = f * w
            sw = gsum(fw)
            xbar = gsum(fw * x) / sw
            ybar = gsum(fw * y) / sw
            u = x - expand(xbar)
            v = y - expand(ybar)
            beta = w * (u / wy + bb * v / wx)
            return fw, sw, xbar, ybar, u, v, beta

        converged = np.zeros(b.shape, dtype=bool)
        for _ in range(_YORK_MAX_ITER):
            fw, _sw, _xb, _yb, u, v, beta = moments(b)
            b_new = gsum(fw * beta * v) / gsum(fw * beta * u)
            step_ok = np.abs(b_new - b) < _YORK_TOL * np.maximum(1.0, np.abs(b))
            converged = step_ok & np.isfinite(b_new)
            b = b_new
            if np.all(converged | ~np.isfinite(b)):
                break

        fw, sw, xbar, ybar, _u, _v, beta = moments(b)
        a = ybar - b * xbar
        x_adj = expand(xbar) + beta
        x_adj_bar = gsum(fw * x_adj) / sw
        var_b = 1.0 / gsum(fw * (x_adj - expand(x_adj_bar)) ** 2)
        var_a = 1.0 / sw + x_adj_bar**2 * var_b
        cov_ab = -x_adj_bar * var_b
    n_distinct = gsum((f > 0).astype(np.float64))
    return _YorkFit(
        intercept=a, slope=b, var_intercept=var_a, var_slope=var_b,
        cov_intercept_slope=cov_ab, converged=converged, n_distinct=n_distinct,
    )


def _reml_random_effects(
    theta: NDArray[np.float64], s2: NDArray[np.float64], mask: NDArray[np.bool_]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    r"""Row-wise REML random-effects synthesis of estimates `theta` with variances `s2`.

    Parameters
    ----------
    theta, s2 : ndarray of float, shape (n_replicates, n_groups)
        Per-group estimates and their (stage-1) sampling variances.
    mask : ndarray of bool, shape (n_replicates, n_groups)
        Which entries take part in each row's synthesis.

    Returns
    -------
    mu : ndarray of float, shape (n_replicates,)
        Pooled estimate, :math:`\sum w_i\theta_i/\sum w_i` with
        :math:`w_i = 1/(s_i^2 + \tau^2)`.
    tau2 : ndarray of float, shape (n_replicates,)
        REML between-group variance, truncated at 0.
    converged : ndarray of bool, shape (n_replicates,)

    Notes
    -----
    Starts from the DerSimonian-Laird estimate and iterates the REML
    fixed point :math:`\tau^2 \leftarrow \sum w_i^2[(\theta_i - \hat\mu)^2
    - s_i^2] / \sum w_i^2 + 1/\sum w_i` (Viechtbauer 2005), truncating
    at 0, until the change is below ``1e-10 (tau^2 + mean s^2)``.
    """

    m = mask & np.isfinite(theta) & np.isfinite(s2) & (s2 > 0)
    th = np.where(m, theta, 0.0)
    v = np.where(m, s2, 1.0)
    k = m.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(m, 1.0 / v, 0.0)
        sw = w.sum(axis=1)
        mu_fe = (w * th).sum(axis=1) / sw
        q = (w * (th - mu_fe[:, None]) ** 2).sum(axis=1)
        c = sw - (w**2).sum(axis=1) / sw
        tau2 = np.maximum(0.0, (q - (k - 1)) / c)
        tau2 = np.where(np.isfinite(tau2), tau2, 0.0)
        scale = np.where(m, v, 0.0).sum(axis=1) / np.maximum(k, 1)
        converged = np.zeros(theta.shape[0], dtype=bool)
        for _ in range(_REML_MAX_ITER):
            w = np.where(m, 1.0 / (v + tau2[:, None]), 0.0)
            sw = w.sum(axis=1)
            mu = (w * th).sum(axis=1) / sw
            num = (w**2 * ((th - mu[:, None]) ** 2 - v)).sum(axis=1)
            tau2_new = np.maximum(0.0, num / (w**2).sum(axis=1) + 1.0 / sw)
            converged = np.abs(tau2_new - tau2) <= _REML_TOL * (tau2 + scale)
            tau2 = tau2_new
            if np.all(converged | (k < 2)):
                break
        w = np.where(m, 1.0 / (v + tau2[:, None]), 0.0)
        mu = (w * th).sum(axis=1) / w.sum(axis=1)
    bad = k < 2
    mu[bad] = np.nan
    tau2[bad] = np.nan
    converged[bad] = False
    return mu, tau2, converged


@dataclass(frozen=True)
class MixedTest3Result(AnalysisResult):
    """Result of :func:`mixed_equivalence_test_3` (model-level agreement).

    Attributes
    ----------
    meta : AnalysisMeta
        Provenance and inclusion/exclusion record for this computation.
    alpha : float
        Fixed intercept: REML random-effects (precision-weighted) pooled
        mean of the per-metabolite York intercepts, summarising agreement
        of the two fitting methods over the evaluated joint metabolite
        set.
    beta : float
        Fixed slope: REML random-effects (precision-weighted) pooled mean
        of the per-metabolite York slopes, over the evaluated joint
        metabolite set.
    tau_alpha_sq : float
        REML between-metabolite variance of the intercepts,
        ``Var(u_i)``: descriptive heterogeneity across the joint set.
    tau_beta_sq : float
        REML between-metabolite variance of the slopes, ``Var(v_i)``:
        descriptive heterogeneity across the joint set.
    alpha_ci : tuple of (float, float)
        Bootstrap percentile confidence interval for `alpha` (scope per
        `inference`), at `confidence_level_per_parameter`.
    beta_ci : tuple of (float, float)
        Bootstrap percentile confidence interval for `beta` (scope per
        `inference`), at `confidence_level_per_parameter`.
    tau_alpha_sq_ci : tuple of (float, float)
        Bootstrap percentile confidence interval for `tau_alpha_sq`
        (scope per `inference`), at `tau_confidence_level`.
    tau_beta_sq_ci : tuple of (float, float)
        Bootstrap percentile confidence interval for `tau_beta_sq`
        (scope per `inference`), at `tau_confidence_level`.
    reject_joint_null : bool
        True if either `alpha_ci` excludes 0 or `beta_ci` excludes 1
        (i.e. the joint null of perfect model-level agreement is
        rejected).
    confidence_level_per_parameter : float
        Per-parameter confidence level actually used for `alpha_ci` and
        `beta_ci`, ``1 - global_alpha / 2`` (the Bonferroni-adjusted
        level).
    tau_confidence_level : float
        Confidence level of the (descriptive) `tau_alpha_sq_ci` and
        `tau_beta_sq_ci`, ``1 - global_alpha``.
    metabolite_labels : tuple of str
        Metabolites entering stage 2, in the order of the per-metabolite
        arrays below.
    metabolite_intercepts : ndarray of float, shape (n_metabolites,)
        Stage-1 York intercepts ``a_i``.
    metabolite_slopes : ndarray of float, shape (n_metabolites,)
        Stage-1 York slopes ``b_i``.
    metabolite_intercept_se : ndarray of float, shape (n_metabolites,)
        Stage-1 York standard errors of ``a_i``.
    metabolite_slope_se : ndarray of float, shape (n_metabolites,)
        Stage-1 York standard errors of ``b_i``.
    n_metabolites : int
        Number of metabolites entering stage 2.
    n_clusters : int
        Number of distinct clusters (spectra/subjects) included in the
        fit (the bootstrap resampling unit).
    n_bootstrap_valid : int
        Number of bootstrap replicates actually used for the intervals
        (replicates with fewer than 2 usable metabolites are dropped).
    inference : {"panel", "population"}
        Inference scope used for the intervals (see
        :func:`mixed_equivalence_test_3`): ``"panel"`` (cluster bootstrap,
        conditional on the evaluated joint metabolite set) or
        ``"population"`` (two-way pigeonhole bootstrap over metabolites and
        clusters).
    """

    alpha: float
    beta: float
    tau_alpha_sq: float
    tau_beta_sq: float
    alpha_ci: tuple[float, float]
    beta_ci: tuple[float, float]
    tau_alpha_sq_ci: tuple[float, float]
    tau_beta_sq_ci: tuple[float, float]
    reject_joint_null: bool
    confidence_level_per_parameter: float
    tau_confidence_level: float
    metabolite_labels: tuple[str, ...]
    metabolite_intercepts: NDArray[np.float64]
    metabolite_slopes: NDArray[np.float64]
    metabolite_intercept_se: NDArray[np.float64]
    metabolite_slope_se: NDArray[np.float64]
    n_metabolites: int
    n_clusters: int
    n_bootstrap_valid: int
    inference: str


def _york_usable(fit: _YorkFit) -> NDArray[np.bool_]:
    return (
        fit.converged
        & (fit.n_distinct >= _MIN_OBS_PER_METABOLITE)
        & np.isfinite(fit.intercept)
        & np.isfinite(fit.slope)
        & np.isfinite(fit.var_intercept)
        & np.isfinite(fit.var_slope)
        & (fit.var_intercept > 0)
        & (fit.var_slope > 0)
    )


def mixed_equivalence_test_3(
    reference_values: NDArray[np.float64],
    candidate_values: NDArray[np.float64],
    metabolite_ids: NDArray[np.str_],
    cluster_ids: NDArray[np.str_],
    sample_ids: NDArray[np.str_],
    *,
    reference_error_variance: float | NDArray[np.float64],
    test_error_variance: float | NDArray[np.float64],
    global_alpha: float = 0.05,
    n_bootstrap: int = 2000,
    seed: int | None = None,
    inference: str = "panel",
) -> MixedTest3Result:
    r"""Model-level equivalence Test 3: agreement, with metabolite random effects.

    Two-stage approximation of the mixed-effects Deming model
    (manuscript ``me_test3``)

    .. math::

        x_{i,n} = (\alpha + u_i) + (\beta + v_i)\,\eta_{i,n}
            + \epsilon_{i,n}, \qquad y_{i,n} = \eta_{i,n} + \delta_{i,n}

    with :math:`u_i \sim N(0, \tau^2_\alpha)`,
    :math:`v_i \sim N(0, \tau^2_\beta)` and per-observation error
    variances :math:`\sigma^2_{\epsilon,i,n}` (candidate) and
    :math:`\sigma^2_{\delta,i,n}` (reference). Stage 1 fits a York
    errors-in-variables regression of candidate on reference within each
    metabolite; stage 2 pools the per-metabolite intercepts and slopes
    with a REML random-effects model. The joint null ``alpha = 0`` and
    ``beta = 1`` is tested with bootstrap percentile intervals.

    The random effects describe heterogeneity across the evaluated joint
    metabolite set. By default (``inference="panel"``) the fixed effects
    summarise agreement between the two fitting methods over that set,
    the cluster bootstrap captures sampling variability across spectra
    (or subjects), and ``tau^2`` is descriptive. ``inference="population"``
    instead targets a population of metabolites from which the panel is
    regarded as a sample (see Notes).

    Parameters
    ----------
    reference_values : ndarray of float, shape (n_obs,)
        Per-observation reference values ``y_{i,n}``.
    candidate_values : ndarray of float, shape (n_obs,)
        Per-observation candidate values ``x_{i,n}``, aligned with
        `reference_values`.
    metabolite_ids : ndarray of str, shape (n_obs,)
        Metabolite label for each observation; the random-effects
        grouping factor.
    cluster_ids : ndarray of str, shape (n_obs,)
        Cluster label for each observation (spectrum id for *in silico*
        data, subject id for *in vivo* data); the resampling unit for the
        cluster bootstrap.
    sample_ids : ndarray of str, shape (n_obs,)
        Per-observation identifiers, used for inclusion/exclusion
        bookkeeping.
    reference_error_variance : float or ndarray of float, shape (n_obs,)
        Error variance of `reference_values`: a scalar (common to all
        observations) or one value per observation. Observations with a
        non-finite value are excluded; any non-positive finite value is
        invalid input.
    test_error_variance : float or ndarray of float, shape (n_obs,)
        Error variance of `candidate_values`, scalar or per observation,
        with the same rules. Together these give the per-observation
        ``lambda_{i,n} = test_error_variance / reference_error_variance``
        (DL / reference) of ``me_test3``.
    global_alpha : float, default 0.05
        Global two-sided significance level for the joint test. Split
        Bonferroni-style across the two joint-null parameters to a
        per-parameter confidence level of ``1 - global_alpha / 2``. The
        ``tau^2`` intervals are reported at ``1 - global_alpha``.
    n_bootstrap : int, default 2000
        Number of bootstrap resamples.
    seed : int or None, default None
        Seed for the random number generator driving the resampling; None
        draws fresh entropy. For a given seed, ``inference="panel"``
        draws exactly the same cluster resamples as earlier versions.
    inference : {"panel", "population"}, default "panel"
        Inference scope. ``"panel"``: cluster bootstrap over spectra/
        subjects only; inference is conditional on the evaluated joint
        metabolite set (the estimand fixed by the basis set). ``"population"``:
        two-way pigeonhole bootstrap (Owen 2007) over metabolites *and*
        clusters, adding between-metabolite variation (``tau^2``) to the
        uncertainty of ``alpha`` and ``beta``, for generalising beyond the
        panel. The effective number of units is then the number of
        metabolites, so intervals are wide for 20-30 metabolites.
        Recorded in ``meta.configuration["inference"]`` and in
        `MixedTest3Result.inference`.

    Returns
    -------
    MixedTest3Result
        Fixed effects, heterogeneity variances, per-metabolite stage-1
        estimates, bootstrap intervals and the joint-null decision. See
        :class:`MixedTest3Result` for the full field list.

    Raises
    ------
    ValueError
        If a variance array cannot be broadcast to ``(n_obs,)``, or if
        `inference` is not ``"panel"`` or ``"population"``.

    Notes
    -----
    **Stage 1.** For each metabolite, York et al. (2004) regression of
    candidate (York's ``Y``) on reference (York's ``X``) with per-point
    variances and zero error correlation (see :func:`_york_grouped`).
    With constant variances this reproduces
    :func:`~mrsvalidate.analysis.method_comparison.deming` with
    ``lambda = reference_error_variance / test_error_variance``.

    **Stage 2.** Separately for the intercepts ``a_i`` and slopes
    ``b_i``, a univariate random-effects synthesis with the stage-1
    variances: REML ``tau^2`` (truncated at 0) and the inverse-variance
    weighted mean with weights ``1/(s_i^2 + tau^2)``. The covariance of
    ``a_i`` and ``b_i`` is not used (stated simplification, module
    Notes).

    **Inference.** ``n_bootstrap`` cluster-bootstrap replicates resample
    whole clusters with replacement (implemented as multinomial
    frequency weights, exactly equivalent to duplicating the resampled
    clusters) and rerun both stages; the York iteration of a replicate
    is warm-started from the full-data slope. A metabolite with fewer
    than 3 distinct observations in a replicate, or whose York fit fails
    or does not converge, is skipped in that replicate (counted in
    ``meta.configuration``); a replicate with fewer than 2 usable
    metabolites is dropped. Percentile intervals are taken over the
    remaining replicates. In this default (``inference="panel"``)
    metabolites are not resampled: the intervals are panel-conditional,
    i.e. for the precision-weighted summary of the evaluated joint
    metabolite set, and ``tau^2`` (with its interval) is descriptive
    heterogeneity across that set (see module Notes).

    **Population inference.** With ``inference="population"`` each
    replicate independently draws ``n_metabolites`` metabolites with
    replacement and clusters with replacement, forms all
    (metabolite, cluster) cells of the two resampled sets, and reruns
    both stages. A metabolite drawn ``j`` times enters stage 2 as ``j``
    separate metabolites (with identical stage-1 fits, since they share
    the resampled clusters); a replicate needs at least 2 *distinct*
    usable metabolites. Because each replicate's stage-2 spread reflects
    a fresh draw of metabolites, the intervals for ``alpha`` and
    ``beta`` include between-metabolite variation. The number of
    metabolites is then the effective sample size, so intervals are
    substantially wider than in panel mode once ``tau^2`` is comparable
    to the stage-1 variances, and with ~10-30 metabolites the percentile
    interval is a small-sample approximation (slightly liberal when
    ``tau^2`` dominates). The pigeonhole bootstrap counts the
    stage-1 sampling error twice (through both the metabolite and the
    cluster resampling) and is therefore conservative when ``tau^2`` is
    not large relative to the stage-1 variances (Owen 2007; see the
    validation in ``tests/unit/test_mixed_effects.py``).

    Metabolites with fewer than 3 valid observations are excluded
    (``meta.exclusion_reasons``). Fewer than 2 remaining metabolites or
    fewer than 2 clusters yields ``ResultStatus.INSUFFICIENT_DATA``;
    non-positive variances yield ``ResultStatus.INVALID_INPUT``; fewer
    than 2 metabolites with a defined York fit yields
    ``ResultStatus.UNDEFINED``; no usable bootstrap replicate yields
    ``ResultStatus.FAILED``.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.mixed_effects import mixed_equivalence_test_3
    >>> rng = np.random.default_rng(0)
    >>> clusters, metabolites, samples, y, x = [], [], [], [], []
    >>> for c in range(20):
    ...     for m in range(4):
    ...         eta = 5.0 + m * 2.0 + rng.normal(0, 1.0)
    ...         y.append(eta + rng.normal(0, 0.2))
    ...         x.append(eta + rng.normal(0, 0.3))
    ...         clusters.append(f"cluster{c}")
    ...         metabolites.append(f"metab{m}")
    ...         samples.append(f"sample{c}_{m}")
    >>> result = mixed_equivalence_test_3(
    ...     np.array(y), np.array(x), np.array(metabolites), np.array(clusters),
    ...     np.array(samples), reference_error_variance=0.04, test_error_variance=0.09,
    ...     n_bootstrap=200, seed=0,
    ... )
    >>> result.meta.status.value
    'valid'
    >>> print(f"{result.alpha:.4f} {result.beta:.4f} {result.tau_beta_sq:.4f}")
    0.0323 0.9949 0.0000
    >>> result.reject_joint_null, result.n_metabolites, result.n_clusters
    (False, 4, 20)
    >>> result.inference
    'panel'
    >>> pop = mixed_equivalence_test_3(
    ...     np.array(y), np.array(x), np.array(metabolites), np.array(clusters),
    ...     np.array(samples), reference_error_variance=0.04, test_error_variance=0.09,
    ...     n_bootstrap=200, seed=0, inference="population",
    ... )
    >>> pop.meta.configuration["ci_method"]
    'two_way_pigeonhole_bootstrap_percentile_two_stage'
    >>> print(f"{pop.beta_ci[0]:.4f} {pop.beta_ci[1]:.4f}")
    0.9092 1.0886
    """

    method_name = "mixed_equivalence_test_3"
    if inference not in _INFERENCE_CHOICES:
        raise ValueError(f"inference must be one of {_INFERENCE_CHOICES}, got {inference!r}")
    population = inference == "population"
    all_ids_arr = np.asarray(sample_ids, dtype=str)
    all_ids = tuple(all_ids_arr.tolist())
    ref = np.asarray(reference_values, dtype=np.float64)
    cand = np.asarray(candidate_values, dtype=np.float64)
    metab = np.asarray(metabolite_ids, dtype=str)
    clusters = np.asarray(cluster_ids, dtype=str)
    n_obs = ref.shape[0]
    scalar_ref_var = np.ndim(reference_error_variance) == 0
    scalar_test_var = np.ndim(test_error_variance) == 0
    try:
        var_ref = np.broadcast_to(np.asarray(reference_error_variance, dtype=np.float64), (n_obs,))
        var_test = np.broadcast_to(np.asarray(test_error_variance, dtype=np.float64), (n_obs,))
    except ValueError as exc:
        raise ValueError(
            "reference_error_variance and test_error_variance must be scalars or arrays of shape (n_obs,)"
        ) from exc

    per_param_level = 1.0 - global_alpha / 2.0
    tau_level = 1.0 - global_alpha
    nan_pair = (float("nan"), float("nan"))
    base_config: dict[str, Any] = {
        "model": "two-stage: per-metabolite York (2004) EIV regression of candidate on reference, "
        "then univariate REML random-effects synthesis of intercepts and slopes",
        "york_error_correlation": 0.0,
        "york_tolerance": _YORK_TOL,
        "york_max_iter": _YORK_MAX_ITER,
        "york_standard_errors": "York expected SEs (based on supplied error variances)",
        "stage2_estimator": "REML (DerSimonian-Laird start, fixed-point iteration), tau^2 truncated at 0",
        "stage2_ab_covariance_used": False,
        "reml_tolerance": _REML_TOL,
        "reml_max_iter": _REML_MAX_ITER,
        "variance_input": {
            "reference_error_variance": "scalar" if scalar_ref_var else "per_observation",
            "test_error_variance": "scalar" if scalar_test_var else "per_observation",
        },
        "inference": inference,
        "inference_description": _INFERENCE_DESCRIPTIONS[inference],
        "bootstrap_resampling_units": ("metabolites x clusters" if population else "clusters"),
        "ci_method": (
            "two_way_pigeonhole_bootstrap_percentile_two_stage" if population
            else "cluster_bootstrap_percentile_two_stage"
        ),
        "n_bootstrap": n_bootstrap,
        "bonferroni_global_alpha": global_alpha,
        "confidence_level_per_parameter": per_param_level,
        "tau_confidence_level": tau_level,
        "min_observations_per_metabolite": _MIN_OBS_PER_METABOLITE,
    }

    def _nan_result(meta: Any, labels: tuple[str, ...] = (), n_metab: int = 0, n_cl: int = 0) -> MixedTest3Result:
        empty = np.full(len(labels), np.nan)
        return MixedTest3Result(
            meta=meta, alpha=float("nan"), beta=float("nan"), tau_alpha_sq=float("nan"),
            tau_beta_sq=float("nan"), alpha_ci=nan_pair, beta_ci=nan_pair, tau_alpha_sq_ci=nan_pair,
            tau_beta_sq_ci=nan_pair, reject_joint_null=False, confidence_level_per_parameter=per_param_level,
            tau_confidence_level=tau_level, metabolite_labels=labels, metabolite_intercepts=empty,
            metabolite_slopes=empty.copy(), metabolite_intercept_se=empty.copy(),
            metabolite_slope_se=empty.copy(), n_metabolites=n_metab, n_clusters=n_cl, n_bootstrap_valid=0,
            inference=inference,
        )

    finite = np.isfinite(ref) & np.isfinite(cand) & np.isfinite(var_ref) & np.isfinite(var_test)
    n_nonfinite = int(np.sum(~finite))
    if np.any(finite & ((var_ref <= 0) | (var_test <= 0))):
        return _nan_result(build_meta(
            status=ResultStatus.INVALID_INPUT, all_ids=all_ids, included_ids=(), method=method_name,
            warnings=("reference_error_variance and test_error_variance must be strictly positive",),
            configuration=base_config, random_seed=seed,
        ))

    labels_all, counts_all = np.unique(metab[finite], return_counts=True)
    small_metabs = set(labels_all[counts_all < _MIN_OBS_PER_METABOLITE].tolist())
    keep = finite & ~np.isin(metab, list(small_metabs)) if small_metabs else finite.copy()
    n_small = int(np.sum(finite & ~keep))
    exclusion: dict[str, int] = {}
    if n_nonfinite:
        exclusion["nan_or_inf"] = n_nonfinite
    if n_small:
        exclusion["metabolite_with_fewer_than_3_observations"] = n_small

    warnings: list[str] = []
    if small_metabs:
        warnings.append(
            f"excluded {len(small_metabs)} metabolite(s) with fewer than {_MIN_OBS_PER_METABOLITE} "
            f"valid observations: {sorted(small_metabs)}"
        )

    # Point estimate (stage 1), dropping metabolites whose York fit is undefined.
    while True:
        labels = np.unique(metab[keep])
        n_cl = int(np.unique(clusters[keep]).shape[0]) if np.any(keep) else 0
        if labels.shape[0] < 2 or n_cl < 2:
            return _nan_result(build_meta(
                status=ResultStatus.INSUFFICIENT_DATA, all_ids=all_ids,
                included_ids=tuple(all_ids_arr[keep].tolist()), exclusion_reasons=exclusion,
                method=method_name, warnings=tuple(warnings), configuration=base_config, random_seed=seed,
            ), n_metab=int(labels.shape[0]), n_cl=n_cl)
        idx = np.flatnonzero(keep)
        _, mcode = np.unique(metab[idx], return_inverse=True)
        order = np.argsort(mcode, kind="stable")
        idx = idx[order]
        mcode = mcode[order]
        starts = np.flatnonzero(np.r_[True, np.diff(mcode) != 0])
        xs, ys, vxs, vys = ref[idx], cand[idx], var_ref[idx], var_test[idx]
        point = _york_grouped(xs, ys, vxs, vys, starts, np.ones((1, idx.shape[0])))
        usable = _york_usable(point)[0]
        if np.all(usable):
            break
        bad = labels[~usable].tolist()
        warnings.append(f"York fit undefined or not converged for metabolite(s) {bad}; excluded")
        n_bad = int(np.sum(np.isin(metab[keep], bad)))
        exclusion["york_fit_undefined"] = exclusion.get("york_fit_undefined", 0) + n_bad
        keep = keep & ~np.isin(metab, bad)
        if np.sum(usable) < 2:
            return _nan_result(build_meta(
                status=ResultStatus.UNDEFINED, all_ids=all_ids, included_ids=tuple(all_ids_arr[keep].tolist()),
                exclusion_reasons=exclusion, method=method_name, warnings=tuple(warnings),
                configuration=base_config, random_seed=seed,
            ), n_metab=int(np.sum(usable)), n_cl=n_cl)

    included_ids = tuple(all_ids_arr[idx].tolist())
    n_metab = int(labels.shape[0])
    ones_mask = np.ones((1, n_metab), dtype=bool)
    alpha_hat, tau_a_hat, conv_a = _reml_random_effects(point.intercept, point.var_intercept, ones_mask)
    beta_hat, tau_b_hat, conv_b = _reml_random_effects(point.slope, point.var_slope, ones_mask)
    if not (conv_a[0] and conv_b[0]):
        warnings.append("stage-2 REML iteration did not converge for the point estimate")

    # Bootstrap via multinomial frequency weights over clusters; in population
    # mode each replicate additionally resamples metabolites with replacement
    # (two-way pigeonhole bootstrap). A cell (metabolite i, cluster c) of the
    # resampled sets carries the data of metabolite i in cluster c, so a
    # metabolite drawn j times contributes j identical stage-1 fits, each
    # entering stage 2 as a separate metabolite: the stage-1 fits are computed
    # once per original metabolite and then indexed by the metabolite draws.
    _, ccode = np.unique(clusters[idx], return_inverse=True)
    n_obs_used = idx.shape[0]
    rng = np.random.default_rng(seed)
    chunk = int(max(1, min(max(n_bootstrap, 1), 2_000_000 // max(n_obs_used, 1))))
    boot = np.full((n_bootstrap, 4), np.nan)
    n_skip_small = n_skip_york = n_reml_nonconv = 0
    done = 0
    while done < n_bootstrap:
        r = min(chunk, n_bootstrap - done)
        draws = rng.integers(0, n_cl, size=(r, n_cl))
        counts = np.bincount((draws + n_cl * np.arange(r)[:, None]).ravel(), minlength=r * n_cl).reshape(r, n_cl)
        freq = counts[:, ccode].astype(np.float64)
        fit = _york_grouped(xs, ys, vxs, vys, starts, freq, slope_start=np.repeat(point.slope, r, axis=0))
        enough = fit.n_distinct >= _MIN_OBS_PER_METABOLITE
        use = _york_usable(fit)
        n_skip_small += int(np.sum(~enough))
        n_skip_york += int(np.sum(enough & ~use))
        if population:
            mdraw = rng.integers(0, n_metab, size=(r, n_metab))
            rows = np.arange(r)[:, None]
            use_s = use[rows, mdraw]
            a_b, ta_b, ca = _reml_random_effects(fit.intercept[rows, mdraw], fit.var_intercept[rows, mdraw], use_s)
            b_b, tb_b, cb = _reml_random_effects(fit.slope[rows, mdraw], fit.var_slope[rows, mdraw], use_s)
            # At least 2 *distinct* usable metabolites (a single metabolite drawn
            # twice would give a degenerate tau^2 = 0 synthesis).
            drawn = np.zeros((r, n_metab), dtype=bool)
            drawn[rows, mdraw] = True
            valid_rows = (drawn & use).sum(axis=1) >= 2
        else:
            a_b, ta_b, ca = _reml_random_effects(fit.intercept, fit.var_intercept, use)
            b_b, tb_b, cb = _reml_random_effects(fit.slope, fit.var_slope, use)
            valid_rows = use.sum(axis=1) >= 2
        n_reml_nonconv += int(np.sum(valid_rows & ~(ca & cb)))
        boot[done:done + r] = np.column_stack([a_b, b_b, ta_b, tb_b])
        boot[done:done + r][~valid_rows] = np.nan
        done += r

    ok = np.all(np.isfinite(boot), axis=1)
    n_valid = int(np.sum(ok))
    n_dropped = n_bootstrap - n_valid
    if n_skip_small or n_skip_york:
        warnings.append(
            f"bootstrap: skipped {n_skip_small} (replicate, metabolite) fit(s) with < "
            f"{_MIN_OBS_PER_METABOLITE} distinct observations and {n_skip_york} with a failed/"
            "non-converged York fit"
        )
    if n_dropped:
        warnings.append(
            f"bootstrap: dropped {n_dropped} replicate(s) with fewer than 2 usable "
            f"{'distinct ' if population else ''}metabolites"
        )
    if n_reml_nonconv:
        warnings.append(f"bootstrap: stage-2 REML did not converge in {n_reml_nonconv} replicate(s)")

    config = dict(base_config)
    config.update({
        "n_metabolites": n_metab,
        "n_clusters": n_cl,
        "n_bootstrap_valid": n_valid,
        "n_bootstrap_dropped": n_dropped,
        "n_bootstrap_metabolite_skips_too_few_observations": n_skip_small,
        "n_bootstrap_metabolite_skips_york_failure": n_skip_york,
        "n_bootstrap_reml_nonconverged": n_reml_nonconv,
        "bootstrap_york_start": "full-data York slope (warm start)",
    })

    labels_t = tuple(labels.tolist())
    common = {
        "alpha": float(alpha_hat[0]), "beta": float(beta_hat[0]), "tau_alpha_sq": float(tau_a_hat[0]),
        "tau_beta_sq": float(tau_b_hat[0]), "confidence_level_per_parameter": per_param_level,
        "tau_confidence_level": tau_level, "metabolite_labels": labels_t,
        "metabolite_intercepts": point.intercept[0].copy(), "metabolite_slopes": point.slope[0].copy(),
        "metabolite_intercept_se": np.sqrt(point.var_intercept[0]), "metabolite_slope_se": np.sqrt(point.var_slope[0]),
        "n_metabolites": n_metab, "n_clusters": n_cl, "n_bootstrap_valid": n_valid, "inference": inference,
    }

    if n_valid == 0:
        meta = build_meta(
            status=ResultStatus.FAILED, all_ids=all_ids, included_ids=included_ids,
            exclusion_reasons=exclusion, method=method_name,
            warnings=(*warnings, "no usable bootstrap replicate"), configuration=config, random_seed=seed,
        )
        return MixedTest3Result(
            meta=meta, alpha_ci=nan_pair, beta_ci=nan_pair, tau_alpha_sq_ci=nan_pair,
            tau_beta_sq_ci=nan_pair, reject_joint_null=False, **common,
        )

    good = boot[ok]
    tail = (1.0 - per_param_level) / 2.0
    tau_tail = (1.0 - tau_level) / 2.0

    def _ci(col: int, t: float) -> tuple[float, float]:
        lo, hi = np.quantile(good[:, col], [t, 1.0 - t])
        return float(lo), float(hi)

    alpha_ci = _ci(0, tail)
    beta_ci = _ci(1, tail)
    tau_alpha_sq_ci = _ci(2, tau_tail)
    tau_beta_sq_ci = _ci(3, tau_tail)
    reject = not (alpha_ci[0] <= 0.0 <= alpha_ci[1] and beta_ci[0] <= 1.0 <= beta_ci[1])

    meta = build_meta(
        status=ResultStatus.VALID, all_ids=all_ids, included_ids=included_ids, exclusion_reasons=exclusion,
        method=method_name, warnings=tuple(warnings), configuration=config, random_seed=seed,
    )
    return MixedTest3Result(
        meta=meta, alpha_ci=alpha_ci, beta_ci=beta_ci, tau_alpha_sq_ci=tau_alpha_sq_ci,
        tau_beta_sq_ci=tau_beta_sq_ci, reject_joint_null=reject, **common,
    )
