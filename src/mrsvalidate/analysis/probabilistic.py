r"""Negative log predictive density (NLPD) metrics.

Implements the marginal (per-metabolite/output), macro-averaged, and
full-covariance joint forms of the negative log predictive density used
to score probabilistic (mean + uncertainty) predictions against ground
truth (spec eq. 43 / manuscript ``NLPD``).

Notes
-----
For a univariate Gaussian predictive distribution the marginal NLPD is

.. math::

    \text{NLPD} = \frac{1}{N}\sum_{n=1}^{N} \frac{1}{2}\left[
        \log \sigma_{\text{pred},n}^2
        + \frac{(y_{\text{true},n} - y_{\text{pred},n})^2}{\sigma_{\text{pred},n}^2}
        + \log(2\pi)
    \right]

Predictive variance must be finite and strictly positive: a nonpositive
or non-finite variance is an invalid-input exclusion and is never
silently floored unless the caller explicitly opts into a variance
floor. A full-covariance joint Gaussian NLPD
(:func:`joint_nlpd_full_covariance`) always uses Cholesky
factorization/solve and a log-determinant derived from the Cholesky
diagonal, never an explicit matrix inverse or ``log(det(...))`` of the
raw matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import LinAlgError, cho_factor, cho_solve

from ..results import AnalysisResult, PerSampleResult, ResultStatus, ScalarMetricResult, build_meta

__all__ = ["nlpd", "nlpd_macro", "joint_nlpd_full_covariance"]

_LOG_2PI = float(np.log(2.0 * np.pi))


def nlpd(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    sigma_pred_sq: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
    *,
    variance_floor: float | None = None,
) -> ScalarMetricResult:
    r"""Compute the marginal (per-metabolite/output) NLPD.

    Averages the univariate Gaussian negative log predictive density
    over all observations for a single metabolite/output.

    Parameters
    ----------
    y_true : ndarray of float
        Ground-truth values, one per observation.
    y_pred : ndarray of float
        Predicted (posterior mean) values, aligned with `y_true`.
    sigma_pred_sq : ndarray of float
        Predictive variance for each observation. Must be finite and
        strictly positive for an observation to be included; see Notes.
    sample_ids : ndarray of str
        Identifier for each observation, used to report which samples
        were included in or excluded from the computation.
    variance_floor : float, optional
        If given, any finite variance below this value is replaced by
        it before validity filtering. Disabled by default (strict
        reproduction mode).

    Returns
    -------
    ScalarMetricResult
        ``value`` is the mean NLPD over included observations.
        ``meta.status`` is
        :attr:`~mrsvalidate.results.ResultStatus.INVALID_INPUT` (and
        ``value`` is ``nan``) if no observation has finite `y_true`,
        `y_pred`, and a strictly positive, finite variance.

    Notes
    -----
    .. math::

        \text{NLPD} = \frac{1}{N}\sum_{n=1}^{N} \frac{1}{2}\left[
            \log \sigma_{\text{pred},n}^2
            + \frac{(y_{\text{true},n} - y_{\text{pred},n})^2}{\sigma_{\text{pred},n}^2}
            + \log(2\pi)
        \right]

    ``variance_floor`` is a separately configured numerical policy,
    disabled by default (strict reproduction mode); when supplied, the
    number of rows it actually affected is recorded in
    ``result.meta.configuration["n_variance_floor_applied"]``, so
    silent flooring never goes unreported.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([0.0, 1.0, 2.0])
    >>> y_pred = np.array([0.0, 1.0, 2.0])
    >>> sigma_pred_sq = np.array([1.0, 1.0, 1.0])
    >>> ids = np.array(["a", "b", "c"])
    >>> result = nlpd(y_true, y_pred, sigma_pred_sq, ids)
    >>> round(result.value, 4)
    0.9189
    """

    all_ids = tuple(np.asarray(sample_ids, dtype=str).tolist())
    y = np.asarray(y_true, dtype=np.float64)
    yhat = np.asarray(y_pred, dtype=np.float64)
    var = np.asarray(sigma_pred_sq, dtype=np.float64)

    n_floored = 0
    if variance_floor is not None:
        floor_mask = np.isfinite(var) & (var < variance_floor)
        n_floored = int(np.sum(floor_mask))
        var = np.where(floor_mask, variance_floor, var)

    valid = np.isfinite(y) & np.isfinite(yhat) & np.isfinite(var) & (var > 0)
    n_invalid = int(np.sum(~valid))
    included_ids = tuple(np.asarray(all_ids)[valid].tolist())
    status = ResultStatus.VALID if included_ids else ResultStatus.INVALID_INPUT

    config = {"variance_floor": variance_floor, "n_variance_floor_applied": n_floored}
    meta = build_meta(
        status=status,
        all_ids=all_ids,
        included_ids=included_ids,
        exclusion_reasons={"nonpositive_or_nonfinite_variance": n_invalid} if n_invalid else {},
        method="nlpd",
        configuration=config,
    )
    if status is not ResultStatus.VALID:
        return ScalarMetricResult(meta=meta, value=float("nan"))

    yv, yhatv, varv = y[valid], yhat[valid], var[valid]
    term = 0.5 * (np.log(varv) + (yv - yhatv) ** 2 / varv + _LOG_2PI)
    return ScalarMetricResult(meta=meta, value=float(np.mean(term)))


def nlpd_macro(per_output_results: Sequence[ScalarMetricResult]) -> ScalarMetricResult:
    """Compute the equal-weight macro-averaged NLPD across metabolites/outputs.

    Parameters
    ----------
    per_output_results : sequence of ScalarMetricResult
        One :class:`~mrsvalidate.results.ScalarMetricResult` per
        metabolite/output, as produced by :func:`nlpd`.

    Returns
    -------
    ScalarMetricResult
        ``value`` is the unweighted mean of ``value`` across every
        input result whose ``meta.status`` is
        :attr:`~mrsvalidate.results.ResultStatus.VALID`. ``value`` is
        ``nan`` and ``meta.status`` is
        :attr:`~mrsvalidate.results.ResultStatus.INSUFFICIENT_DATA` if
        none are valid.

    Notes
    -----
    Averaging is unweighted by design (spec): this prevents metabolites
    with more valid observations from dominating the aggregate unless a
    weighted aggregate is explicitly requested.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.results import ScalarMetricResult, ResultStatus, build_meta
    >>> def _scalar(value):
    ...     meta = build_meta(status=ResultStatus.VALID, all_ids=("s",), included_ids=("s",))
    ...     return ScalarMetricResult(meta=meta, value=value)
    >>> results = [_scalar(1.0), _scalar(3.0)]
    >>> nlpd_macro(results).value
    2.0
    """

    valid_values = [r.value for r in per_output_results if r.meta.status is ResultStatus.VALID]
    all_ids = tuple(f"output_{i}" for i in range(len(per_output_results)))
    included_ids = tuple(
        f"output_{i}" for i, r in enumerate(per_output_results) if r.meta.status is ResultStatus.VALID
    )
    status = ResultStatus.VALID if valid_values else ResultStatus.INSUFFICIENT_DATA
    meta = build_meta(
        status=status,
        all_ids=all_ids,
        included_ids=included_ids,
        method="nlpd_macro",
        configuration={"aggregation": "equal_weight_mean"},
    )
    value = float(np.mean(valid_values)) if valid_values else float("nan")
    return ScalarMetricResult(meta=meta, value=value)


def joint_nlpd_full_covariance(
    errors: NDArray[np.float64],
    covariance: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
) -> PerSampleResult:
    r"""Compute the per-sample joint multivariate Gaussian NLPD.

    Parameters
    ----------
    errors : ndarray of float, shape (n_samples,) or (n_samples, D)
        Per-sample error vectors ``e = y_true - y_pred``. A 1-D array
        is treated as ``D = 1``.
    covariance : ndarray of float, shape (D, D) or (n_samples, D, D)
        Either one covariance matrix shared by every sample, or one
        matrix per sample.
    sample_ids : ndarray of str
        Identifier for each sample.

    Returns
    -------
    PerSampleResult
        ``values`` holds the per-sample NLPD, aligned with
        ``meta.included_ids``, for every sample whose error vector and
        covariance are finite and whose covariance is positive
        definite. Samples with a singular or otherwise invalid
        covariance are excluded and counted under the
        ``singular_or_invalid_covariance`` exclusion reason.

    Notes
    -----
    .. math::

        \text{NLPD}_n = \frac{1}{2}\left[
            \log\det(\Sigma) + e_n^{\mathsf{T}} \Sigma^{-1} e_n + D\log(2\pi)
        \right]

    computed via a Cholesky factorization and solve — never an explicit
    matrix inverse or a raw ``det(...)`` — with the log-determinant
    obtained as twice the sum of the log of the Cholesky factor's
    diagonal, which is both faster and numerically more stable than
    forming the determinant directly.

    Examples
    --------
    >>> import numpy as np
    >>> errors = np.array([[0.0, 0.0], [1.0, 0.0]])
    >>> covariance = np.eye(2)
    >>> ids = np.array(["a", "b"])
    >>> result = joint_nlpd_full_covariance(errors, covariance, ids)
    >>> [round(float(v), 4) for v in result.values]
    [1.8379, 2.3379]
    """

    all_ids = tuple(np.asarray(sample_ids, dtype=str).tolist())
    e = np.asarray(errors, dtype=np.float64)
    if e.ndim == 1:
        e = e[:, np.newaxis]
    n, d = e.shape
    cov = np.asarray(covariance, dtype=np.float64)
    shared = cov.ndim == 2

    values = np.full(n, np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)

    shared_factor = None
    shared_logdet = None
    if shared:
        try:
            shared_factor = cho_factor(cov)
            shared_logdet = 2.0 * float(np.sum(np.log(np.diag(shared_factor[0]))))
        except LinAlgError:
            shared_factor = None

    for i in range(n):
        row = e[i]
        c_i = cov if shared else cov[i]
        if not np.all(np.isfinite(row)) or not np.all(np.isfinite(c_i)):
            continue
        try:
            if shared:
                if shared_factor is None:
                    continue
                factor, logdet = shared_factor, shared_logdet
            else:
                factor = cho_factor(c_i)
                logdet = 2.0 * float(np.sum(np.log(np.diag(factor[0]))))
            x = cho_solve(factor, row)
            quad = float(row @ x)
            values[i] = 0.5 * (logdet + quad + d * _LOG_2PI)
            valid[i] = True
        except LinAlgError:
            continue

    included_ids = tuple(np.asarray(all_ids)[valid].tolist())
    n_invalid = n - len(included_ids)
    status = ResultStatus.VALID if included_ids else ResultStatus.INVALID_INPUT
    meta = build_meta(
        status=status,
        all_ids=all_ids,
        included_ids=included_ids,
        exclusion_reasons={"singular_or_invalid_covariance": n_invalid} if n_invalid else {},
        method="joint_nlpd_full_covariance",
        configuration={"inversion_method": "cholesky_solve", "dimension": d},
    )
    return PerSampleResult(meta=meta, values=values[valid])
