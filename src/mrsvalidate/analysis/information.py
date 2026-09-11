r"""External-CRLB/information-standardized analysis.

Compares predictions and their uncertainty against an externally
supplied Cramer-Rao lower bound (CRLB) reference (spec eqs. 47-49, 53,
65-66 / manuscript :math:`\sigma_{\text{CRLB}}`, :math:`R_m`,
:math:`D^2`).

Notes
-----
CRLB values are **external data only**: this module never computes a
Cramer-Rao lower bound itself, and it refuses any uncertainty array
whose representation (variance / std / percent_std / covariance) is
not explicitly stated via :class:`UncertaintyRepresentation`.
Covariance inversion always goes through Cholesky factorization +
solve, never an explicit matrix inverse; a singular covariance is
invalid unless the caller explicitly opts into a pseudoinverse
extension (in which case the effective rank/tolerance used are
reported in ``meta.configuration``).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import LinAlgError, cho_factor, cho_solve

from ..exceptions import InvalidUncertaintyRepresentationError
from ..results import AnalysisResult, PerSampleResult, ResultStatus, build_meta

__all__ = [
    "UncertaintyRepresentation",
    "to_std",
    "crlb_sd_from_covariance",
    "standardized_error",
    "VarianceRatioResult",
    "variance_ratio",
    "mahalanobis_squared_error",
]

#: sqrt(2/pi): the expected absolute standardized error under a correctly
#: specified zero-mean Gaussian error whose SD equals the normalizing CRLB
#: SD — NOT 1. Never label epsilon == 1 as the theoretical optimum.
EXPECTED_ABS_STANDARDIZED_ERROR_UNDER_CRLB = float(np.sqrt(2.0 / np.pi))


class UncertaintyRepresentation(str, Enum):
    """The explicit numerical representation of an externally supplied uncertainty array.

    There is no default representation anywhere in this module: callers
    must always state which of these four forms their uncertainty
    values are in.

    Attributes
    ----------
    VARIANCE
        Values are variances (:math:`\\sigma^2`).
    STD
        Values are standard deviations (:math:`\\sigma`).
    PERCENT_STD
        Values are standard deviations expressed as a percentage of a
        separately supplied reference quantity.
    COVARIANCE
        Values form a full covariance matrix (not accepted by
        :func:`to_std`; use :func:`crlb_sd_from_covariance` instead).
    """

    VARIANCE = "variance"
    STD = "std"
    PERCENT_STD = "percent_std"
    COVARIANCE = "covariance"


def to_std(
    values: NDArray[np.float64],
    representation: UncertaintyRepresentation,
    sample_ids: NDArray[np.str_],
    *,
    reference: NDArray[np.float64] | None = None,
) -> PerSampleResult:
    """Convert an externally supplied uncertainty array to standard deviations.

    Parameters
    ----------
    values : ndarray of float
        The uncertainty values to convert, in the form declared by
        `representation`.
    representation : UncertaintyRepresentation
        The representation `values` is stored in. Mandatory — there is
        no default. Must not be
        :attr:`UncertaintyRepresentation.COVARIANCE`; use
        :func:`crlb_sd_from_covariance` for a full covariance matrix.
    sample_ids : ndarray of str
        Identifier for each observation.
    reference : ndarray of float, optional
        The quantity the percentage is relative to, required only when
        `representation` is
        :attr:`UncertaintyRepresentation.PERCENT_STD`.

    Returns
    -------
    PerSampleResult
        ``values`` holds the standard deviation for every observation
        with a finite, nonnegative result, aligned with
        ``meta.included_ids``.

    Raises
    ------
    InvalidUncertaintyRepresentationError
        If `representation` is not an :class:`UncertaintyRepresentation`
        member, is :attr:`~UncertaintyRepresentation.COVARIANCE`, or is
        :attr:`~UncertaintyRepresentation.PERCENT_STD` without a
        `reference` array.

    Examples
    --------
    >>> import numpy as np
    >>> values = np.array([4.0, 9.0, -1.0])
    >>> ids = np.array(["a", "b", "c"])
    >>> result = to_std(values, UncertaintyRepresentation.VARIANCE, ids)
    >>> result.values
    array([2., 3.])
    >>> result.meta.included_ids
    ('a', 'b')
    """

    if not isinstance(representation, UncertaintyRepresentation):
        raise InvalidUncertaintyRepresentationError(
            f"representation must be an UncertaintyRepresentation, got {representation!r}"
        )
    if representation is UncertaintyRepresentation.COVARIANCE:
        raise InvalidUncertaintyRepresentationError(
            "to_std() does not accept COVARIANCE; use crlb_sd_from_covariance()"
        )

    all_ids = tuple(np.asarray(sample_ids, dtype=str).tolist())
    values_arr = np.asarray(values, dtype=np.float64)

    if representation is UncertaintyRepresentation.VARIANCE:
        std = np.sqrt(np.where(values_arr >= 0, values_arr, np.nan))
    elif representation is UncertaintyRepresentation.STD:
        std = values_arr.copy()
    else:  # PERCENT_STD
        if reference is None:
            raise InvalidUncertaintyRepresentationError(
                "PERCENT_STD requires an explicit `reference` array"
            )
        reference_arr = np.asarray(reference, dtype=np.float64)
        std = np.where(reference_arr >= 0, values_arr / 100.0 * reference_arr, np.nan)

    valid = np.isfinite(std) & (std >= 0)
    included_ids = tuple(np.asarray(all_ids)[valid].tolist())
    n_invalid = int(np.sum(~valid))
    status = ResultStatus.VALID if included_ids else ResultStatus.INSUFFICIENT_DATA
    meta = build_meta(
        status=status,
        all_ids=all_ids,
        included_ids=included_ids,
        exclusion_reasons={"invalid_uncertainty": n_invalid} if n_invalid else {},
        method="to_std",
        configuration={"representation": representation.value},
    )
    return PerSampleResult(meta=meta, values=std[valid])


def crlb_sd_from_covariance(
    covariance: NDArray[np.float64], *, symmetry_atol: float = 1e-6
) -> NDArray[np.float64]:
    r"""Extract per-parameter CRLB standard deviations from a CRLB covariance matrix.

    Parameters
    ----------
    covariance : ndarray of float, shape (M, M)
        The externally supplied CRLB covariance matrix
        :math:`C_{\text{CRLB}}`.
    symmetry_atol : float, default 1e-6
        Absolute tolerance used when checking that `covariance` is
        symmetric.

    Returns
    -------
    ndarray of float, shape (M,)
        :math:`\sigma_{\text{CRLB},m} = \sqrt{\operatorname{diag}(C_{\text{CRLB}})}`.

    Raises
    ------
    ValueError
        If `covariance` is not a square 2-D matrix, contains non-finite
        values, is not symmetric within `symmetry_atol`, or has a
        negative diagonal element.

    Notes
    -----
    .. math::

        \sigma_{\text{CRLB},m} = \sqrt{\operatorname{diag}(C_{\text{CRLB}})}

    after validating finiteness, symmetry (within `symmetry_atol`), and
    a nonnegative diagonal.

    Examples
    --------
    >>> import numpy as np
    >>> covariance = np.array([[4.0, 0.0], [0.0, 9.0]])
    >>> crlb_sd_from_covariance(covariance)
    array([2., 3.])
    """

    cov = np.asarray(covariance, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"covariance must be a square 2D matrix, got shape {cov.shape}")
    if not np.all(np.isfinite(cov)):
        raise ValueError("covariance contains non-finite values")
    if not np.allclose(cov, cov.T, atol=symmetry_atol):
        raise ValueError("covariance is not symmetric within the documented tolerance")
    diag = np.diag(cov)
    if np.any(diag < 0):
        raise ValueError("covariance has a negative diagonal element")
    return np.sqrt(diag)


def standardized_error(
    reference: NDArray[np.float64],
    prediction: NDArray[np.float64],
    sigma_crlb: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
) -> PerSampleResult:
    r"""Compute the per-observation information-standardized disagreement.

    Parameters
    ----------
    reference : ndarray of float
        Ground-truth (or otherwise reference) value for each
        observation.
    prediction : ndarray of float
        Predicted value for each observation, aligned with `reference`.
    sigma_crlb : ndarray of float
        The externally supplied CRLB standard deviation for each
        observation. Must be finite and strictly positive.
    sample_ids : ndarray of str
        Identifier for each observation.

    Returns
    -------
    PerSampleResult
        ``values`` holds :math:`\epsilon_n` for every observation with
        finite `reference`, `prediction`, and a finite, strictly
        positive `sigma_crlb`, aligned with ``meta.included_ids``.
        ``meta.configuration["expected_value_under_correct_gaussian_crlb"]``
        always records
        :data:`EXPECTED_ABS_STANDARDIZED_ERROR_UNDER_CRLB`; see Notes.

    Notes
    -----
    .. math::

        \epsilon_n = \frac{|\hat{y}_n - y_n|}{\sigma_{\text{CRLB},n}}

    Under a correctly specified zero-mean Gaussian error whose SD equals
    :math:`\sigma_{\text{CRLB}}`,
    :math:`\mathbb{E}[\epsilon] = \sqrt{2/\pi} \approx 0.7979`, **not 1**
    — this expected value is recorded in the result configuration so
    that ``epsilon == 1`` is never read as a universal theoretical
    optimum.

    Examples
    --------
    >>> import numpy as np
    >>> reference = np.array([1.0, 2.0])
    >>> prediction = np.array([1.5, 2.4])
    >>> sigma_crlb = np.array([0.5, 0.5])
    >>> ids = np.array(["a", "b"])
    >>> result = standardized_error(reference, prediction, sigma_crlb, ids)
    >>> result.values
    array([1. , 0.8])
    >>> round(result.meta.configuration["expected_value_under_correct_gaussian_crlb"], 4)
    0.7979
    """

    all_ids = tuple(np.asarray(sample_ids, dtype=str).tolist())
    ref = np.asarray(reference, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    sigma = np.asarray(sigma_crlb, dtype=np.float64)

    valid = np.isfinite(ref) & np.isfinite(pred) & np.isfinite(sigma) & (sigma > 0)
    n_invalid = int(np.sum(~valid))
    included_ids = tuple(np.asarray(all_ids)[valid].tolist())
    status = ResultStatus.VALID if included_ids else ResultStatus.INSUFFICIENT_DATA
    meta = build_meta(
        status=status,
        all_ids=all_ids,
        included_ids=included_ids,
        exclusion_reasons={"invalid_sigma_crlb_or_input": n_invalid} if n_invalid else {},
        method="standardized_error",
        configuration={
            "expected_value_under_correct_gaussian_crlb": EXPECTED_ABS_STANDARDIZED_ERROR_UNDER_CRLB,
        },
    )
    epsilon = np.abs(pred[valid] - ref[valid]) / sigma[valid]
    return PerSampleResult(meta=meta, values=epsilon)


@dataclass(frozen=True)
class VarianceRatioResult(AnalysisResult):
    r"""Result of :func:`variance_ratio`.

    Descriptive only — not a formal efficiency claim (spec/manuscript:
    the DL estimator may be biased and need not satisfy CRLB regularity
    conditions).

    Attributes
    ----------
    meta : AnalysisMeta
        Provenance and inclusion/exclusion record (inherited from
        :class:`~mrsvalidate.results.AnalysisResult`).
    empirical_sd : float
        The sample standard deviation :math:`\sigma_{\text{DL},m}` of
        the repeated-noise-realization predictions (``ddof=1``).
    sigma_crlb : float
        The externally supplied CRLB standard deviation
        :math:`\sigma_{\text{CRLB},m}` for this condition.
    ratio : float
        :math:`R_m = \sigma_{\text{DL},m} / \sigma_{\text{CRLB},m}`.
    excess_variance : float
        :math:`\Delta V_m = \sigma_{\text{DL},m}^2 - \sigma_{\text{CRLB},m}^2`.
    """

    empirical_sd: float
    sigma_crlb: float
    ratio: float
    excess_variance: float


def variance_ratio(
    repeated_predictions: NDArray[np.float64], sigma_crlb: float, *, sample_id: str = "condition"
) -> VarianceRatioResult:
    r"""Compute the information-standardized variability ratio at one fixed condition.

    Parameters
    ----------
    repeated_predictions : ndarray of float
        Predictions from ``K`` repeated noise realizations at one fixed
        condition (same ground truth, independent noise draws).
    sigma_crlb : float
        The externally supplied CRLB standard deviation
        :math:`\sigma_{\text{CRLB},m}` for this condition. Must be
        finite and strictly positive.
    sample_id : str, default "condition"
        Identifier recorded for this condition in the result metadata.

    Returns
    -------
    VarianceRatioResult
        See :class:`VarianceRatioResult`. ``meta.status`` is
        :attr:`~mrsvalidate.results.ResultStatus.INSUFFICIENT_DATA` (and
        all numeric fields ``nan``) if fewer than 2 finite predictions
        are available or `sigma_crlb` is not finite and strictly
        positive.

    Notes
    -----
    :math:`R_m` and the excess-variance diagnostic
    :math:`\Delta V_m` are descriptive quantities only, not a formal
    efficiency claim: the underlying (e.g. deep-learning) estimator may
    be biased and need not satisfy CRLB regularity conditions.

    Examples
    --------
    >>> import numpy as np
    >>> repeated_predictions = np.array([10.0, 10.5, 9.5, 10.2])
    >>> result = variance_ratio(repeated_predictions, 0.3)
    >>> round(result.empirical_sd, 4)
    0.4203
    >>> round(result.ratio, 4)
    1.4011
    """

    values = np.asarray(repeated_predictions, dtype=np.float64)
    valid = np.isfinite(values)
    k = int(np.sum(valid))
    all_ids = (sample_id,)

    if k < 2 or not np.isfinite(sigma_crlb) or sigma_crlb <= 0:
        meta = build_meta(
            status=ResultStatus.INSUFFICIENT_DATA,
            all_ids=all_ids,
            included_ids=(),
            method="variance_ratio",
        )
        return VarianceRatioResult(
            meta=meta,
            empirical_sd=float("nan"),
            sigma_crlb=float(sigma_crlb),
            ratio=float("nan"),
            excess_variance=float("nan"),
        )

    empirical_sd = float(np.std(values[valid], ddof=1))
    meta = build_meta(
        status=ResultStatus.VALID,
        all_ids=all_ids,
        included_ids=all_ids,
        method="variance_ratio",
        configuration={"k_realizations": k, "interpretation": "descriptive, not an efficiency claim"},
    )
    return VarianceRatioResult(
        meta=meta,
        empirical_sd=empirical_sd,
        sigma_crlb=float(sigma_crlb),
        ratio=empirical_sd / float(sigma_crlb),
        excess_variance=empirical_sd**2 - float(sigma_crlb) ** 2,
    )


def mahalanobis_squared_error(
    errors: NDArray[np.float64],
    covariance: NDArray[np.float64],
    sample_ids: NDArray[np.str_],
    *,
    allow_pseudoinverse: bool = False,
    rcond: float = 1e-15,
) -> PerSampleResult:
    r"""Compute the per-sample Mahalanobis squared error against a CRLB covariance.

    Parameters
    ----------
    errors : ndarray of float, shape (n_samples,) or (n_samples, M)
        Per-sample error vectors. A 1-D array is treated as ``M = 1``.
    covariance : ndarray of float, shape (M, M) or (n_samples, M, M)
        Either one CRLB covariance matrix shared by every sample, or
        one matrix per sample.
    sample_ids : ndarray of str
        Identifier for each sample.
    allow_pseudoinverse : bool, default False
        If ``False`` (default), a singular covariance excludes the
        corresponding sample(s) as invalid input. If ``True``, a
        singular covariance instead falls back to a Moore-Penrose
        pseudoinverse (via `rcond`), and the effective rank and
        tolerance used are recorded in ``meta.configuration``.
    rcond : float, default 1e-15
        Cutoff for small singular values, forwarded to
        :func:`numpy.linalg.pinv` and :func:`numpy.linalg.matrix_rank`
        when `allow_pseudoinverse` is ``True``.

    Returns
    -------
    PerSampleResult
        ``values`` holds :math:`D^2` for every sample with a finite
        error vector and a usable (non-singular, or pseudoinverted when
        allowed) covariance, aligned with ``meta.included_ids``.

    Notes
    -----
    .. math::

        D^2 = e^{\mathsf{T}} C_{\text{CRLB}}^{-1} e

    computed per sample via a Cholesky factorization and solve, never
    an explicit matrix inverse. A singular covariance yields
    :attr:`~mrsvalidate.results.ResultStatus.INVALID_INPUT` for the
    affected sample(s) unless `allow_pseudoinverse` is ``True``, in
    which case ``meta.configuration`` records the effective rank and
    tolerance used for every sample that required the pseudoinverse
    fallback.

    Examples
    --------
    >>> import numpy as np
    >>> errors = np.array([[1.0, 0.0], [0.0, 2.0]])
    >>> covariance = np.eye(2)
    >>> ids = np.array(["a", "b"])
    >>> result = mahalanobis_squared_error(errors, covariance, ids)
    >>> result.values
    array([1., 4.])
    """

    all_ids = tuple(np.asarray(sample_ids, dtype=str).tolist())
    e = np.asarray(errors, dtype=np.float64)
    if e.ndim == 1:
        e = e[:, np.newaxis]
    n = e.shape[0]

    cov = np.asarray(covariance, dtype=np.float64)
    shared = cov.ndim == 2
    if shared and (n == 0 or cov.shape != (e.shape[1], e.shape[1])):
        raise ValueError(f"covariance shape {cov.shape} incompatible with errors shape {e.shape}")
    if not shared and cov.shape != (n, e.shape[1], e.shape[1]):
        raise ValueError(
            f"per-sample covariance must have shape {(n, e.shape[1], e.shape[1])}, got {cov.shape}"
        )

    d2 = np.full(n, np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    pseudo_used = np.zeros(n, dtype=bool)
    effective_ranks: list[int] = []

    shared_factor = None
    if shared:
        try:
            shared_factor = cho_factor(cov)
        except LinAlgError:
            shared_factor = None

    for i in range(n):
        row = e[i]
        if not np.all(np.isfinite(row)):
            continue
        c_i = cov if shared else cov[i]
        if not np.all(np.isfinite(c_i)):
            continue

        try:
            factor = shared_factor if shared else cho_factor(c_i)
            if factor is None:
                raise LinAlgError("singular")
            x = cho_solve(factor, row)
            d2[i] = float(row @ x)
            valid[i] = True
        except LinAlgError:
            if not allow_pseudoinverse:
                continue
            pinv = np.linalg.pinv(c_i, rcond=rcond)
            rank = int(np.linalg.matrix_rank(c_i, tol=rcond * np.max(c_i.shape)))
            effective_ranks.append(rank)
            d2[i] = float(row @ pinv @ row)
            valid[i] = True
            pseudo_used[i] = True

    included_ids = tuple(np.asarray(all_ids)[valid].tolist())
    n_invalid = n - len(included_ids)
    status = ResultStatus.VALID if included_ids else ResultStatus.INVALID_INPUT
    config = {
        "inversion_method": "cholesky_solve",
        "pseudoinverse_enabled": allow_pseudoinverse,
    }
    if pseudo_used.any():
        config["pseudoinverse_used_for_n_samples"] = int(pseudo_used.sum())
        config["effective_ranks"] = effective_ranks
        config["rcond"] = rcond

    meta = build_meta(
        status=status,
        all_ids=all_ids,
        included_ids=included_ids,
        exclusion_reasons={"singular_or_invalid_covariance": n_invalid} if n_invalid else {},
        method="mahalanobis_squared_error",
        configuration=config,
    )
    return PerSampleResult(meta=meta, values=d2[valid])
