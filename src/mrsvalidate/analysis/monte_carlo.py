"""Monte Carlo characterization of failure modes.

Implements the manuscript section "Monte Carlo Characterization of
Failure Modes", specifically the equations following eq.
``sigma_crlb`` at manuscript lines ~934-1016.

Notes
-----
**Hierarchical design.** An outer loop samples ``J`` underlying
synthetic parameter configurations :math:`\\theta_j \\sim p_{sim}(\\theta)`;
for each fixed :math:`\\theta_j`, an inner loop of ``K`` repeated noisy
realizations

.. math::

    x_{j,k} = f(\\theta_j) + \\epsilon_{j,k}, \\quad
    \\epsilon_{j,k} = \\epsilon_R + i \\epsilon_I, \\quad
    \\epsilon_R, \\epsilon_I \\sim \\mathcal{N}(0, \\sigma_j^2) \\text{ i.i.d.}

is generated — only the noise realization changes across ``k``; the
underlying (noise-free) signal :math:`f(\\theta_j)` and the noise scale
:math:`\\sigma_j` are held fixed. The (external) DL estimator is then
applied to each realization, :math:`\\hat{c}_{j,k} = g_{DL}(x_{j,k})`.
**This library never calls, trains, or wraps that estimator** —
consistent with the architectural rule that mrsvalidate does not embed
model-specific code — so ``g_DL``'s predictions are always supplied *to*
this module as plain arrays, already computed elsewhere.

What this module *does* provide for the in silico regime:

1. :func:`generate_noise_realizations` — reproducible generation of the
   ``K`` noisy realizations ``x_{j,k}`` from a supplied noise-free
   waveform ``f(theta_j)`` and noise scale, so the noise model itself
   (not model fitting) is deterministic given a seed;
2. per-condition characterization (:func:`monte_carlo_condition`):
   mean prediction, empirical bias, empirical variance, RMSE (eqs. at
   lines 972-986), and — when an external CRLB reference is supplied
   for that condition — the univariate variability ratio ``R_{m,j}``
   and excess variance ``Delta_V_{m,j}`` (eqs. at lines 1006-1012), by
   delegating to :func:`mrsvalidate.analysis.information.variance_ratio`
   rather than recomputing that formula;
3. hierarchical aggregation across all ``J`` conditions
   (:func:`monte_carlo_characterization`), retaining any covariates
   (mSNR, concentration, linewidth, ...) supplied per condition, per
   the manuscript's explicit instruction to retain "the simulation
   parameters and corresponding estimates ... before any aggregation
   across conditions is performed" — the resulting per-condition table
   is exactly the input :mod:`mrsvalidate.analysis.response_curves`
   expects.

**mSNR is normally an external input, not something this module
computes.** In the manuscript's standard *in silico* workflow,
metabolite-level SNR is computed once during data simulation (from the
known noise-free basis functions and the sampled noise level) and
supplied downstream alongside the simulation parameters — it is not
recomputed per Monte Carlo replicate. The *only* setting where mSNR
must be derived rather than supplied is *in vivo*, where it is
reconstructed post-hoc from the conventional algorithm's **fitted**
parameters (modulating the basis functions with the fit and measuring
the resulting waveform's power) rather than from any noise-free
ground truth, since none exists in vivo.

Consequently, the primary way to bring mSNR (or any other controlled
variable — concentration, linewidth, frequency offset) into this
module is simply to pass it as a precomputed array in
``covariates`` to :func:`monte_carlo_characterization` — most callers
already have it. :func:`mrsvalidate.analysis.snr.spectral_snr_from_spectra`
(re-exported here as :data:`condition_spectral_snr`,
``max(Re(spectrum)) / sigma_noise``) is provided only for the
narrower case where a caller has noise-free *in silico* spectra on
hand and has not already computed mSNR from them; it is not invoked
implicitly anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ..results import AnalysisResult, ResultStatus, build_meta
from .information import variance_ratio
from .snr import spectral_snr_from_spectra as condition_spectral_snr

__all__ = [
    "generate_noise_realizations",
    "monte_carlo_condition",
    "monte_carlo_characterization",
    "condition_spectral_snr",
    "MonteCarloConditionResult",
    "MonteCarloCharacterizationResult",
]


def generate_noise_realizations(
    noise_free_waveform: NDArray[np.complexfloating],
    sigma_noise: float,
    n_realizations: int,
    *,
    seed: int | None = None,
) -> NDArray[np.complex128]:
    r"""Generate repeated noisy realizations of one fixed noise-free waveform.

    Implements
    :math:`x_{j,k} = f(\theta_j) + \epsilon_{j,k}`, with
    :math:`\epsilon_R, \epsilon_I \sim \mathcal{N}(0, \sigma_{noise}^2)`
    i.i.d., for :math:`k = 1 \ldots` `n_realizations`. Only the noise
    realization varies across ``k``; the underlying condition
    :math:`\theta_j` (encoded in `noise_free_waveform`) and the noise
    scale are held fixed — this is the inner loop of the module's
    hierarchical Monte Carlo design (see module Notes).

    Parameters
    ----------
    noise_free_waveform : ndarray of complex, shape (n_points,)
        The fixed, noise-free signal :math:`f(\theta_j)` for one
        underlying condition.
    sigma_noise : float
        Standard deviation of the (real and imaginary, independently
        applied) Gaussian noise. Must be finite and strictly positive.
    n_realizations : int
        Number of noisy realizations ``K`` to generate.
    seed : int or None, default None
        Seed for the random number generator; None draws fresh entropy.

    Returns
    -------
    ndarray of complex128, shape (n_realizations, n_points)
        Row ``k`` is one Monte Carlo realization ``x_{j,k}`` of the
        fixed underlying condition; only the noise differs from row to
        row.

    Raises
    ------
    ValueError
        If `sigma_noise` is not finite and strictly positive.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.monte_carlo import generate_noise_realizations
    >>> waveform = np.array([1.0 + 0j, 2.0 + 0j, 3.0 + 0j, 2.0 + 0j, 1.0 + 0j])
    >>> realizations = generate_noise_realizations(waveform, sigma_noise=0.1, n_realizations=3, seed=0)
    >>> realizations.shape
    (3, 5)
    >>> print(np.round(realizations[0], 4))
    [1.0126-0.0732j 1.9868-0.0544j 3.064 -0.0316j 2.0105+0.0412j
     0.9464+0.1043j]
    """

    if sigma_noise <= 0 or not np.isfinite(sigma_noise):
        raise ValueError(f"sigma_noise must be finite and positive, got {sigma_noise}")

    waveform = np.asarray(noise_free_waveform, dtype=np.complex128)
    n = waveform.shape[0]
    rng = np.random.default_rng(seed)
    noise_real = rng.normal(scale=sigma_noise, size=(n_realizations, n))
    noise_imag = rng.normal(scale=sigma_noise, size=(n_realizations, n))
    return waveform[np.newaxis, :] + noise_real + 1j * noise_imag


@dataclass(frozen=True)
class MonteCarloConditionResult(AnalysisResult):
    """Result of :func:`monte_carlo_condition` for one fixed condition.

    Attributes
    ----------
    meta : AnalysisMeta
        Provenance and inclusion/exclusion record for this computation.
    condition_id : str
        Identifier of the underlying condition :math:`\\theta_j`.
    n_realizations : int
        Number of valid (finite) predictions ``K`` used.
    mean_prediction : float
        Mean of the ``K`` predictions, :math:`\\bar{\\hat{c}}_j`.
    bias : float
        Empirical bias, ``mean_prediction - true_value``.
    variance : float
        Empirical variance of the predictions (``ddof=1``).
    rmse : float
        Root-mean-square error of the predictions against `true_value`.
    sigma_crlb : float or None
        The CRLB reference standard deviation supplied for this
        condition, unchanged, or None if not supplied.
    variability_ratio : float or None
        Variability ratio :math:`R_j` = empirical SD / `sigma_crlb`, or
        None if `sigma_crlb` was not supplied (or the underlying
        :func:`~mrsvalidate.analysis.information.variance_ratio` call was
        not valid).
    excess_variance : float or None
        Excess variance :math:`\\Delta V_j` = empirical variance -
        `sigma_crlb`:sup:`2`, or None under the same conditions as
        `variability_ratio`.
    """

    condition_id: str
    n_realizations: int
    mean_prediction: float
    bias: float
    variance: float
    rmse: float
    sigma_crlb: float | None
    variability_ratio: float | None
    excess_variance: float | None


def monte_carlo_condition(
    predictions: NDArray[np.float64],
    true_value: float,
    condition_id: str,
    *,
    sigma_crlb: float | None = None,
) -> MonteCarloConditionResult:
    r"""Characterize one fixed condition from its repeated-noise predictions.

    Summarizes the ``K`` repeated-noise-realization predictions
    :math:`\hat{c}_{j,1..K}` of a single, fixed underlying condition
    :math:`\theta_j` by their mean, empirical bias, empirical variance,
    and RMSE (manuscript eqs. at lines 972-986).

    Parameters
    ----------
    predictions : ndarray of float, shape (n_realizations,)
        The DL estimator's predictions :math:`\hat{c}_{j,k}` across the
        ``K`` noise realizations of this condition, computed externally
        (this module never calls the estimator itself).
    true_value : float
        The known ground-truth value for this condition, :math:`c_j`.
    condition_id : str
        Identifier of this condition, used for bookkeeping in the
        returned result's ``meta``.
    sigma_crlb : float or None, default None
        The condition's CRLB reference standard deviation, an external
        input (this module never derives a CRLB itself). If supplied,
        the variability ratio and excess variance are also reported via
        :func:`mrsvalidate.analysis.information.variance_ratio`.

    Returns
    -------
    MonteCarloConditionResult
        Mean prediction, bias, variance, RMSE, and (if `sigma_crlb` was
        supplied) the variability ratio and excess variance. See
        :class:`MonteCarloConditionResult` for the full field list.

    Notes
    -----
    Fewer than 2 finite predictions yields a NaN-filled result with
    ``meta.status = ResultStatus.INSUFFICIENT_DATA``.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.monte_carlo import monte_carlo_condition
    >>> predictions = np.array([4.8, 5.1, 4.9, 5.3, 4.7])
    >>> result = monte_carlo_condition(predictions, true_value=5.0, condition_id="cond_a", sigma_crlb=0.2)
    >>> print(f"{result.mean_prediction:.4f} {result.bias:.4f} {result.rmse:.4f}")
    4.9600 -0.0400 0.2191
    >>> print(f"{result.variability_ratio:.4f} {result.excess_variance:.4f}")
    1.2042 0.0180
    """

    values = np.asarray(predictions, dtype=np.float64)
    valid = np.isfinite(values)
    k = int(np.sum(valid))
    all_ids = (condition_id,)

    if k < 2:
        meta = build_meta(
            status=ResultStatus.INSUFFICIENT_DATA, all_ids=all_ids, included_ids=(),
            method="monte_carlo_condition",
        )
        return MonteCarloConditionResult(
            meta=meta, condition_id=condition_id, n_realizations=k,
            mean_prediction=float("nan"), bias=float("nan"), variance=float("nan"),
            rmse=float("nan"), sigma_crlb=sigma_crlb, variability_ratio=None, excess_variance=None,
        )

    v = values[valid]
    mean_prediction = float(np.mean(v))
    bias = mean_prediction - true_value
    variance = float(np.var(v, ddof=1))
    rmse = float(np.sqrt(np.mean((v - true_value) ** 2)))

    ratio: float | None = None
    excess: float | None = None
    if sigma_crlb is not None:
        vr = variance_ratio(v, sigma_crlb, sample_id=condition_id)
        if vr.meta.status is ResultStatus.VALID:
            ratio, excess = vr.ratio, vr.excess_variance

    meta = build_meta(
        status=ResultStatus.VALID, all_ids=all_ids, included_ids=all_ids,
        method="monte_carlo_condition",
        configuration={"n_realizations": k, "has_crlb_reference": sigma_crlb is not None},
    )
    return MonteCarloConditionResult(
        meta=meta, condition_id=condition_id, n_realizations=k,
        mean_prediction=mean_prediction, bias=bias, variance=variance, rmse=rmse,
        sigma_crlb=sigma_crlb, variability_ratio=ratio, excess_variance=excess,
    )


@dataclass(frozen=True)
class MonteCarloCharacterizationResult(AnalysisResult):
    """Result of :func:`monte_carlo_characterization` across all conditions.

    One row per condition ``j``, retained *before* any aggregation
    across conditions (manuscript: report simulation parameters and
    per-condition estimates before aggregating).

    Attributes
    ----------
    meta : AnalysisMeta
        Provenance and inclusion/exclusion record for this computation.
    condition_ids : tuple of str
        Identifier of each condition, in input order.
    n_realizations : ndarray of int
        Number of valid predictions used per condition.
    mean_prediction : ndarray of float
        Per-condition mean prediction.
    bias : ndarray of float
        Per-condition empirical bias.
    variance : ndarray of float
        Per-condition empirical variance.
    rmse : ndarray of float
        Per-condition RMSE.
    sigma_crlb : ndarray of float
        Per-condition CRLB reference standard deviation as supplied (NaN
        where not supplied).
    variability_ratio : ndarray of float
        Per-condition variability ratio :math:`R_j` (NaN where a CRLB
        reference was not supplied or not valid).
    excess_variance : ndarray of float
        Per-condition excess variance :math:`\\Delta V_j` (NaN under the
        same conditions as `variability_ratio`).
    covariates : Mapping of str to ndarray of float
        Whatever controlled experimental variables (mSNR, concentration,
        linewidth, frequency offset, ...) the caller supplied per
        condition, ready for
        :mod:`mrsvalidate.analysis.response_curves`.
    """

    condition_ids: tuple[str, ...]
    n_realizations: NDArray[np.int64]
    mean_prediction: NDArray[np.float64]
    bias: NDArray[np.float64]
    variance: NDArray[np.float64]
    rmse: NDArray[np.float64]
    sigma_crlb: NDArray[np.float64]
    variability_ratio: NDArray[np.float64]
    excess_variance: NDArray[np.float64]
    covariates: Mapping[str, NDArray[np.float64]]


def monte_carlo_characterization(
    predictions: Sequence[NDArray[np.float64]],
    true_values: NDArray[np.float64],
    condition_ids: NDArray[np.str_],
    *,
    sigma_crlb: NDArray[np.float64] | None = None,
    covariates: Mapping[str, NDArray[np.float64]] | None = None,
) -> MonteCarloCharacterizationResult:
    """Aggregate per-condition Monte Carlo characterization across conditions.

    Calls :func:`monte_carlo_condition` once per condition and collects
    the results into per-condition arrays, retaining every condition's
    row (and any supplied `covariates`) rather than collapsing across
    conditions — the hierarchical aggregation step of the module's
    design (see module Notes).

    Parameters
    ----------
    predictions : sequence of ndarray of float
        Length-``J`` sequence, one 1D array of ``K_j`` realizations per
        condition. ``K`` may vary by condition — the manuscript's design
        does not require a balanced inner loop.
    true_values : ndarray of float, shape (n_conditions,)
        Ground-truth value for each condition, aligned with
        `condition_ids`.
    condition_ids : ndarray of str, shape (n_conditions,)
        Identifier for each condition, ``J`` in total.
    sigma_crlb : ndarray of float, shape (n_conditions,), or None, default None
        CRLB reference standard deviation per condition, an external
        input (see module Notes on mSNR/CRLB always coming from outside
        this module). None disables the variability-ratio/excess-variance
        diagnostics for every condition.
    covariates : Mapping of str to ndarray of float, or None, default None
        Named controlled-variable arrays (mSNR, concentration, linewidth,
        ...), each of length ``J`` and aligned with `condition_ids`.
        Passed through unchanged into the result for downstream response
        curve analysis; this function never computes or infers them.

    Returns
    -------
    MonteCarloCharacterizationResult
        Per-condition mean prediction, bias, variance, RMSE, and (when
        available) variability ratio/excess variance, plus the passed
        through `covariates`. See :class:`MonteCarloCharacterizationResult`
        for the full field list.

    Raises
    ------
    ValueError
        If `predictions` does not have exactly one entry per
        `condition_ids`, or if `true_values` does not have length
        ``J = len(condition_ids)``.

    Notes
    -----
    A condition whose own :func:`monte_carlo_condition` call is not
    ``ResultStatus.VALID`` (e.g. fewer than 2 finite predictions) is
    excluded from ``meta.included_ids`` and counted under
    ``exclusion_reasons["insufficient_realizations"]``, but its row is
    still present (NaN-filled) in every returned array — no condition is
    silently dropped from the table.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.monte_carlo import monte_carlo_characterization
    >>> predictions = [np.array([4.8, 5.1, 4.9, 5.3, 4.7]), np.array([9.6, 10.2, 9.9, 10.1])]
    >>> true_values = np.array([5.0, 10.0])
    >>> condition_ids = np.array(["cond_a", "cond_b"])
    >>> sigma_crlb = np.array([0.2, 0.3])
    >>> covariates = {"mSNR": np.array([15.0, 20.0])}
    >>> result = monte_carlo_characterization(
    ...     predictions, true_values, condition_ids,
    ...     sigma_crlb=sigma_crlb, covariates=covariates,
    ... )
    >>> result.meta.status.value
    'valid'
    >>> print(np.round(result.mean_prediction, 4))
    [4.96 9.95]
    >>> print(np.round(result.rmse, 4))
    [0.2191 0.2345]
    >>> result.covariates
    {'mSNR': array([15., 20.])}
    """

    all_ids = tuple(np.asarray(condition_ids, dtype=str).tolist())
    j = len(all_ids)
    if len(predictions) != j:
        raise ValueError(f"predictions must have one entry per condition_id ({j}), got {len(predictions)}")
    true_vals = np.asarray(true_values, dtype=np.float64)
    if true_vals.shape[0] != j:
        raise ValueError(f"true_values must have length {j}, got {true_vals.shape[0]}")
    crlb_arr = (
        np.asarray(sigma_crlb, dtype=np.float64) if sigma_crlb is not None else np.full(j, np.nan)
    )

    per_condition = [
        monte_carlo_condition(
            predictions[i], float(true_vals[i]), all_ids[i],
            sigma_crlb=(float(crlb_arr[i]) if np.isfinite(crlb_arr[i]) else None),
        )
        for i in range(j)
    ]

    valid_mask = np.array([r.meta.status is ResultStatus.VALID for r in per_condition])
    included_ids = tuple(np.asarray(all_ids)[valid_mask].tolist())
    status = ResultStatus.VALID if included_ids else ResultStatus.INSUFFICIENT_DATA

    def _field(name: str) -> NDArray[np.float64]:
        return np.array([getattr(r, name) if getattr(r, name) is not None else np.nan for r in per_condition])

    meta = build_meta(
        status=status, all_ids=all_ids, included_ids=included_ids,
        exclusion_reasons={"insufficient_realizations": int(np.sum(~valid_mask))} if np.any(~valid_mask) else {},
        method="monte_carlo_characterization",
        configuration={"n_conditions": j},
    )
    return MonteCarloCharacterizationResult(
        meta=meta,
        condition_ids=all_ids,
        n_realizations=np.array([r.n_realizations for r in per_condition]),
        mean_prediction=_field("mean_prediction"),
        bias=_field("bias"),
        variance=_field("variance"),
        rmse=_field("rmse"),
        sigma_crlb=crlb_arr,
        variability_ratio=_field("variability_ratio"),
        excess_variance=_field("excess_variance"),
        covariates=dict(covariates or {}),
    )
