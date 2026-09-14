"""Metabolite-specific SNR (spec eqs. 1-2 / manuscript spectral & power SNR).

Two distinct, explicitly named definitions are supported and must
never be conflated: :func:`spectral_snr` (mSNR_S), the peak-amplitude
definition, and :func:`power_snr` (mSNR_P), the RMS-power definition
following the manuscript's *first* displayed equality.

Both functions operate over a batch of samples (one value per sample,
like every other analysis function in this package) and never guess a
spectral noise region or FID tail from hard-coded indices — the noise
scale is always supplied by the caller, per-sample or as one shared
scalar.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ..results import PerSampleResult, ResultStatus, build_meta

__all__ = ["power_snr", "spectral_snr", "spectral_snr_from_spectra"]


def _broadcast_sigma_noise(sigma_noise: float | NDArray[np.float64], n: int) -> NDArray[np.float64]:
    """Broadcast a scalar or per-sample noise scale to length `n`.

    Parameters
    ----------
    sigma_noise : float or ndarray of float
        Either one shared noise standard deviation, or one value per
        sample.
    n : int
        Number of samples to broadcast to.

    Returns
    -------
    sigma : ndarray of float
        Array of length `n`, either the broadcast scalar repeated `n`
        times or `sigma_noise` unchanged (when already length `n`).

    Raises
    ------
    ValueError
        If `sigma_noise` is neither a scalar nor of length `n`.
    """

    arr = np.atleast_1d(np.asarray(sigma_noise, dtype=np.float64))
    if arr.size == 1:
        return np.full(n, float(arr[0]))
    if arr.size != n:
        raise ValueError(
            f"sigma_noise must be a scalar or have length {n} (one per sample), "
            f"got length {arr.size}"
        )
    return arr


def spectral_snr(
    peak_real_amplitude: NDArray[np.float64],
    sigma_noise: float | NDArray[np.float64],
    sample_ids: NDArray[np.str_],
) -> PerSampleResult:
    """Metabolite spectral SNR (mSNR_S), the peak-amplitude definition.

    Computes, per sample, the ratio of the spectrum's peak real
    amplitude to the noise standard deviation. This module does not
    perform spectral simulation: `peak_real_amplitude` (already
    ``max(Re(x_n(omega)))`` for each sample) must be supplied by the
    caller — use :func:`spectral_snr_from_spectra` instead if a full
    frequency-domain spectrum is available per sample.

    Parameters
    ----------
    peak_real_amplitude : ndarray of float
        ``max(Re(x_n(omega)))`` for each sample, i.e. the peak real
        part of the sample's frequency-domain spectrum.
    sigma_noise : float or ndarray of float
        Noise standard deviation. Either one value shared across all
        samples, or one value per sample (same length as
        `peak_real_amplitude`).
    sample_ids : ndarray of str
        Sample identifier aligned with `peak_real_amplitude`.

    Returns
    -------
    result : PerSampleResult
        ``result.values`` holds ``peak_real_amplitude / sigma_noise``
        for each included sample, ordered like
        ``result.meta.included_ids``. A sample is excluded (reason
        ``"nonpositive_or_nonfinite_sigma_noise_or_amplitude"``) if
        its amplitude or ``sigma_noise`` is non-finite, or if
        ``sigma_noise <= 0``. ``result.meta.status`` is
        ``ResultStatus.INSUFFICIENT_DATA`` if no sample remains.
        ``result.meta.configuration["definition"]`` is always
        ``"manuscript_eq_spectral_snr"``.

    Notes
    -----
    .. math::

        \\text{mSNR}_S(n) = \\frac{\\max\\left(\\operatorname{Re}(x_n(\\omega))\\right)}
            {\\sigma_{\\text{noise}}}

    This is the peak-amplitude SNR definition and must not be confused
    with :func:`power_snr` (mSNR_P), the RMS-power definition.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.snr import spectral_snr
    >>> amplitude = np.array([10.0, 20.0])
    >>> ids = np.array(["a", "b"])
    >>> r = spectral_snr(amplitude, 2.0, ids)
    >>> r.values
    array([ 5., 10.])
    """

    all_ids = tuple(np.asarray(sample_ids, dtype=str).tolist())
    amplitude = np.asarray(peak_real_amplitude, dtype=np.float64)
    sigma = _broadcast_sigma_noise(sigma_noise, len(all_ids))

    valid = np.isfinite(amplitude) & np.isfinite(sigma) & (sigma > 0)
    n_invalid = int(np.sum(~valid))
    included_ids = tuple(np.asarray(all_ids)[valid].tolist())

    status = ResultStatus.VALID if np.any(valid) else ResultStatus.INSUFFICIENT_DATA
    meta = build_meta(
        status=status,
        all_ids=all_ids,
        included_ids=included_ids,
        exclusion_reasons=(
            {"nonpositive_or_nonfinite_sigma_noise_or_amplitude": n_invalid}
            if n_invalid
            else {}
        ),
        method="spectral_snr",
        configuration={"definition": "manuscript_eq_spectral_snr"},
    )
    values = amplitude[valid] / sigma[valid]
    return PerSampleResult(meta=meta, values=values)


def power_snr(
    fid_or_spectrum: Sequence[NDArray[np.complexfloating]],
    sigma_noise: float | NDArray[np.float64],
    sample_ids: NDArray[np.str_],
) -> PerSampleResult:
    """Metabolite power SNR (mSNR_P), the RMS-power definition.

    Computes, per sample, the RMS power of the sample's complex
    waveform divided by the noise standard deviation, following the
    manuscript's *first* displayed equality. The manuscript's
    subsequently expanded equality is treated as a typesetting
    discrepancy (a misplaced square root) and is deliberately not
    reproduced; every result instead records
    ``definition="manuscript_first_equality_rms_power"`` so the choice
    is always auditable.

    Parameters
    ----------
    fid_or_spectrum : sequence of ndarray of complex
        One complex waveform ``y`` per sample (a time-domain FID or a
        frequency-domain spectrum — the RMS-power computation is the
        same either way). Waveforms may have different lengths ``Ns``.
    sigma_noise : float or ndarray of float
        Noise standard deviation. Either one value shared across all
        samples, or one value per sample.
    sample_ids : ndarray of str
        Sample identifier aligned with `fid_or_spectrum`.

    Returns
    -------
    result : PerSampleResult
        ``result.values`` holds the RMS-power SNR for each included
        sample, ordered like ``result.meta.included_ids``. A sample is
        excluded (reason ``"invalid_snr_input"``) if its waveform is
        empty, contains a non-finite real or imaginary part, or if the
        corresponding ``sigma_noise`` is non-finite or non-positive.
        ``result.meta.status`` is ``ResultStatus.INSUFFICIENT_DATA`` if
        no sample remains.
        ``result.meta.configuration["definition"]`` is always
        ``"manuscript_first_equality_rms_power"``.

    Raises
    ------
    ValueError
        If `fid_or_spectrum` does not have exactly one entry per
        `sample_ids`.

    Notes
    -----
    .. math::

        \\text{mSNR}_P(n) = \\sqrt{\\frac{1}{\\sigma_{\\text{noise}}^2}
            \\cdot \\frac{1}{N_s}\\sum_{k=1}^{N_s} |y_n[k]|^2}

    This is the manuscript's *first* displayed equality. The
    manuscript's subsequently expanded form is not reproduced here, as
    it is judged to contain a typesetting error (a misplaced square
    root).

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.snr import power_snr
    >>> fid = [np.array([2.0 + 0j, 0.0 + 0j])]
    >>> ids = np.array(["a"])
    >>> r = power_snr(fid, 2.0, ids)
    >>> r.values
    array([0.70710678])
    >>> r.meta.configuration["definition"]
    'manuscript_first_equality_rms_power'
    """

    all_ids = tuple(np.asarray(sample_ids, dtype=str).tolist())
    n = len(all_ids)
    if len(fid_or_spectrum) != n:
        raise ValueError(
            f"fid_or_spectrum must have one waveform per sample_id "
            f"({n}), got {len(fid_or_spectrum)}"
        )
    sigma = _broadcast_sigma_noise(sigma_noise, n)

    values_list: list[float] = []
    valid_mask = np.zeros(n, dtype=bool)
    for i, waveform in enumerate(fid_or_spectrum):
        y = np.asarray(waveform)
        sigma_i = sigma[i]
        if (
            y.size == 0
            or not np.all(np.isfinite(y.real))
            or not np.all(np.isfinite(y.imag))
            or not np.isfinite(sigma_i)
            or sigma_i <= 0
        ):
            continue
        power = np.sum(np.abs(y) ** 2) / y.size
        values_list.append(float(np.sqrt(power / sigma_i**2)))
        valid_mask[i] = True

    included_ids = tuple(np.asarray(all_ids)[valid_mask].tolist())
    n_invalid = n - len(included_ids)
    status = ResultStatus.VALID if included_ids else ResultStatus.INSUFFICIENT_DATA
    meta = build_meta(
        status=status,
        all_ids=all_ids,
        included_ids=included_ids,
        exclusion_reasons={"invalid_snr_input": n_invalid} if n_invalid else {},
        method="power_snr",
        configuration={"definition": "manuscript_first_equality_rms_power"},
    )
    return PerSampleResult(meta=meta, values=np.asarray(values_list, dtype=np.float64))


def spectral_snr_from_spectra(
    spectra: Sequence[NDArray[np.complexfloating]],
    sigma_noise: float | NDArray[np.float64],
    sample_ids: NDArray[np.str_],
) -> PerSampleResult:
    """Convenience wrapper computing mSNR_S directly from each spectrum.

    For each sample, extracts ``peak_real_amplitude =
    max(Re(x_n(omega)))`` from the supplied **frequency-domain
    spectrum** ``x_n(omega)`` and forwards it to :func:`spectral_snr`.
    Use this when the noise-free (or noisy) spectrum itself is
    available rather than a precomputed peak amplitude — e.g. for the
    Monte Carlo characterization workflow in
    :mod:`mrsvalidate.analysis.monte_carlo`, which needs mSNR per
    simulated condition directly from its noise-free spectrum.

    Parameters
    ----------
    spectra : sequence of ndarray of complex
        One frequency-domain spectrum ``x_n(omega)`` per sample.
        Spectra may have different lengths.
    sigma_noise : float or ndarray of float
        Noise standard deviation. Either one value shared across all
        samples, or one value per sample.
    sample_ids : ndarray of str
        Sample identifier aligned with `spectra`.

    Returns
    -------
    result : PerSampleResult
        Identical in shape and semantics to the result of
        :func:`spectral_snr`; see that function for the meaning of
        ``result.values`` and the exclusion policy.

    Raises
    ------
    ValueError
        If `spectra` does not have exactly one entry per `sample_ids`.

    Notes
    -----
    .. math::

        \\text{mSNR}_S(n) = \\frac{\\max\\left(\\operatorname{Re}(x_n(\\omega))\\right)}
            {\\sigma_{\\text{noise}}}

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.snr import spectral_snr_from_spectra
    >>> spectra = [np.array([1.0 + 2j, 5.0 - 1j, 3.0 + 0j]), np.array([2.0 + 0j, 2.0 + 0j])]
    >>> ids = np.array(["a", "b"])
    >>> r = spectral_snr_from_spectra(spectra, 2.0, ids)
    >>> r.values
    array([2.5, 1. ])
    """

    all_ids = tuple(np.asarray(sample_ids, dtype=str).tolist())
    if len(spectra) != len(all_ids):
        raise ValueError(
            f"spectra must have one entry per sample_id ({len(all_ids)}), got {len(spectra)}"
        )
    peak_real_amplitude = np.array(
        [float(np.max(np.real(np.asarray(s)))) if np.asarray(s).size else np.nan for s in spectra],
        dtype=np.float64,
    )
    return spectral_snr(peak_real_amplitude, sigma_noise, sample_ids)
