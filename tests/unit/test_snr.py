import numpy as np
import pytest

from mrsvalidate.analysis import snr
from mrsvalidate.results import ResultStatus


def test_spectral_snr_basic():
    amplitude = np.array([10.0, 20.0])
    ids = np.array(["a", "b"])
    r = snr.spectral_snr(amplitude, 2.0, ids)
    assert r.meta.status is ResultStatus.VALID
    np.testing.assert_allclose(r.values, [5.0, 10.0])


def test_spectral_snr_excludes_nonpositive_sigma_rows():
    amplitude = np.array([10.0, 20.0])
    ids = np.array(["a", "b"])
    r = snr.spectral_snr(amplitude, np.array([2.0, -1.0]), ids)
    assert r.meta.n_used == 1
    assert r.meta.excluded_ids == ("b",)
    np.testing.assert_allclose(r.values, [5.0])


def test_power_snr_matches_manuscript_first_equality():
    fid = [np.array([1.0 + 0j, 1.0 + 0j])]
    ids = np.array(["a"])
    r = snr.power_snr(fid, 1.0, ids)
    assert r.meta.status is ResultStatus.VALID
    assert r.meta.configuration["definition"] == "manuscript_first_equality_rms_power"
    np.testing.assert_allclose(r.values, [1.0])


def test_power_snr_hand_computed_with_nontrivial_noise():
    # y = [2, 0], Ns=2, sigma_noise=2 -> power = (4+0)/2=2; snr=sqrt(2/4)=sqrt(0.5)
    fid = [np.array([2.0 + 0j, 0.0 + 0j])]
    ids = np.array(["a"])
    r = snr.power_snr(fid, 2.0, ids)
    np.testing.assert_allclose(r.values, [np.sqrt(0.5)])


def test_power_snr_excludes_empty_waveform():
    fid = [np.array([1.0 + 0j]), np.array([], dtype=complex)]
    ids = np.array(["a", "b"])
    r = snr.power_snr(fid, 1.0, ids)
    assert r.meta.n_used == 1
    assert r.meta.excluded_ids == ("b",)


def test_spectral_snr_from_spectra_uses_max_real_part():
    spectra = [np.array([1.0 + 2j, 5.0 - 1j, 3.0 + 0j]), np.array([2.0 + 0j, 2.0 + 0j])]
    ids = np.array(["a", "b"])
    r = snr.spectral_snr_from_spectra(spectra, 2.0, ids)
    np.testing.assert_allclose(r.values, [5.0 / 2.0, 2.0 / 2.0])


def test_spectral_snr_from_spectra_rejects_mismatched_length():
    spectra = [np.array([1.0 + 0j])]
    ids = np.array(["a", "b"])
    with pytest.raises(ValueError):
        snr.spectral_snr_from_spectra(spectra, 1.0, ids)
