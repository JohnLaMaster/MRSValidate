import numpy as np
import pytest

from mrsvalidate.analysis import monte_carlo as mc
from mrsvalidate.analysis import information
from mrsvalidate.results import ResultStatus


def test_generate_noise_realizations_shape_and_reproducibility():
    waveform = np.array([1.0 + 0j, 2.0 + 0j, 3.0 + 0j])
    r1 = mc.generate_noise_realizations(waveform, sigma_noise=0.5, n_realizations=10, seed=0)
    r2 = mc.generate_noise_realizations(waveform, sigma_noise=0.5, n_realizations=10, seed=0)
    assert r1.shape == (10, 3)
    np.testing.assert_array_equal(r1, r2)


def test_generate_noise_realizations_mean_recovers_noise_free_waveform():
    waveform = np.array([5.0 + 2j, -3.0 + 1j])
    realizations = mc.generate_noise_realizations(waveform, sigma_noise=0.2, n_realizations=20000, seed=0)
    mean_realization = np.mean(realizations, axis=0)
    np.testing.assert_allclose(mean_realization, waveform, atol=0.05)


def test_generate_noise_realizations_rejects_nonpositive_sigma():
    waveform = np.array([1.0 + 0j])
    with pytest.raises(ValueError):
        mc.generate_noise_realizations(waveform, sigma_noise=0.0, n_realizations=5)


def test_monte_carlo_condition_hand_computed():
    predictions = np.array([9.0, 10.0, 11.0, 10.0])
    true_value = 10.0
    r = mc.monte_carlo_condition(predictions, true_value, "cond_0")
    assert r.mean_prediction == pytest.approx(10.0)
    assert r.bias == pytest.approx(0.0)
    assert r.variance == pytest.approx(np.var(predictions, ddof=1))
    assert r.rmse == pytest.approx(np.sqrt(np.mean((predictions - true_value) ** 2)))


def test_monte_carlo_condition_with_crlb_matches_variance_ratio():
    predictions = np.array([9.5, 10.5, 10.0, 9.8, 10.2])
    sigma_crlb = 0.3
    r = mc.monte_carlo_condition(predictions, 10.0, "cond_0", sigma_crlb=sigma_crlb)
    expected = information.variance_ratio(predictions, sigma_crlb)
    assert r.variability_ratio == pytest.approx(expected.ratio)
    assert r.excess_variance == pytest.approx(expected.excess_variance)


def test_monte_carlo_condition_insufficient_realizations():
    r = mc.monte_carlo_condition(np.array([1.0]), 1.0, "cond_0")
    assert r.meta.status is ResultStatus.INSUFFICIENT_DATA


def test_monte_carlo_characterization_aggregates_across_conditions():
    rng = np.random.default_rng(0)
    j = 5
    true_values = np.linspace(1, 10, j)
    predictions = [true_values[i] + rng.normal(scale=0.1 * (i + 1), size=20) for i in range(j)]
    condition_ids = np.array([f"cond{i}" for i in range(j)])
    snr_covariate = np.array([50.0, 40.0, 30.0, 20.0, 10.0])

    result = mc.monte_carlo_characterization(
        predictions, true_values, condition_ids,
        sigma_crlb=np.full(j, 0.1),
        covariates={"mSNR": snr_covariate},
    )
    assert result.meta.status is ResultStatus.VALID
    assert result.bias.shape == (j,)
    assert result.covariates["mSNR"].shape == (j,)
    # Variance should generally increase as the injected noise scale grows.
    assert result.variance[-1] > result.variance[0]


def test_monte_carlo_characterization_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        mc.monte_carlo_characterization(
            [np.array([1.0, 2.0])], np.array([1.0, 2.0]), np.array(["a"])
        )
