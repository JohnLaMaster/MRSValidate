import numpy as np
import pytest

from mrsvalidate.analysis import clustering as cl
from mrsvalidate.results import ResultStatus


def _simulate_one_way(rng, n_subjects=20, n_per_subject=10, between_sd=2.0, within_sd=1.0):
    subject_ids = np.repeat([f"subj{i}" for i in range(n_subjects)], n_per_subject)
    subject_effects = rng.normal(scale=between_sd, size=n_subjects)
    values = np.repeat(subject_effects, n_per_subject) + rng.normal(
        scale=within_sd, size=n_subjects * n_per_subject
    )
    sample_ids = np.array([f"s{i}" for i in range(len(values))])
    return values, subject_ids, sample_ids


def test_one_way_icc_recovers_approximate_rho():
    rng = np.random.default_rng(42)
    between_sd, within_sd = 2.0, 1.0
    values, subject_ids, sample_ids = _simulate_one_way(
        rng, between_sd=between_sd, within_sd=within_sd
    )
    r = cl.one_way_icc(values, subject_ids, sample_ids)
    assert r.meta.status is ResultStatus.VALID
    true_rho = between_sd**2 / (between_sd**2 + within_sd**2)
    assert r.rho == pytest.approx(true_rho, abs=0.15)
    assert 0.0 <= r.rho <= 1.0
    assert r.design_effect >= 1.0
    assert r.n_eff <= r.n_total


def test_one_way_icc_near_zero_when_no_subject_effect():
    rng = np.random.default_rng(7)
    values, subject_ids, sample_ids = _simulate_one_way(
        rng, between_sd=1e-6, within_sd=1.0
    )
    r = cl.one_way_icc(values, subject_ids, sample_ids)
    assert r.rho < 0.1
    assert r.n_eff == pytest.approx(r.n_total, rel=0.1)


def test_one_way_icc_insufficient_data_for_single_subject():
    values = np.array([1.0, 2.0, 3.0])
    subject_ids = np.array(["s1", "s1", "s1"])
    sample_ids = np.array(["a", "b", "c"])
    r = cl.one_way_icc(values, subject_ids, sample_ids)
    assert r.meta.status is ResultStatus.INSUFFICIENT_DATA


def test_one_way_icc_excludes_nan_rows():
    rng = np.random.default_rng(1)
    values, subject_ids, sample_ids = _simulate_one_way(rng)
    values = values.copy()
    values[0] = np.nan
    r = cl.one_way_icc(values, subject_ids, sample_ids)
    assert r.meta.n_omitted == 1
    assert r.meta.exclusion_reasons == {"nan_or_inf": 1}


def test_cross_classified_icc_runs_and_reports_both_dimensions():
    rng = np.random.default_rng(3)
    n_metabolites, n_subjects, n_spectra_per_subject = 5, 8, 4
    metabolite_effects = rng.normal(scale=1.5, size=n_metabolites)
    subject_effects = rng.normal(scale=1.0, size=n_subjects)

    rows = []
    for si in range(n_subjects):
        for ni in range(n_spectra_per_subject):
            spectrum_id = f"subj{si}_spec{ni}"
            for mi in range(n_metabolites):
                val = (
                    metabolite_effects[mi]
                    + subject_effects[si]
                    + rng.normal(scale=0.5)
                )
                rows.append((val, f"met{mi}", f"subj{si}", spectrum_id))

    values = np.array([r[0] for r in rows])
    metabolite_ids = np.array([r[1] for r in rows])
    subject_ids = np.array([r[2] for r in rows])
    sample_ids = np.array([f"{r[3]}_{r[1]}" for r in rows])

    r = cl.cross_classified_icc(values, metabolite_ids, subject_ids, sample_ids)
    assert r.meta.status is ResultStatus.VALID
    assert 0.0 <= r.rho_metabolite <= 1.0
    assert 0.0 <= r.rho_subject <= 1.0
    assert r.n_metabolites == n_metabolites
    assert r.n_subjects == n_subjects
