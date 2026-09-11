import numpy as np
import pytest

from mrsvalidate.alignment import align_observations
from mrsvalidate.exceptions import AlignmentError
from mrsvalidate.schema import ObservationSet, ParameterRole, ParameterSchema, ParameterSpec

SCHEMA = ParameterSchema((ParameterSpec("NAA", ParameterRole.CONCENTRATION, "mM"),))


def make_obs(sample_ids, values):
    return ObservationSet(
        sample_ids=np.array(sample_ids), values=np.array(values).reshape(-1, 1), parameters=SCHEMA
    )


def test_paired_alignment_matches_by_id_and_reorders_deterministically():
    reference = make_obs(["c", "a", "b"], [3.0, 1.0, 2.0])
    candidate = make_obs(["b", "c", "a"], [20.0, 30.0, 10.0])

    aligned = align_observations(reference, candidate, pairing="paired")
    report = aligned.report

    assert report.matched_ids == ("a", "b", "c")
    assert report.n_reference_input == 3
    assert report.n_candidate_input == 3
    assert report.n_final == 3
    assert report.only_in_reference == ()
    assert report.only_in_candidate == ()
    assert report.pairing_status == "paired"

    np.testing.assert_array_equal(aligned.reference.values.ravel(), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(aligned.candidate.values.ravel(), [10.0, 20.0, 30.0])


def test_paired_alignment_reports_unmatched_ids():
    reference = make_obs(["a", "b", "c"], [1.0, 2.0, 3.0])
    candidate = make_obs(["b", "c", "d"], [20.0, 30.0, 40.0])

    aligned = align_observations(reference, candidate, pairing="paired")
    report = aligned.report

    assert report.matched_ids == ("b", "c")
    assert report.only_in_reference == ("a",)
    assert report.only_in_candidate == ("d",)
    assert report.exclusion_reasons == {"only_in_reference": 1, "only_in_candidate": 1}
    assert set(report.excluded_ids) == {"a", "d"}


def test_duplicate_ids_error_by_default():
    reference = make_obs(["a", "a", "b"], [1.0, 2.0, 3.0])
    candidate = make_obs(["a", "b"], [10.0, 20.0])
    with pytest.raises(AlignmentError):
        align_observations(reference, candidate, pairing="paired")


def test_duplicate_policy_first_keeps_first_occurrence():
    reference = make_obs(["a", "a", "b"], [1.0, 999.0, 3.0])
    candidate = make_obs(["a", "b"], [10.0, 20.0])
    aligned = align_observations(
        reference, candidate, pairing="paired", duplicate_policy="first"
    )
    assert aligned.report.duplicate_ids_reference == ("a",)
    naa = dict(zip(aligned.reference.sample_ids.tolist(), aligned.reference.values.ravel()))
    assert naa["a"] == 1.0


def test_duplicate_policy_aggregate_averages_values():
    reference = make_obs(["a", "a", "b"], [1.0, 3.0, 5.0])
    candidate = make_obs(["a", "b"], [10.0, 20.0])
    aligned = align_observations(
        reference, candidate, pairing="paired", duplicate_policy="aggregate"
    )
    naa = dict(zip(aligned.reference.sample_ids.tolist(), aligned.reference.values.ravel()))
    assert naa["a"] == pytest.approx(2.0)


def test_independent_pairing_does_not_intersect_ids():
    reference = make_obs(["a", "b", "c"], [1.0, 2.0, 3.0])
    candidate = make_obs(["x", "y"], [10.0, 20.0])
    aligned = align_observations(reference, candidate, pairing="independent")
    assert aligned.report.pairing_status == "independent"
    assert aligned.report.matched_ids == ()
    assert aligned.report.n_final == 5
    assert aligned.reference.n_samples == 3
    assert aligned.candidate.n_samples == 2
