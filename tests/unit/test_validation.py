import numpy as np
import pytest

from mrsvalidate.exceptions import ValidationError
from mrsvalidate.schema import ObservationSet, ParameterRole, ParameterSchema, ParameterSpec
from mrsvalidate.validation import (
    find_duplicate_ids,
    require_non_empty,
    require_role,
    validate_observation_set,
)


def make_obs(sample_ids, values):
    schema = ParameterSchema(
        (
            ParameterSpec("NAA", ParameterRole.CONCENTRATION, "mM"),
            ParameterSpec("mSNR", ParameterRole.SNR, None),
        )
    )
    return ObservationSet(sample_ids=np.array(sample_ids), values=np.array(values), parameters=schema)


def test_find_duplicate_ids():
    assert find_duplicate_ids(np.array(["a", "b", "a", "c", "b", "b"])) == ("a", "b")


def test_validate_observation_set_reports_duplicates_without_raising():
    obs = make_obs(["a", "a", "b"], [[1.0, 10.0], [2.0, 11.0], [3.0, 12.0]])
    report = validate_observation_set(obs)
    assert report.has_duplicate_ids
    assert report.duplicate_sample_ids == ("a",)


def test_validate_observation_set_empty_dataset_warns():
    schema = ParameterSchema((ParameterSpec("NAA", ParameterRole.CONCENTRATION, "mM"),))
    obs = ObservationSet(
        sample_ids=np.array([], dtype=str), values=np.zeros((0, 1)), parameters=schema
    )
    report = validate_observation_set(obs)
    assert report.n_samples == 0
    assert any("zero samples" in w for w in report.warnings)


def test_require_role_success_and_failure():
    obs = make_obs(["a", "b"], [[1.0, 10.0], [2.0, 11.0]])
    assert require_role(obs, ParameterRole.SNR) == (1,)
    with pytest.raises(ValidationError):
        require_role(obs, ParameterRole.PHASE_ZERO_ORDER)


def test_require_non_empty():
    obs = make_obs(["a"], [[1.0, 10.0]])
    require_non_empty(obs)  # does not raise

    schema = ParameterSchema((ParameterSpec("NAA", ParameterRole.CONCENTRATION, "mM"),))
    empty_obs = ObservationSet(
        sample_ids=np.array([], dtype=str), values=np.zeros((0, 1)), parameters=schema
    )
    with pytest.raises(ValidationError):
        require_non_empty(empty_obs)
