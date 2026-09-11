import numpy as np
import pytest

from mrsvalidate.exceptions import SchemaError
from mrsvalidate.schema import (
    ObservationSet,
    ParameterRole,
    ParameterSchema,
    ParameterSpec,
)


def make_schema() -> ParameterSchema:
    return ParameterSchema(
        (
            ParameterSpec("NAA", ParameterRole.CONCENTRATION, "mM", metabolite="NAA"),
            ParameterSpec("Cr", ParameterRole.CONCENTRATION, "mM", metabolite="Cr"),
            ParameterSpec("linewidth", ParameterRole.LORENTZIAN_LINESHAPE, "Hz"),
        )
    )


def test_parameter_schema_rejects_duplicate_names():
    with pytest.raises(SchemaError):
        ParameterSchema(
            (
                ParameterSpec("NAA", ParameterRole.CONCENTRATION, "mM"),
                ParameterSpec("NAA", ParameterRole.CONCENTRATION, "mM"),
            )
        )


def test_parameter_schema_lookup():
    schema = make_schema()
    assert schema.names == ("NAA", "Cr", "linewidth")
    assert schema.index_of("Cr") == 1
    assert schema.indices_by_role(ParameterRole.CONCENTRATION) == (0, 1)
    assert schema.indices_by_metabolite("NAA") == (0,)


def test_observation_set_valid_construction():
    schema = make_schema()
    obs = ObservationSet(
        sample_ids=np.array(["s1", "s2"]),
        values=np.array([[1.0, 2.0, 10.0], [3.0, 4.0, 12.0]]),
        parameters=schema,
    )
    assert obs.n_samples == 2
    assert obs.n_parameters == 3
    np.testing.assert_array_equal(obs.column("Cr"), [2.0, 4.0])


def test_observation_set_rejects_mismatched_sample_id_length():
    schema = make_schema()
    with pytest.raises(SchemaError):
        ObservationSet(
            sample_ids=np.array(["s1", "s2", "s3"]),
            values=np.array([[1.0, 2.0, 10.0], [3.0, 4.0, 12.0]]),
            parameters=schema,
        )


def test_observation_set_rejects_mismatched_parameter_count():
    schema = make_schema()
    with pytest.raises(SchemaError):
        ObservationSet(
            sample_ids=np.array(["s1", "s2"]),
            values=np.array([[1.0, 2.0], [3.0, 4.0]]),
            parameters=schema,
        )


def test_observation_set_rejects_non_2d_values():
    schema = make_schema()
    with pytest.raises(SchemaError):
        ObservationSet(
            sample_ids=np.array(["s1"]),
            values=np.array([1.0, 2.0, 3.0]),
            parameters=schema,
        )


def test_observation_set_subject_cluster_ids_shape_checked():
    schema = make_schema()
    with pytest.raises(SchemaError):
        ObservationSet(
            sample_ids=np.array(["s1", "s2"]),
            values=np.array([[1.0, 2.0, 10.0], [3.0, 4.0, 12.0]]),
            parameters=schema,
            subject_ids=np.array(["subj1"]),
        )


def test_index_by_role_and_metabolite_finds_unique_column():
    schema = ParameterSchema(
        (
            ParameterSpec("NAA", ParameterRole.CONCENTRATION, "mM", metabolite="NAA"),
            ParameterSpec("NAA_linewidth", ParameterRole.LORENTZIAN_LINESHAPE, "Hz", metabolite="NAA"),
            ParameterSpec("Cr_linewidth", ParameterRole.LORENTZIAN_LINESHAPE, "Hz", metabolite="Cr"),
        )
    )
    assert schema.index_by_role_and_metabolite(ParameterRole.LORENTZIAN_LINESHAPE, "NAA") == 1
    assert schema.index_by_role_and_metabolite(ParameterRole.LORENTZIAN_LINESHAPE, "Cr") == 2


def test_index_by_role_and_metabolite_raises_when_not_exactly_one_match():
    schema = ParameterSchema(
        (
            ParameterSpec("NAA", ParameterRole.CONCENTRATION, "mM", metabolite="NAA"),
        )
    )
    with pytest.raises(SchemaError):
        schema.index_by_role_and_metabolite(ParameterRole.LORENTZIAN_LINESHAPE, "NAA")
    with pytest.raises(SchemaError):
        schema.index_by_role_and_metabolite(ParameterRole.CONCENTRATION, "Cr")


def test_column_for_metabolite_role_on_observation_set():
    schema = ParameterSchema(
        (
            ParameterSpec("NAA", ParameterRole.CONCENTRATION, "mM", metabolite="NAA"),
            ParameterSpec("NAA_linewidth", ParameterRole.LORENTZIAN_LINESHAPE, "Hz", metabolite="NAA"),
        )
    )
    obs = ObservationSet(
        sample_ids=np.array(["s1", "s2"]),
        values=np.array([[1.0, 10.0], [2.0, 12.0]]),
        parameters=schema,
    )
    np.testing.assert_array_equal(
        obs.column_for_metabolite_role(ParameterRole.LORENTZIAN_LINESHAPE, "NAA"), [10.0, 12.0]
    )
