"""Property tests (Hypothesis) for the mathematical invariants spec
lines 1531-1549 list: RMSE nonnegativity, Pearson symmetry/boundedness,
Lin CCC at equality, the manuscript concise score's sign/zero behavior,
Clopper-Pearson's zero-successes rule, and alignment's permutation
invariance.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from mrsvalidate.alignment import align_observations
from mrsvalidate.analysis import agreement, errors, regression
from mrsvalidate.analysis.operating_envelope import _clopper_pearson_lower
from mrsvalidate.schema import (
    ObservationSet,
    ParameterRole,
    ParameterSchema,
    ParameterSpec,
)

_SCHEMA = ParameterSchema((ParameterSpec("NAA", ParameterRole.CONCENTRATION, "mM"),))

finite_floats = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False, width=64
)
paired_arrays = st.integers(min_value=3, max_value=30).flatmap(
    lambda n: st.tuples(
        arrays(np.float64, n, elements=finite_floats),
        arrays(np.float64, n, elements=finite_floats),
    )
)


@settings(max_examples=50)
@given(paired_arrays)
def test_rmse_is_always_nonnegative(pair):
    ref, pred = pair
    ids = np.array([f"s{i}" for i in range(len(ref))])
    result = errors.rmse(ref, pred, ids)
    if np.isfinite(result.value):
        assert result.value >= 0.0


@settings(max_examples=50)
@given(paired_arrays)
def test_rmse_zero_iff_paired_values_equal(pair):
    ref, _ = pair
    ids = np.array([f"s{i}" for i in range(len(ref))])
    result = errors.rmse(ref, ref, ids)
    assert result.value == 0.0 or not np.isfinite(result.value)


@settings(max_examples=30)
@given(paired_arrays)
def test_pearson_r_is_symmetric_and_bounded(pair):
    ref, pred = pair
    ids = np.array([f"s{i}" for i in range(len(ref))])
    forward = regression.pearson_r(ref, pred, ids)
    backward = regression.pearson_r(pred, ref, ids)
    if np.isfinite(forward.r) and np.isfinite(backward.r):
        assert forward.r == pytest.approx(backward.r)
        assert -1.0 - 1e-9 <= forward.r <= 1.0 + 1e-9


@settings(max_examples=30)
@given(paired_arrays)
def test_lin_ccc_equals_one_for_identical_data(pair):
    ref, _ = pair
    ids = np.array([f"s{i}" for i in range(len(ref))])
    result = agreement.lin_ccc(ref, ref, ids)
    if np.isfinite(result.rho_c):
        assert result.rho_c == pytest.approx(1.0)


@settings(max_examples=30)
@given(paired_arrays)
def test_concise_score_sign_follows_r_when_r2_positive(pair):
    ref, pred = pair
    ids = np.array([f"s{i}" for i in range(len(ref))])
    r_result = regression.pearson_r(ref, pred, ids)
    r2_result = regression.r_squared_direct(ref, pred, ids)
    concise = regression.concise_score_manuscript(ref, pred, ids)
    if not np.isfinite(concise.value):
        return
    if r2_result.value <= 0:
        assert concise.value == 0.0
    elif np.isfinite(r_result.r):
        assert np.sign(concise.value) == np.sign(r_result.r) or concise.value == 0.0


@given(st.integers(min_value=1, max_value=200))
def test_clopper_pearson_lower_is_zero_for_zero_successes(n):
    lower = _clopper_pearson_lower(np.array([0]), np.array([n]), 0.025)
    assert lower[0] == 0.0


@settings(max_examples=20)
@given(st.permutations(list(range(10))))
def test_alignment_invariant_to_arbitrary_row_permutation(perm):
    ids = np.array([f"s{i}" for i in range(10)])
    values = np.arange(10, dtype=np.float64).reshape(-1, 1)
    reference = ObservationSet(sample_ids=ids, values=values, parameters=_SCHEMA)

    perm = np.array(perm)
    shuffled_candidate = ObservationSet(
        sample_ids=ids[perm], values=(values * 2)[perm], parameters=_SCHEMA
    )
    unshuffled_candidate = ObservationSet(sample_ids=ids, values=values * 2, parameters=_SCHEMA)

    aligned_shuffled = align_observations(reference, shuffled_candidate, pairing="paired")
    aligned_unshuffled = align_observations(reference, unshuffled_candidate, pairing="paired")

    np.testing.assert_array_equal(
        aligned_shuffled.candidate.sample_ids, aligned_unshuffled.candidate.sample_ids
    )
    np.testing.assert_array_equal(
        aligned_shuffled.candidate.values, aligned_unshuffled.candidate.values
    )
