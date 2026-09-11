"""Central alignment routine every comparison must pass through.

Row position is never a valid pairing contract by itself. Every
comparison between two :class:`~mrsvalidate.schema.ObservationSet`
instances — paired analyses in particular — must go through
:func:`align_observations`, which pairs by explicit sample ID,
deterministically reorders the result, and reports exactly what was
matched, unmatched, duplicated, or excluded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .exceptions import AlignmentError
from .schema import ObservationSet
from .validation import find_duplicate_ids

PairingMode = Literal["paired", "independent"]
DuplicatePolicy = Literal["error", "first", "aggregate"]


@dataclass(frozen=True)
class AlignmentReport:
    """Full audit trail of what :func:`align_observations` did.

    Every field needed to explain, after the fact, exactly which
    samples were matched, which were unmatched, which were duplicated,
    and whether the rows were reordered relative to the caller's input.

    Attributes
    ----------
    matched_ids : tuple of str
        Sample IDs present in both inputs after deduplication, sorted
        lexicographically. Empty when ``pairing_status ==
        "independent"``, since identity-matching does not apply to
        that design.
    only_in_reference : tuple of str
        Sample IDs present only in the (deduplicated) reference set.
        Empty when ``pairing_status == "independent"``.
    only_in_candidate : tuple of str
        Sample IDs present only in the (deduplicated) candidate set.
        Empty when ``pairing_status == "independent"``.
    duplicate_ids_reference : tuple of str
        Sample IDs that were duplicated in the original reference
        input, before deduplication.
    duplicate_ids_candidate : tuple of str
        Sample IDs that were duplicated in the original candidate
        input, before deduplication.
    excluded_ids : tuple of str
        All sample IDs excluded from the final paired comparison
        (the union of ``only_in_reference`` and ``only_in_candidate``,
        sorted). Empty when ``pairing_status == "independent"``.
    exclusion_reasons : dict[str, int]
        Count of excluded IDs per reason (``"only_in_reference"``,
        ``"only_in_candidate"``), omitting reasons with zero occurrences.
    n_reference_input : int
        Number of samples in the reference set as originally supplied,
        before deduplication or alignment.
    n_candidate_input : int
        Number of samples in the candidate set as originally supplied,
        before deduplication or alignment.
    n_final : int
        For ``pairing_status == "paired"``, the number of matched
        pairs. For ``pairing_status == "independent"``, the combined
        size of the two independent, deduplicated samples.
    pairing_status : {"paired", "independent"}
        Which pairing mode was used to produce this report.
    reorder_occurred : bool
        Whether the final row order differs from the reference input's
        original row order. Always ``False`` for
        ``pairing_status == "independent"``.

    Notes
    -----
    For ``pairing="independent"``, ID-matching fields are not
    applicable (the two samples are not paired by identity) and are
    left empty; ``n_final`` is documented per-mode above rather than
    silently overloaded with an ambiguous meaning.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.alignment import align_observations
    >>> from mrsvalidate.schema import (
    ...     ObservationSet,
    ...     ParameterRole,
    ...     ParameterSchema,
    ...     ParameterSpec,
    ... )
    >>> schema = ParameterSchema(
    ...     (ParameterSpec("NAA", ParameterRole.CONCENTRATION, "mM"),)
    ... )
    >>> reference = ObservationSet(
    ...     sample_ids=np.array(["s1", "s2"]),
    ...     values=np.array([[1.0], [2.0]]),
    ...     parameters=schema,
    ... )
    >>> candidate = ObservationSet(
    ...     sample_ids=np.array(["s2", "s3"]),
    ...     values=np.array([[2.1], [3.0]]),
    ...     parameters=schema,
    ... )
    >>> comparison = align_observations(reference, candidate, pairing="paired")
    >>> comparison.report.matched_ids
    ('s2',)
    >>> comparison.report.only_in_reference
    ('s1',)
    >>> comparison.report.only_in_candidate
    ('s3',)
    """

    matched_ids: tuple[str, ...]
    only_in_reference: tuple[str, ...]
    only_in_candidate: tuple[str, ...]
    duplicate_ids_reference: tuple[str, ...]
    duplicate_ids_candidate: tuple[str, ...]
    excluded_ids: tuple[str, ...]
    exclusion_reasons: dict[str, int]
    n_reference_input: int
    n_candidate_input: int
    n_final: int
    pairing_status: PairingMode
    reorder_occurred: bool


@dataclass(frozen=True)
class AlignedComparison:
    """The two observation sets after alignment, plus the report explaining how.

    Attributes
    ----------
    reference : ObservationSet
        The reference observation set after duplicate resolution and
        (for ``pairing="paired"``) subsetting/reordering to the matched
        sample IDs.
    candidate : ObservationSet
        The candidate observation set after the same processing applied
        to ``reference``.
    report : AlignmentReport
        Full audit trail describing what matching, exclusion, and
        reordering was performed to produce ``reference`` and
        ``candidate``.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.alignment import align_observations
    >>> from mrsvalidate.schema import (
    ...     ObservationSet,
    ...     ParameterRole,
    ...     ParameterSchema,
    ...     ParameterSpec,
    ... )
    >>> schema = ParameterSchema(
    ...     (ParameterSpec("NAA", ParameterRole.CONCENTRATION, "mM"),)
    ... )
    >>> reference = ObservationSet(
    ...     sample_ids=np.array(["s1", "s2"]),
    ...     values=np.array([[1.0], [2.0]]),
    ...     parameters=schema,
    ... )
    >>> candidate = ObservationSet(
    ...     sample_ids=np.array(["s1", "s2"]),
    ...     values=np.array([[1.1], [2.1]]),
    ...     parameters=schema,
    ... )
    >>> comparison = align_observations(reference, candidate, pairing="paired")
    >>> comparison.reference.sample_ids.tolist()
    ['s1', 's2']
    >>> comparison.candidate.column("NAA")
    array([1.1, 2.1])
    """

    reference: ObservationSet
    candidate: ObservationSet
    report: AlignmentReport


def _resolve_duplicates(
    obs: ObservationSet, policy: DuplicatePolicy, *, label: str
) -> tuple[ObservationSet, tuple[str, ...]]:
    """Apply a duplicate-handling policy to one observation set.

    Parameters
    ----------
    obs : ObservationSet
        The observation set to check and, if needed, deduplicate.
    policy : {"error", "first", "aggregate"}
        How to resolve duplicate sample IDs: raise, keep the first
        occurrence, or aggregate duplicate rows by their elementwise
        mean.
    label : str
        Human-readable label for ``obs`` (e.g. ``"reference"`` or
        ``"candidate"``), used only to build a clear error message.

    Returns
    -------
    tuple[ObservationSet, tuple[str, ...]]
        The (possibly deduplicated) ``ObservationSet`` — unchanged if
        there were no duplicates — and the tuple of sample IDs that
        were duplicated in the original input.

    Raises
    ------
    AlignmentError
        If ``obs`` contains duplicate sample IDs and ``policy ==
        "error"``.

    Notes
    -----
    For ``policy="aggregate"``, only ``values`` is meaningfully
    averaged across duplicate rows for the same ``sample_id``;
    ``subject_id``/``cluster_id``/``metadata`` are not averaged, so the
    first occurrence's is retained.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.alignment import _resolve_duplicates
    >>> from mrsvalidate.schema import (
    ...     ObservationSet,
    ...     ParameterRole,
    ...     ParameterSchema,
    ...     ParameterSpec,
    ... )
    >>> schema = ParameterSchema(
    ...     (ParameterSpec("NAA", ParameterRole.CONCENTRATION, "mM"),)
    ... )
    >>> obs = ObservationSet(
    ...     sample_ids=np.array(["s1", "s1", "s2"]),
    ...     values=np.array([[1.0], [3.0], [2.0]]),
    ...     parameters=schema,
    ... )
    >>> deduped, dup_ids = _resolve_duplicates(obs, "aggregate", label="reference")
    >>> dup_ids
    ('s1',)
    >>> deduped.column("NAA")
    array([2., 2.])
    """

    ids = obs.sample_ids
    dup_ids = find_duplicate_ids(ids)
    if not dup_ids:
        return obs, ()

    if policy == "error":
        raise AlignmentError(
            f"{label} contains duplicate sample IDs {dup_ids} and "
            'duplicate_policy="error" (the default). Pass an explicit '
            'duplicate_policy ("first" or "aggregate") to resolve this.'
        )

    order: list[str] = []
    groups: dict[str, list[int]] = {}
    for i, sid in enumerate(ids.tolist()):
        if sid not in groups:
            groups[sid] = []
            order.append(sid)
        groups[sid].append(i)

    if policy == "first":
        keep_idx = [groups[sid][0] for sid in order]
        new_values = obs.values[keep_idx]
    elif policy == "aggregate":
        # Documented, explicit choice: aggregate duplicate rows for the
        # same sample_id by taking the elementwise mean of `values`.
        # subject_id/cluster_id/metadata are not meaningfully averaged,
        # so the first occurrence's is retained.
        new_values = np.vstack(
            [obs.values[groups[sid]].mean(axis=0) for sid in order]
        )
        keep_idx = [groups[sid][0] for sid in order]
    else:
        raise ValueError(f"Unknown duplicate_policy: {policy!r}")

    new_ids = np.asarray(order, dtype=str)
    new_subject = obs.subject_ids[keep_idx] if obs.subject_ids is not None else None
    new_cluster = obs.cluster_ids[keep_idx] if obs.cluster_ids is not None else None

    deduped = ObservationSet(
        sample_ids=new_ids,
        values=new_values,
        parameters=obs.parameters,
        subject_ids=new_subject,
        cluster_ids=new_cluster,
        model_id=obs.model_id,
        metadata=obs.metadata,
    )
    return deduped, dup_ids


def _subset_in_order(obs: ObservationSet, id_order: tuple[str, ...]) -> ObservationSet:
    """Subset and reorder an observation set to an explicit sequence of sample IDs.

    Parameters
    ----------
    obs : ObservationSet
        The observation set to subset. Every ID in ``id_order`` must be
        present in ``obs.sample_ids``.
    id_order : tuple of str
        The exact sample IDs to keep, in the order they should appear
        in the result.

    Returns
    -------
    ObservationSet
        A new ``ObservationSet`` containing exactly the rows named in
        ``id_order``, reordered to match it.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.alignment import _subset_in_order
    >>> from mrsvalidate.schema import (
    ...     ObservationSet,
    ...     ParameterRole,
    ...     ParameterSchema,
    ...     ParameterSpec,
    ... )
    >>> schema = ParameterSchema(
    ...     (ParameterSpec("NAA", ParameterRole.CONCENTRATION, "mM"),)
    ... )
    >>> obs = ObservationSet(
    ...     sample_ids=np.array(["s1", "s2", "s3"]),
    ...     values=np.array([[1.0], [2.0], [3.0]]),
    ...     parameters=schema,
    ... )
    >>> subset = _subset_in_order(obs, ("s3", "s1"))
    >>> subset.sample_ids.tolist()
    ['s3', 's1']
    >>> subset.column("NAA")
    array([3., 1.])
    """
    id_to_idx = {sid: i for i, sid in enumerate(obs.sample_ids.tolist())}
    idx = [id_to_idx[sid] for sid in id_order]
    return ObservationSet(
        sample_ids=np.asarray(id_order, dtype=str),
        values=obs.values[idx],
        parameters=obs.parameters,
        subject_ids=obs.subject_ids[idx] if obs.subject_ids is not None else None,
        cluster_ids=obs.cluster_ids[idx] if obs.cluster_ids is not None else None,
        model_id=obs.model_id,
        metadata=obs.metadata,
    )


def align_observations(
    reference: ObservationSet,
    candidate: ObservationSet,
    *,
    pairing: PairingMode,
    duplicate_policy: DuplicatePolicy = "error",
) -> AlignedComparison:
    """Align two observation sets by sample ID for a scientific comparison.

    Parameters
    ----------
    reference : ObservationSet
        The reference (e.g. ground-truth) observation set.
    candidate : ObservationSet
        The candidate (e.g. predicted) observation set to compare
        against ``reference``.
    pairing : {"paired", "independent"}
        Comparison design to use.

        ``"paired"``
            Rows are matched by intersection of ``sample_ids``, then
            deterministically reordered (lexicographically by ID) so
            that identical inputs always align identically regardless
            of input row order.

        ``"independent"``
            The two sets are treated as independent samples (e.g. for
            the manuscript's independent two-sample distributional
            tests). No ID intersection or reordering is performed;
            duplicates are still resolved per ``duplicate_policy``
            within each set independently.
    duplicate_policy : {"error", "first", "aggregate"}, optional
        How to resolve duplicate sample IDs found within either input.
        Default is ``"error"``.

    Returns
    -------
    AlignedComparison
        The aligned ``reference`` and ``candidate`` observation sets
        together with an :class:`AlignmentReport` describing exactly
        what was matched, excluded, duplicated, and reordered. For
        ``pairing="paired"``, ``report.n_final`` is the number of
        matched pairs. For ``pairing="independent"``, ``report.n_final``
        is the combined size of the two independent (deduplicated)
        samples, and all ID-matching report fields (``matched_ids``,
        ``only_in_reference``, ``only_in_candidate``) are empty because
        identity-matching does not apply to this design.

    Raises
    ------
    AlignmentError
        If duplicate sample IDs are present within either input and
        ``duplicate_policy == "error"`` (the default).
    ValueError
        If ``pairing`` is not one of ``"paired"`` or ``"independent"``.

    Notes
    -----
    Row position is never a valid pairing contract by itself; this
    function is the only sanctioned way to compare two
    :class:`~mrsvalidate.schema.ObservationSet` instances that must be
    paired by identity.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.alignment import align_observations
    >>> from mrsvalidate.schema import (
    ...     ObservationSet,
    ...     ParameterRole,
    ...     ParameterSchema,
    ...     ParameterSpec,
    ... )
    >>> schema = ParameterSchema(
    ...     (ParameterSpec("NAA", ParameterRole.CONCENTRATION, "mM"),)
    ... )
    >>> reference = ObservationSet(
    ...     sample_ids=np.array(["s2", "s1"]),
    ...     values=np.array([[2.0], [1.0]]),
    ...     parameters=schema,
    ... )
    >>> candidate = ObservationSet(
    ...     sample_ids=np.array(["s1", "s2"]),
    ...     values=np.array([[1.1], [2.1]]),
    ...     parameters=schema,
    ... )
    >>> comparison = align_observations(reference, candidate, pairing="paired")
    >>> comparison.reference.sample_ids.tolist()
    ['s1', 's2']
    >>> comparison.candidate.sample_ids.tolist()
    ['s1', 's2']
    >>> comparison.report.n_final
    2
    >>> comparison.report.reorder_occurred
    True
    """

    n_reference_input = reference.n_samples
    n_candidate_input = candidate.n_samples

    ref_deduped, dup_ref = _resolve_duplicates(
        reference, duplicate_policy, label="reference"
    )
    cand_deduped, dup_cand = _resolve_duplicates(
        candidate, duplicate_policy, label="candidate"
    )

    if pairing == "independent":
        report = AlignmentReport(
            matched_ids=(),
            only_in_reference=(),
            only_in_candidate=(),
            duplicate_ids_reference=dup_ref,
            duplicate_ids_candidate=dup_cand,
            excluded_ids=(),
            exclusion_reasons={},
            n_reference_input=n_reference_input,
            n_candidate_input=n_candidate_input,
            n_final=ref_deduped.n_samples + cand_deduped.n_samples,
            pairing_status="independent",
            reorder_occurred=False,
        )
        return AlignedComparison(
            reference=ref_deduped, candidate=cand_deduped, report=report
        )

    if pairing != "paired":
        raise ValueError(f"Unknown pairing mode: {pairing!r}")

    ref_ids = set(ref_deduped.sample_ids.tolist())
    cand_ids = set(cand_deduped.sample_ids.tolist())
    matched_ids = tuple(sorted(ref_ids & cand_ids))
    only_in_reference = tuple(sorted(ref_ids - cand_ids))
    only_in_candidate = tuple(sorted(cand_ids - ref_ids))

    original_ref_order = tuple(
        sid for sid in ref_deduped.sample_ids.tolist() if sid in ref_ids & cand_ids
    )
    reorder_occurred = original_ref_order != matched_ids

    exclusion_reasons: dict[str, int] = {}
    if only_in_reference:
        exclusion_reasons["only_in_reference"] = len(only_in_reference)
    if only_in_candidate:
        exclusion_reasons["only_in_candidate"] = len(only_in_candidate)
    excluded_ids = tuple(sorted(only_in_reference + only_in_candidate))

    aligned_reference = _subset_in_order(ref_deduped, matched_ids)
    aligned_candidate = _subset_in_order(cand_deduped, matched_ids)

    report = AlignmentReport(
        matched_ids=matched_ids,
        only_in_reference=only_in_reference,
        only_in_candidate=only_in_candidate,
        duplicate_ids_reference=dup_ref,
        duplicate_ids_candidate=dup_cand,
        excluded_ids=excluded_ids,
        exclusion_reasons=exclusion_reasons,
        n_reference_input=n_reference_input,
        n_candidate_input=n_candidate_input,
        n_final=len(matched_ids),
        pairing_status="paired",
        reorder_occurred=reorder_occurred,
    )
    return AlignedComparison(
        reference=aligned_reference, candidate=aligned_candidate, report=report
    )
