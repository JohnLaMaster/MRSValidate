"""Semantic validation of :class:`~mrsvalidate.schema.ObservationSet` instances.

``schema.py`` enforces *structural* invariants (shapes, dtypes, unique
parameter names). This module enforces *semantic* invariants: whether a
dataset actually contains what an analysis needs (a required parameter
role, at least one sample, no duplicate sample IDs when the caller
requires strict identity). Validation never repairs data — it reports
problems so the caller (or :mod:`~mrsvalidate.alignment`) can decide the
documented policy to apply.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .exceptions import ValidationError
from .schema import ObservationSet, ParameterRole


@dataclass(frozen=True)
class ValidationReport:
    """Outcome of validating a single :class:`~mrsvalidate.schema.ObservationSet` in isolation.

    A ``ValidationReport`` never causes a failure by itself: it records
    what :func:`validate_observation_set` found (sample and parameter
    counts, duplicate sample IDs, free-text warnings) so that a
    downstream policy — alignment, missingness handling, or a specific
    analysis — can decide what to do about it.

    Attributes
    ----------
    n_samples : int
        Number of rows (samples) in the validated observation set.
    n_parameters : int
        Number of columns (fitted parameters) in the validated
        observation set.
    duplicate_sample_ids : tuple of str
        Sample IDs that occurred more than once, in first-seen order.
        Empty if there were no duplicates.
    warnings : tuple of str
        Human-readable descriptions of issues found (e.g. zero samples,
        duplicate IDs present), for logging or reporting; these do not
        by themselves raise an exception.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.schema import (
    ...     ObservationSet,
    ...     ParameterRole,
    ...     ParameterSchema,
    ...     ParameterSpec,
    ... )
    >>> from mrsvalidate.validation import validate_observation_set
    >>> schema = ParameterSchema(
    ...     (ParameterSpec("NAA", ParameterRole.CONCENTRATION, "mM"),)
    ... )
    >>> obs = ObservationSet(
    ...     sample_ids=np.array(["s1", "s1"]),
    ...     values=np.array([[1.0], [2.0]]),
    ...     parameters=schema,
    ... )
    >>> report = validate_observation_set(obs)
    >>> report.n_samples, report.has_duplicate_ids
    (2, True)
    """

    n_samples: int
    n_parameters: int
    duplicate_sample_ids: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def has_duplicate_ids(self) -> bool:
        """Whether any sample ID was duplicated.

        Returns
        -------
        bool
            ``True`` if ``duplicate_sample_ids`` is non-empty, ``False``
            otherwise.
        """
        return len(self.duplicate_sample_ids) > 0


def find_duplicate_ids(sample_ids: np.ndarray) -> tuple[str, ...]:
    """Find sample IDs that occur more than once.

    Parameters
    ----------
    sample_ids : ndarray of str
        One-dimensional array of sample identifiers, possibly
        containing repeats.

    Returns
    -------
    tuple of str
        The distinct sample IDs that occur more than once in
        ``sample_ids``, in first-seen order. Empty if every ID is
        unique.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.validation import find_duplicate_ids
    >>> find_duplicate_ids(np.array(["s1", "s2", "s1", "s3", "s2"]))
    ('s1', 's2')
    >>> find_duplicate_ids(np.array(["s1", "s2"]))
    ()
    """

    ids = np.asarray(sample_ids, dtype=str)
    seen: dict[str, int] = {}
    for sid in ids.tolist():
        seen[sid] = seen.get(sid, 0) + 1
    return tuple(sid for sid, count in seen.items() if count > 1)


def validate_observation_set(obs: ObservationSet) -> ValidationReport:
    """Validate an observation set in isolation, reporting rather than raising.

    Parameters
    ----------
    obs : ObservationSet
        The observation set to check for zero samples and duplicate
        sample IDs.

    Returns
    -------
    ValidationReport
        A report describing ``obs``'s sample/parameter counts, any
        duplicate sample IDs found, and human-readable warnings. Issues
        found here are reported, never raised — resolving them (e.g. by
        alignment or missingness handling) is the caller's
        responsibility.

    Notes
    -----
    This function never raises for the issues it detects; it is the
    caller — typically :mod:`~mrsvalidate.alignment` or
    :mod:`~mrsvalidate.missingness` — that decides the documented policy
    to apply to a zero-sample or duplicate-ID observation set.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.schema import (
    ...     ObservationSet,
    ...     ParameterRole,
    ...     ParameterSchema,
    ...     ParameterSpec,
    ... )
    >>> from mrsvalidate.validation import validate_observation_set
    >>> schema = ParameterSchema(
    ...     (ParameterSpec("NAA", ParameterRole.CONCENTRATION, "mM"),)
    ... )
    >>> obs = ObservationSet(
    ...     sample_ids=np.array(["s1", "s2"]),
    ...     values=np.array([[1.0], [2.0]]),
    ...     parameters=schema,
    ... )
    >>> report = validate_observation_set(obs)
    >>> report.warnings
    ()
    >>> report.n_samples, report.n_parameters
    (2, 1)
    """

    warnings: list[str] = []
    if obs.n_samples == 0:
        warnings.append("ObservationSet contains zero samples")

    duplicates = find_duplicate_ids(obs.sample_ids)
    if duplicates:
        warnings.append(
            f"ObservationSet contains {len(duplicates)} duplicate sample ID(s)"
        )

    return ValidationReport(
        n_samples=obs.n_samples,
        n_parameters=obs.n_parameters,
        duplicate_sample_ids=duplicates,
        warnings=tuple(warnings),
    )


def require_role(obs: ObservationSet, role: ParameterRole) -> tuple[int, ...]:
    """Return the column indices for a required parameter role, or raise.

    Used by analyses that require a specific parameter category (e.g.
    SNR calculations requiring ``ParameterRole.SNR``) instead of
    guessing column position from count.

    Parameters
    ----------
    obs : ObservationSet
        The observation set whose schema is searched for ``role``.
    role : ParameterRole
        The parameter role that must be present, e.g.
        ``ParameterRole.SNR``.

    Returns
    -------
    tuple of int
        The zero-based column indices of every parameter in
        ``obs.parameters`` whose role equals ``role``.

    Raises
    ------
    ValidationError
        If no column in ``obs.parameters`` has the requested ``role``.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.schema import (
    ...     ObservationSet,
    ...     ParameterRole,
    ...     ParameterSchema,
    ...     ParameterSpec,
    ... )
    >>> from mrsvalidate.validation import require_role
    >>> schema = ParameterSchema(
    ...     (ParameterSpec("snr_NAA", ParameterRole.SNR, None),)
    ... )
    >>> obs = ObservationSet(
    ...     sample_ids=np.array(["s1", "s2"]),
    ...     values=np.array([[10.0], [12.0]]),
    ...     parameters=schema,
    ... )
    >>> require_role(obs, ParameterRole.SNR)
    (0,)
    """

    indices = obs.parameters.indices_by_role(role)
    if not indices:
        raise ValidationError(
            f"ObservationSet has no parameter with role {role.value!r}; "
            "column order/count must not be assumed to imply it"
        )
    return indices


def require_non_empty(obs: ObservationSet) -> None:
    """Raise if an observation set contains zero samples.

    Parameters
    ----------
    obs : ObservationSet
        The observation set to check.

    Returns
    -------
    None
        Returns nothing when ``obs`` contains at least one sample.

    Raises
    ------
    ValidationError
        If ``obs.n_samples == 0``.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.schema import (
    ...     ObservationSet,
    ...     ParameterRole,
    ...     ParameterSchema,
    ...     ParameterSpec,
    ... )
    >>> from mrsvalidate.validation import require_non_empty
    >>> schema = ParameterSchema(
    ...     (ParameterSpec("NAA", ParameterRole.CONCENTRATION, "mM"),)
    ... )
    >>> obs = ObservationSet(
    ...     sample_ids=np.array(["s1"]), values=np.array([[1.0]]), parameters=schema
    ... )
    >>> require_non_empty(obs)
    """
    if obs.n_samples == 0:
        raise ValidationError("ObservationSet contains zero samples")
