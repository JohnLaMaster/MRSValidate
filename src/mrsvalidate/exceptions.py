"""Exception hierarchy for mrsvalidate.

All exceptions raised deliberately by this package derive from
:class:`MRSDLEvalError` so callers can catch package errors without
catching unrelated ``ValueError``/``TypeError`` instances raised by
dependencies.
"""

from __future__ import annotations


class MRSDLEvalError(Exception):
    """Base class for all exceptions raised deliberately by mrsvalidate.

    Every exception that this package raises on purpose — as opposed to
    exceptions propagating up from a dependency such as NumPy or SciPy —
    is a subclass of this type. Catching ``MRSDLEvalError`` therefore
    catches exactly the package's own, documented failure modes.

    Notes
    -----
    Catch this type (rather than a bare ``Exception`` or a builtin such
    as ``ValueError``) when you want to distinguish a deliberate,
    documented mrsvalidate failure from an unrelated bug or a dependency
    error.

    Examples
    --------
    >>> from mrsvalidate.exceptions import MRSDLEvalError, SchemaError
    >>> issubclass(SchemaError, MRSDLEvalError)
    True
    """


class SchemaError(MRSDLEvalError):
    """Raised when a parameter schema or observation set is malformed.

    This covers violations of the *structural* invariants of
    :class:`~mrsvalidate.schema.ParameterSchema` and
    :class:`~mrsvalidate.schema.ObservationSet` — for example a shape
    mismatch between ``values`` and ``sample_ids``, a duplicate
    parameter name in a schema, or a lookup for a parameter or
    metabolite that does not exist.

    Notes
    -----
    ``SchemaError`` is about the *shape and bookkeeping* of a dataset,
    not whether the dataset is scientifically adequate for a given
    analysis; the latter is signalled by
    :class:`~mrsvalidate.exceptions.ValidationError` instead.

    Examples
    --------
    >>> from mrsvalidate.exceptions import SchemaError
    >>> try:
    ...     raise SchemaError("duplicate parameter name: 'NAA'")
    ... except SchemaError as exc:
    ...     print(exc)
    duplicate parameter name: 'NAA'
    """


class ValidationError(MRSDLEvalError):
    """Raised when an observation set fails semantic validation.

    This is raised, for example, when an
    :class:`~mrsvalidate.schema.ObservationSet` is structurally valid
    but does not satisfy a semantic precondition that a specific
    analysis requires — such as a required
    :class:`~mrsvalidate.schema.ParameterRole` being entirely absent
    from its schema.

    Notes
    -----
    Structural problems (shape mismatches, duplicate names) raise
    :class:`SchemaError` instead; ``ValidationError`` is reserved for
    semantic preconditions checked by
    :mod:`mrsvalidate.validation`.

    Examples
    --------
    >>> from mrsvalidate.exceptions import ValidationError
    >>> try:
    ...     raise ValidationError("no parameter with role 'snr'")
    ... except ValidationError as exc:
    ...     print(exc)
    no parameter with role 'snr'
    """


class AlignmentError(MRSDLEvalError):
    """Raised when two observation sets cannot be safely aligned.

    :func:`~mrsvalidate.alignment.align_observations` raises this when
    duplicate sample IDs are present within an input and
    ``duplicate_policy="error"`` (the default), or when no valid
    pairing between the reference and candidate sets can be formed.

    Notes
    -----
    This error exists so that ambiguous row identity is never silently
    resolved by falling back to row position; the caller must
    explicitly choose a ``duplicate_policy`` (``"first"`` or
    ``"aggregate"``) to proceed.

    Examples
    --------
    >>> from mrsvalidate.exceptions import AlignmentError
    >>> try:
    ...     raise AlignmentError("duplicate sample IDs in reference: ('s1',)")
    ... except AlignmentError as exc:
    ...     print(exc)
    duplicate sample IDs in reference: ('s1',)
    """


class InvalidUncertaintyRepresentationError(MRSDLEvalError):
    """Raised when an uncertainty array lacks an explicit representation.

    This is raised when an uncertainty or CRLB (Cramer-Rao lower bound)
    array is supplied to an analysis without an explicit, supported
    ``UncertaintyRepresentation`` describing whether the values are,
    for example, standard deviations or variances.

    Notes
    -----
    The library never guesses whether a supplied uncertainty array
    holds standard deviations or variances; the caller must state this
    explicitly, and this exception is raised when that statement is
    missing or unsupported.

    Examples
    --------
    >>> from mrsvalidate.exceptions import InvalidUncertaintyRepresentationError
    >>> try:
    ...     raise InvalidUncertaintyRepresentationError(
    ...         "uncertainty representation must be given explicitly"
    ...     )
    ... except InvalidUncertaintyRepresentationError as exc:
    ...     print(exc)
    uncertainty representation must be given explicitly
    """


class SingularCovarianceError(MRSDLEvalError):
    """Raised when a required covariance matrix is singular.

    This is raised when a covariance matrix needed for a
    standardized-error or Mahalanobis-distance calculation is singular
    (or numerically indistinguishable from singular) and strict mode is
    in effect, so that the caller is never silently handed a result
    computed from a pseudo-inverse without being told.

    Notes
    -----
    Callers that want a best-effort result in the presence of a
    singular covariance matrix must opt out of strict mode explicitly,
    rather than relying on an implicit fallback.

    Examples
    --------
    >>> from mrsvalidate.exceptions import SingularCovarianceError
    >>> try:
    ...     raise SingularCovarianceError("covariance matrix is singular")
    ... except SingularCovarianceError as exc:
    ...     print(exc)
    covariance matrix is singular
    """
