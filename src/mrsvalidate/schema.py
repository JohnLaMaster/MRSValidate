"""Canonical parameter and observation schema.

The principal fitted-parameter matrix is ``values.shape == (n_samples,
n_fit_parameters)``. Column order is never assumed from column count or
from any fixed manuscript convention — every column is described by an
explicit, ordered :class:`ParameterSpec` in a :class:`ParameterSchema`.
Row position is never a valid pairing contract by itself; stable sample
identity (:class:`ObservationSet.sample_ids`) is required for every
comparison and alignment is always performed by
:func:`mrsvalidate.alignment.align_observations`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .exceptions import SchemaError


class ParameterRole(str, Enum):
    """Semantic role of a column in the fitted-parameter matrix.

    Every column of a fitted-parameter matrix is tagged with exactly
    one ``ParameterRole`` so that analyses can look up columns by
    meaning (e.g. "give me the SNR column") instead of by an assumed,
    fragile position.

    Attributes
    ----------
    CONCENTRATION : str
        A metabolite concentration estimate.
    LORENTZIAN_LINESHAPE : str
        A Lorentzian line-shape parameter (e.g. linewidth).
    GAUSSIAN_LINESHAPE : str
        A Gaussian line-shape parameter.
    FREQUENCY_SHIFT : str
        A frequency-shift parameter.
    SNR : str
        A signal-to-noise ratio estimate.
    PHASE_ZERO_ORDER : str
        A zero-order phase parameter.
    PHASE_FIRST_ORDER : str
        A first-order phase parameter.
    BASELINE_SPLINE : str
        A baseline spline coefficient.
    OTHER : str
        Any column supplied by a dataset that does not map onto one of
        the categories above. A dataset is never required to contain
        every role.

    Notes
    -----
    This enumerates the column categories the supplied manuscript uses
    (spec: metabolite concentrations, Lorentzian/Gaussian line-shape
    parameters, frequency shifts, SNR, zero/first-order phase, optional
    baseline spline coefficients) plus ``OTHER`` for anything a dataset
    supplies that does not map onto those categories.

    Examples
    --------
    >>> from mrsvalidate.schema import ParameterRole
    >>> ParameterRole.CONCENTRATION.value
    'concentration'
    >>> ParameterRole.SNR == "snr"
    True
    """

    CONCENTRATION = "concentration"
    LORENTZIAN_LINESHAPE = "lorentzian_lineshape"
    GAUSSIAN_LINESHAPE = "gaussian_lineshape"
    FREQUENCY_SHIFT = "frequency_shift"
    SNR = "snr"
    PHASE_ZERO_ORDER = "phase_zero_order"
    PHASE_FIRST_ORDER = "phase_first_order"
    BASELINE_SPLINE = "baseline_spline"
    OTHER = "other"


@dataclass(frozen=True)
class ParameterSpec:
    """Metadata describing a single column of a fitted-parameter matrix.

    One ``ParameterSpec`` fully describes one column: its name, its
    semantic :class:`ParameterRole`, its physical unit, and (where
    applicable) which basis component and metabolite it belongs to.

    Attributes
    ----------
    name : str
        Unique, non-empty name of the column, e.g. ``"NAA"`` or
        ``"linewidth"``.
    role : ParameterRole
        Semantic role of the column; see :class:`ParameterRole`.
    unit : str or None
        Physical unit of the column's values (e.g. ``"mM"``, ``"Hz"``),
        or ``None`` if the column is dimensionless or unit-less.
    basis_name : str or None, optional
        Name of the basis-set signal this column corresponds to, if
        applicable. Default is ``None``.
    metabolite : str or None, optional
        Name of the metabolite this column pertains to, if applicable
        (e.g. ``"NAA"`` for a concentration or linewidth column tied to
        NAA). Default is ``None``.
    component_index : int or None, optional
        Index of the component within a multi-component parameterization
        (e.g. which Lorentzian component of a multi-component
        line shape), if applicable. Default is ``None``.

    Raises
    ------
    SchemaError
        If ``name`` is empty, or if ``role`` is not a
        :class:`ParameterRole`.

    Examples
    --------
    >>> from mrsvalidate.schema import ParameterRole, ParameterSpec
    >>> spec = ParameterSpec("NAA", ParameterRole.CONCENTRATION, "mM", metabolite="NAA")
    >>> spec.name, spec.unit
    ('NAA', 'mM')
    """

    name: str
    role: ParameterRole
    unit: str | None
    basis_name: str | None = None
    metabolite: str | None = None
    component_index: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise SchemaError("ParameterSpec.name must be a non-empty string")
        if not isinstance(self.role, ParameterRole):
            raise SchemaError(
                f"ParameterSpec.role must be a ParameterRole, got {self.role!r}"
            )


@dataclass(frozen=True)
class ParameterSchema:
    """Ordered, authoritative description of every column in ``values``.

    A ``ParameterSchema`` is the single source of truth for what each
    column of an :class:`ObservationSet`'s ``values`` matrix means.
    Column order in the underlying array must match ``parameters``
    order exactly; nothing in this library infers meaning from position
    alone.

    Attributes
    ----------
    parameters : tuple of ParameterSpec
        Ordered specifications, one per column of the associated
        ``values`` matrix, in the exact order those columns appear.

    Raises
    ------
    SchemaError
        If ``parameters`` contains two or more entries with the same
        ``name``.

    Examples
    --------
    >>> from mrsvalidate.schema import ParameterRole, ParameterSchema, ParameterSpec
    >>> schema = ParameterSchema(
    ...     (
    ...         ParameterSpec("NAA", ParameterRole.CONCENTRATION, "mM", metabolite="NAA"),
    ...         ParameterSpec("Cr", ParameterRole.CONCENTRATION, "mM", metabolite="Cr"),
    ...     )
    ... )
    >>> schema.names
    ('NAA', 'Cr')
    >>> len(schema)
    2
    """

    parameters: tuple[ParameterSpec, ...]

    def __post_init__(self) -> None:
        names = [p.name for p in self.parameters]
        if len(names) != len(set(names)):
            duplicates = sorted({n for n in names if names.count(n) > 1})
            raise SchemaError(f"Duplicate parameter names in schema: {duplicates}")

    def __len__(self) -> int:
        """Return the number of parameters (columns) described by this schema.

        Returns
        -------
        int
            Number of entries in ``parameters``.

        Examples
        --------
        >>> from mrsvalidate.schema import ParameterRole, ParameterSchema, ParameterSpec
        >>> schema = ParameterSchema(
        ...     (ParameterSpec("NAA", ParameterRole.CONCENTRATION, "mM"),)
        ... )
        >>> len(schema)
        1
        """
        return len(self.parameters)

    def __iter__(self) -> Iterator[ParameterSpec]:
        """Iterate over the parameter specs in column order.

        Returns
        -------
        Iterator[ParameterSpec]
            An iterator yielding each :class:`ParameterSpec` in the
            same order as the underlying ``values`` columns.

        Examples
        --------
        >>> from mrsvalidate.schema import ParameterRole, ParameterSchema, ParameterSpec
        >>> schema = ParameterSchema(
        ...     (ParameterSpec("NAA", ParameterRole.CONCENTRATION, "mM"),)
        ... )
        >>> [p.name for p in schema]
        ['NAA']
        """
        return iter(self.parameters)

    def __getitem__(self, index: int) -> ParameterSpec:
        """Return the parameter spec at a given column index.

        Parameters
        ----------
        index : int
            Zero-based column index into ``parameters``.

        Returns
        -------
        ParameterSpec
            The parameter specification at ``index``.

        Examples
        --------
        >>> from mrsvalidate.schema import ParameterRole, ParameterSchema, ParameterSpec
        >>> schema = ParameterSchema(
        ...     (ParameterSpec("NAA", ParameterRole.CONCENTRATION, "mM"),)
        ... )
        >>> schema[0].name
        'NAA'
        """
        return self.parameters[index]

    @property
    def names(self) -> tuple[str, ...]:
        """Names of every parameter, in column order.

        Returns
        -------
        tuple of str
            Column names in the exact order they appear in ``values``.

        Examples
        --------
        >>> from mrsvalidate.schema import ParameterRole, ParameterSchema, ParameterSpec
        >>> schema = ParameterSchema(
        ...     (ParameterSpec("NAA", ParameterRole.CONCENTRATION, "mM"),)
        ... )
        >>> schema.names
        ('NAA',)
        """
        return tuple(p.name for p in self.parameters)

    def index_of(self, name: str) -> int:
        """Return the column index of the parameter with a given name.

        Parameters
        ----------
        name : str
            Parameter name to look up.

        Returns
        -------
        int
            Zero-based column index of the parameter named ``name``.

        Raises
        ------
        SchemaError
            If no parameter named ``name`` exists in this schema.

        Examples
        --------
        >>> from mrsvalidate.schema import ParameterRole, ParameterSchema, ParameterSpec
        >>> schema = ParameterSchema(
        ...     (
        ...         ParameterSpec("NAA", ParameterRole.CONCENTRATION, "mM"),
        ...         ParameterSpec("Cr", ParameterRole.CONCENTRATION, "mM"),
        ...     )
        ... )
        >>> schema.index_of("Cr")
        1
        """
        for i, p in enumerate(self.parameters):
            if p.name == name:
                return i
        raise SchemaError(f"No parameter named {name!r} in schema")

    def indices_by_role(self, role: ParameterRole) -> tuple[int, ...]:
        """Return the column indices of every parameter with a given role.

        Parameters
        ----------
        role : ParameterRole
            Semantic role to match against every parameter's ``role``.

        Returns
        -------
        tuple of int
            Zero-based column indices, in column order, of every
            parameter whose ``role`` equals ``role``. Empty if no
            column has that role.

        Examples
        --------
        >>> from mrsvalidate.schema import ParameterRole, ParameterSchema, ParameterSpec
        >>> schema = ParameterSchema(
        ...     (
        ...         ParameterSpec("NAA", ParameterRole.CONCENTRATION, "mM"),
        ...         ParameterSpec("Cr", ParameterRole.CONCENTRATION, "mM"),
        ...         ParameterSpec("linewidth", ParameterRole.LORENTZIAN_LINESHAPE, "Hz"),
        ...     )
        ... )
        >>> schema.indices_by_role(ParameterRole.CONCENTRATION)
        (0, 1)
        """
        return tuple(i for i, p in enumerate(self.parameters) if p.role is role)

    def indices_by_metabolite(self, metabolite: str) -> tuple[int, ...]:
        """Return the column indices of every parameter for a given metabolite.

        Parameters
        ----------
        metabolite : str
            Metabolite name to match against every parameter's
            ``metabolite`` field.

        Returns
        -------
        tuple of int
            Zero-based column indices, in column order, of every
            parameter whose ``metabolite`` equals ``metabolite``. Empty
            if no column pertains to that metabolite.

        Examples
        --------
        >>> from mrsvalidate.schema import ParameterRole, ParameterSchema, ParameterSpec
        >>> schema = ParameterSchema(
        ...     (
        ...         ParameterSpec(
        ...             "NAA", ParameterRole.CONCENTRATION, "mM", metabolite="NAA"
        ...         ),
        ...     )
        ... )
        >>> schema.indices_by_metabolite("NAA")
        (0,)
        """
        return tuple(
            i for i, p in enumerate(self.parameters) if p.metabolite == metabolite
        )

    def index_by_role_and_metabolite(self, role: ParameterRole, metabolite: str) -> int:
        """Return the single column index for one metabolite's parameter of a given role.

        Parameters
        ----------
        role : ParameterRole
            Semantic role to match, e.g. ``ParameterRole.LORENTZIAN_LINESHAPE``.
        metabolite : str
            Metabolite name to match, e.g. ``"NAA"``.

        Returns
        -------
        int
            The single zero-based column index matching both ``role``
            and ``metabolite`` — e.g.
            ``index_by_role_and_metabolite(ParameterRole.LORENTZIAN_LINESHAPE, "NAA")``
            for NAA's Lorentzian linewidth column.

        Raises
        ------
        SchemaError
            If there is not exactly one match (zero, or an ambiguous
            more-than-one).

        Examples
        --------
        >>> from mrsvalidate.schema import ParameterRole, ParameterSchema, ParameterSpec
        >>> schema = ParameterSchema(
        ...     (
        ...         ParameterSpec(
        ...             "NAA_linewidth",
        ...             ParameterRole.LORENTZIAN_LINESHAPE,
        ...             "Hz",
        ...             metabolite="NAA",
        ...         ),
        ...     )
        ... )
        >>> schema.index_by_role_and_metabolite(
        ...     ParameterRole.LORENTZIAN_LINESHAPE, "NAA"
        ... )
        0
        """

        matches = tuple(
            i
            for i, p in enumerate(self.parameters)
            if p.role is role and p.metabolite == metabolite
        )
        if len(matches) != 1:
            raise SchemaError(
                f"Expected exactly one parameter with role={role.value!r} and "
                f"metabolite={metabolite!r}, found {len(matches)}"
            )
        return matches[0]


@dataclass(frozen=True)
class ObservationSet:
    """A validated block of fitted-parameter values with stable sample identity.

    An ``ObservationSet`` couples a 2D matrix of fitted-parameter
    ``values`` with the :class:`ParameterSchema` describing its columns
    and the ``sample_ids`` giving each row a stable identity.
    ``sample_ids`` is the only valid pairing key across observation
    sets; row position alone must never be treated as a pairing
    contract. Use :func:`mrsvalidate.alignment.align_observations` to
    compare two ``ObservationSet`` instances.

    Attributes
    ----------
    sample_ids : ndarray of str
        One identifier per row of ``values``, of shape ``(n_samples,)``.
        The only valid key for pairing rows across observation sets.
    values : ndarray of float
        Fitted-parameter matrix of shape ``(n_samples,
        n_fit_parameters)``.
    parameters : ParameterSchema
        Ordered description of every column of ``values``; its length
        must equal ``values.shape[1]``.
    subject_ids : ndarray of str or None, optional
        One subject identifier per row, for datasets where multiple
        samples (e.g. repeated scans) belong to the same subject.
        Default is ``None``.
    cluster_ids : ndarray of str or None, optional
        One cluster identifier per row, for analyses that must account
        for non-independence among samples (e.g. mixed-effects models).
        Default is ``None``.
    model_id : str or None, optional
        Identifier of the fitting model or pipeline that produced these
        values. Default is ``None``.
    metadata : Mapping[str, Any], optional
        Free-form additional information about this observation set.
        Default is an empty mapping.

    Raises
    ------
    SchemaError
        If ``values`` is not 2D, if ``sample_ids`` is not 1D, if the
        lengths of ``sample_ids``, ``subject_ids``, or ``cluster_ids``
        do not match ``values.shape[0]``, or if ``len(parameters)``
        does not match ``values.shape[1]``.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.schema import (
    ...     ObservationSet,
    ...     ParameterRole,
    ...     ParameterSchema,
    ...     ParameterSpec,
    ... )
    >>> schema = ParameterSchema(
    ...     (
    ...         ParameterSpec("NAA", ParameterRole.CONCENTRATION, "mM", metabolite="NAA"),
    ...         ParameterSpec("Cr", ParameterRole.CONCENTRATION, "mM", metabolite="Cr"),
    ...     )
    ... )
    >>> obs = ObservationSet(
    ...     sample_ids=np.array(["s1", "s2"]),
    ...     values=np.array([[1.0, 2.0], [3.0, 4.0]]),
    ...     parameters=schema,
    ... )
    >>> obs.n_samples, obs.n_parameters
    (2, 2)
    >>> obs.column("Cr")
    array([2., 4.])
    """

    sample_ids: NDArray[np.str_]
    values: NDArray[np.float64]
    parameters: ParameterSchema
    subject_ids: NDArray[np.str_] | None = None
    cluster_ids: NDArray[np.str_] | None = None
    model_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sample_ids = np.asarray(self.sample_ids, dtype=str)
        values = np.asarray(self.values, dtype=np.float64)

        if values.ndim != 2:
            raise SchemaError(
                f"ObservationSet.values must be 2D (n_samples, n_fit_parameters), "
                f"got shape {values.shape}"
            )
        if sample_ids.ndim != 1:
            raise SchemaError("ObservationSet.sample_ids must be 1D")
        if sample_ids.shape[0] != values.shape[0]:
            raise SchemaError(
                "ObservationSet.sample_ids length "
                f"({sample_ids.shape[0]}) must match values.shape[0] "
                f"({values.shape[0]})"
            )
        if len(self.parameters) != values.shape[1]:
            raise SchemaError(
                "ObservationSet.parameters length "
                f"({len(self.parameters)}) must match values.shape[1] "
                f"({values.shape[1]})"
            )

        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "values", values)

        for attr_name in ("subject_ids", "cluster_ids"):
            attr = getattr(self, attr_name)
            if attr is None:
                continue
            arr = np.asarray(attr, dtype=str)
            if arr.shape != sample_ids.shape:
                raise SchemaError(
                    f"ObservationSet.{attr_name} shape {arr.shape} must match "
                    f"sample_ids shape {sample_ids.shape}"
                )
            object.__setattr__(self, attr_name, arr)

    @property
    def n_samples(self) -> int:
        """Number of rows (samples) in ``values``.

        Returns
        -------
        int
            ``values.shape[0]``.
        """
        return self.values.shape[0]

    @property
    def n_parameters(self) -> int:
        """Number of columns (fitted parameters) in ``values``.

        Returns
        -------
        int
            ``values.shape[1]``.
        """
        return self.values.shape[1]

    def column(self, name: str) -> NDArray[np.float64]:
        """Return the column of ``values`` for a named parameter.

        Parameters
        ----------
        name : str
            Name of the parameter (column) to retrieve, as it appears
            in ``self.parameters``.

        Returns
        -------
        ndarray of float
            One-dimensional array of shape ``(n_samples,)`` holding the
            values of the named parameter for every sample.

        Raises
        ------
        SchemaError
            If no parameter named ``name`` exists in ``self.parameters``.

        Examples
        --------
        >>> import numpy as np
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
        ...     sample_ids=np.array(["s1", "s2"]),
        ...     values=np.array([[1.0], [3.0]]),
        ...     parameters=schema,
        ... )
        >>> obs.column("NAA")
        array([1., 3.])
        """

        return self.values[:, self.parameters.index_of(name)]

    def column_for_metabolite_role(self, role: ParameterRole, metabolite: str) -> NDArray[np.float64]:
        """Return the column for one metabolite's parameter of a given role.

        Parameters
        ----------
        role : ParameterRole
            Semantic role to match, e.g. ``ParameterRole.LORENTZIAN_LINESHAPE``.
        metabolite : str
            Metabolite name to match, e.g. ``"NAA"``.

        Returns
        -------
        ndarray of float
            One-dimensional array of shape ``(n_samples,)`` holding the
            values of the matched column for every sample — e.g.
            ``column_for_metabolite_role(ParameterRole.LORENTZIAN_LINESHAPE, "NAA")``
            for NAA's Lorentzian linewidth, looked up via the schema
            rather than an assumed column position.

        Raises
        ------
        SchemaError
            If there is not exactly one column matching both ``role``
            and ``metabolite``.

        Examples
        --------
        >>> import numpy as np
        >>> from mrsvalidate.schema import (
        ...     ObservationSet,
        ...     ParameterRole,
        ...     ParameterSchema,
        ...     ParameterSpec,
        ... )
        >>> schema = ParameterSchema(
        ...     (
        ...         ParameterSpec(
        ...             "NAA_linewidth",
        ...             ParameterRole.LORENTZIAN_LINESHAPE,
        ...             "Hz",
        ...             metabolite="NAA",
        ...         ),
        ...     )
        ... )
        >>> obs = ObservationSet(
        ...     sample_ids=np.array(["s1", "s2"]),
        ...     values=np.array([[5.0], [6.0]]),
        ...     parameters=schema,
        ... )
        >>> obs.column_for_metabolite_role(ParameterRole.LORENTZIAN_LINESHAPE, "NAA")
        array([5., 6.])
        """

        return self.values[:, self.parameters.index_by_role_and_metabolite(role, metabolite)]
