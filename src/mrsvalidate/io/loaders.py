"""Minimal loaders for building an ``ObservationSet`` from plain data.

Two entry points are provided: one from in-memory arrays and one from
a wide-form CSV file.

Notes
-----
DataFrames are adapters here, never the core numerical representation
(spec: "Do not require DataFrame objects in core numerical APIs").
:func:`observation_set_from_csv` uses ``pandas`` only internally, to
read the file; the returned :class:`~mrsvalidate.schema.ObservationSet`
holds plain NumPy arrays.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..schema import ObservationSet, ParameterSchema

__all__ = ["observation_set_from_columns", "observation_set_from_csv"]


def observation_set_from_columns(
    sample_ids: NDArray[np.str_],
    columns: dict[str, NDArray[np.float64]],
    parameters: ParameterSchema,
    *,
    subject_ids: NDArray[np.str_] | None = None,
    cluster_ids: NDArray[np.str_] | None = None,
    model_id: str | None = None,
) -> ObservationSet:
    """Build an ``ObservationSet`` from a dict of parameter columns.

    Columns are looked up by name from ``columns`` and stacked in the
    order given by ``parameters`` — never in the dict's insertion
    order — so the resulting array's column order always matches its
    schema.

    Parameters
    ----------
    sample_ids : ndarray of str
        Stable sample identifier for each row, in the same order as
        every array in ``columns``.
    columns : dict of str to ndarray of float
        Mapping from parameter name to its 1D column of values. Must
        contain an entry for every ``parameters[i].name``; extra keys
        not referenced by ``parameters`` are ignored.
    parameters : ParameterSchema
        Ordered schema describing each column; determines both the
        set of required keys in ``columns`` and the output column
        order.
    subject_ids : ndarray of str or None, optional
        Subject identifier for each row, for clustered/repeated-measures
        data. Default is ``None``.
    cluster_ids : ndarray of str or None, optional
        Cluster identifier for each row. Default is ``None``.
    model_id : str or None, optional
        Identifier of the model that produced these values, for
        provenance. Default is ``None``.

    Returns
    -------
    ObservationSet
        A validated observation set with ``values`` stacked in
        ``parameters`` order.

    Raises
    ------
    ValueError
        If ``columns`` is missing an entry for any parameter named in
        ``parameters``.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.schema import ParameterRole, ParameterSchema, ParameterSpec
    >>> from mrsvalidate.io import observation_set_from_columns
    >>> parameters = ParameterSchema(
    ...     (ParameterSpec(name="NAA", role=ParameterRole.CONCENTRATION, unit="mM"),)
    ... )
    >>> sample_ids = np.array(["s1", "s2"])
    >>> columns = {"NAA": np.array([8.0, 9.5])}
    >>> obs = observation_set_from_columns(sample_ids, columns, parameters)
    >>> obs.column("NAA")
    array([8. , 9.5])
    """

    missing = [p.name for p in parameters if p.name not in columns]
    if missing:
        raise ValueError(f"columns missing for parameters: {missing}")

    values = np.column_stack([np.asarray(columns[p.name], dtype=np.float64) for p in parameters])
    return ObservationSet(
        sample_ids=sample_ids,
        values=values,
        parameters=parameters,
        subject_ids=subject_ids,
        cluster_ids=cluster_ids,
        model_id=model_id,
    )


def observation_set_from_csv(
    path: str | Path,
    parameters: ParameterSchema,
    *,
    sample_id_column: str = "sample_id",
    subject_id_column: str | None = None,
    cluster_id_column: str | None = None,
    model_id: str | None = None,
) -> ObservationSet:
    """Load a wide-form CSV into an ``ObservationSet``.

    The file must have one row per sample and one column per
    parameter, plus a sample-identifier column.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the wide-form CSV file.
    parameters : ParameterSchema
        Ordered schema describing which columns to read and in what
        order to stack them.
    sample_id_column : str, optional
        Name of the column holding the sample identifier. Default is
        ``"sample_id"``.
    subject_id_column : str or None, optional
        Name of the column holding the subject identifier, if any.
        Default is ``None`` (no subject IDs are read).
    cluster_id_column : str or None, optional
        Name of the column holding the cluster identifier, if any.
        Default is ``None``.
    model_id : str or None, optional
        Identifier of the model that produced these values, for
        provenance. Default is ``None``.

    Returns
    -------
    ObservationSet
        A validated observation set built from the CSV's columns.

    Notes
    -----
    Requires ``pandas``, used only internally to parse the file; the
    returned ``ObservationSet`` holds plain NumPy arrays (spec: "Do
    not require DataFrame objects in core numerical APIs").

    Examples
    --------
    >>> import tempfile, os
    >>> import numpy as np
    >>> from mrsvalidate.schema import ParameterRole, ParameterSchema, ParameterSpec
    >>> from mrsvalidate.io import observation_set_from_csv
    >>> parameters = ParameterSchema(
    ...     (ParameterSpec(name="NAA", role=ParameterRole.CONCENTRATION, unit="mM"),)
    ... )
    >>> path = tempfile.mktemp(suffix=".csv")
    >>> _ = open(path, "w").write("sample_id,NAA\\ns1,8.0\\ns2,9.5\\n")
    >>> obs = observation_set_from_csv(path, parameters)
    >>> obs.column("NAA")
    array([8. , 9.5])
    >>> os.remove(path)
    """

    import pandas as pd

    df = pd.read_csv(path)
    sample_ids = df[sample_id_column].to_numpy(dtype=str)
    columns = {p.name: df[p.name].to_numpy(dtype=np.float64) for p in parameters}
    subject_ids = df[subject_id_column].to_numpy(dtype=str) if subject_id_column else None
    cluster_ids = df[cluster_id_column].to_numpy(dtype=str) if cluster_id_column else None
    return observation_set_from_columns(
        sample_ids, columns, parameters,
        subject_ids=subject_ids, cluster_ids=cluster_ids, model_id=model_id,
    )
