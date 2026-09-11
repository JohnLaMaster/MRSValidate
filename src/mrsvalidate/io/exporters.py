"""Serialize typed analysis results to CSV / JSON / NPZ.

Notes
-----
Serialization never recomputes or "cleans up" a result — it converts
exactly the numbers already present on the result object into a
portable format. NaN/Inf are written as-is (Python's ``json`` module
emits the non-standard-but-universally-read ``NaN``/``Infinity``
tokens) rather than silently dropped or replaced.
"""

from __future__ import annotations

import dataclasses
import json
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..results import AnalysisResult

__all__ = ["result_to_jsonable", "export_result_json", "export_results_csv", "export_arrays_npz"]


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    return obj


def result_to_jsonable(result: AnalysisResult) -> dict:
    """Convert any typed analysis result to a plain, JSON-serializable dict.

    Recursively walks the (possibly nested) dataclass, converting
    enums to their ``.value``, NumPy scalars/arrays to Python
    scalars/lists, and mappings/sequences to plain dicts/lists.

    Parameters
    ----------
    result : AnalysisResult
        Any typed analysis result (any :class:`~mrsvalidate.results.AnalysisResult`
        subclass), including its nested ``meta``.

    Returns
    -------
    dict
        A plain dict containing only ``dict``, ``list``, ``str``,
        ``int``, ``float``, ``bool``, and ``None`` values — safe to
        pass directly to :func:`json.dump`.

    Examples
    --------
    >>> from mrsvalidate.results import AnalysisMeta, ResultStatus, ScalarMetricResult
    >>> from mrsvalidate.io import result_to_jsonable
    >>> meta = AnalysisMeta(status=ResultStatus.VALID, n_total=2, n_used=2, n_omitted=0)
    >>> result = ScalarMetricResult(meta=meta, value=0.5)
    >>> result_to_jsonable(result)["value"]
    0.5
    >>> result_to_jsonable(result)["meta"]["status"]
    'valid'
    """

    return _to_jsonable(result)


def export_result_json(result: AnalysisResult, path: str | Path) -> None:
    """Write one analysis result to a JSON file.

    Parameters
    ----------
    result : AnalysisResult
        The result to serialize; converted via :func:`result_to_jsonable`.
    path : str or pathlib.Path
        Destination file path. Parent directories are created if they
        do not already exist.

    Returns
    -------
    None
        The result is written to ``path`` as a side effect.

    Notes
    -----
    ``NaN``/``Infinity`` values are written using Python's ``json``
    module defaults (the non-standard but universally read ``NaN``,
    ``Infinity``, ``-Infinity`` tokens) rather than silently dropped or
    replaced with ``null``.

    Examples
    --------
    >>> import json, tempfile, os
    >>> from mrsvalidate.results import AnalysisMeta, ResultStatus, ScalarMetricResult
    >>> from mrsvalidate.io import export_result_json
    >>> meta = AnalysisMeta(status=ResultStatus.VALID, n_total=1, n_used=1, n_omitted=0)
    >>> result = ScalarMetricResult(meta=meta, value=0.5)
    >>> path = os.path.join(tempfile.mkdtemp(), "result.json")
    >>> export_result_json(result, path)
    >>> json.load(open(path))["value"]
    0.5
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result_to_jsonable(result), f, indent=2)


def export_results_csv(results: Mapping[str, AnalysisResult], path: str | Path) -> None:
    """Write a long-form CSV summary of several named results.

    One row is written per named result, with its status and
    (if present) scalar-like fields flattened into columns.

    Parameters
    ----------
    results : Mapping[str, AnalysisResult]
        Mapping from a result name (used as the row's ``name``) to its
        computed :class:`~mrsvalidate.results.AnalysisResult`.
    path : str or pathlib.Path
        Destination file path. Parent directories are created if they
        do not already exist.

    Returns
    -------
    None
        The CSV is written to ``path`` as a side effect.

    Notes
    -----
    Array-valued fields are omitted from the CSV (they belong in the
    NPZ/JSON artifacts, via :func:`export_arrays_npz` /
    :func:`export_result_json`) rather than flattened or truncated
    into a cell.

    Examples
    --------
    >>> import csv, tempfile, os
    >>> from mrsvalidate.results import AnalysisMeta, ResultStatus, ScalarMetricResult
    >>> from mrsvalidate.io import export_results_csv
    >>> meta = AnalysisMeta(status=ResultStatus.VALID, n_total=1, n_used=1, n_omitted=0)
    >>> results = {"rmse": ScalarMetricResult(meta=meta, value=0.5)}
    >>> path = os.path.join(tempfile.mkdtemp(), "summary.csv")
    >>> export_results_csv(results, path)
    >>> rows = list(csv.DictReader(open(path)))
    >>> rows[0]["name"], rows[0]["value"]
    ('rmse', '0.5')
    """

    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    all_columns: set[str] = {"name", "status", "method", "n_total", "n_used", "n_omitted"}
    for name, result in results.items():
        row: dict[str, Any] = {
            "name": name,
            "status": result.meta.status.value,
            "method": result.meta.method,
            "n_total": result.meta.n_total,
            "n_used": result.meta.n_used,
            "n_omitted": result.meta.n_omitted,
        }
        for f in dataclasses.fields(result):
            if f.name == "meta":
                continue
            value = getattr(result, f.name)
            if isinstance(value, (int, float, str, bool)) or value is None:
                row[f.name] = value
                all_columns.add(f.name)
            elif isinstance(value, Enum):
                row[f.name] = value.value
                all_columns.add(f.name)
        rows.append(row)

    columns = sorted(all_columns)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def export_arrays_npz(arrays: Mapping[str, np.ndarray], path: str | Path) -> None:
    """Write named arrays to a single compressed-free NumPy ``.npz`` archive.

    Parameters
    ----------
    arrays : Mapping[str, ndarray]
        Mapping from array name to array; each is converted with
        ``np.asarray`` before being written.
    path : str or pathlib.Path
        Destination file path. Parent directories are created if they
        do not already exist.

    Returns
    -------
    None
        The archive is written to ``path`` as a side effect.

    Examples
    --------
    >>> import os, tempfile
    >>> import numpy as np
    >>> from mrsvalidate.io import export_arrays_npz
    >>> path = os.path.join(tempfile.mkdtemp(), "arrays.npz")
    >>> export_arrays_npz({"residuals": np.array([0.1, -0.2, 0.05])}, path)
    >>> loaded = np.load(path)
    >>> loaded["residuals"]
    array([ 0.1 , -0.2 ,  0.05])
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: np.asarray(v) for k, v in arrays.items()})
