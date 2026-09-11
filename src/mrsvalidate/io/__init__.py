"""mrsvalidate.io: loaders, exporters, and run manifests.

Loaders build a validated
:class:`~mrsvalidate.schema.ObservationSet` from plain arrays or a
CSV file; exporters serialize an
:class:`~mrsvalidate.results.AnalysisResult` to CSV/JSON/NPZ; the
manifest helpers record a reproducible summary of a batch run.

Notes
-----
Nothing in this subpackage computes a scientific quantity — it only
constructs the typed containers analysis functions consume, or
serializes the results they produce.

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

from __future__ import annotations

from .exporters import export_arrays_npz, export_result_json, export_results_csv, result_to_jsonable
from .loaders import observation_set_from_columns, observation_set_from_csv
from .manifest import build_manifest, config_digest

__all__ = [
    "observation_set_from_columns",
    "observation_set_from_csv",
    "export_result_json",
    "export_results_csv",
    "export_arrays_npz",
    "result_to_jsonable",
    "build_manifest",
    "config_digest",
]
