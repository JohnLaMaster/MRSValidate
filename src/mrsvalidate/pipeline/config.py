"""Declarative batch-run configuration.

A :class:`RunConfig` is a plain, serializable description of what to
run — resolving it (which module.function each step calls, with which
arguments) happens once, up front, so the manifest can record exactly
what happened.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..results import AnalysisResult

__all__ = ["AnalysisStep", "RunConfig"]


@dataclass(frozen=True)
class AnalysisStep:
    """One named analysis call in a run plan.

    Attributes
    ----------
    name : str
        Name of this step, used as its key in the run's results,
        statuses, and errors mappings, and as the stem of its
        exported ``metrics/<name>.json`` file.
    func : Callable[..., AnalysisResult]
        The analysis function to call for this step. Must return an
        :class:`~mrsvalidate.results.AnalysisResult` (any subclass).
    kwargs : Mapping[str, Any], optional
        Keyword arguments passed through to ``func`` verbatim when the
        step is run. Default is an empty mapping.

    Examples
    --------
    >>> import numpy as np
    >>> from mrsvalidate.analysis.regression import regression_metrics
    >>> from mrsvalidate.pipeline import AnalysisStep
    >>> step = AnalysisStep(
    ...     name="naa_regression",
    ...     func=regression_metrics,
    ...     kwargs={
    ...         "reference": np.array([1.0, 2.0, 3.0]),
    ...         "prediction": np.array([1.1, 2.1, 2.9]),
    ...         "sample_ids": np.array(["s1", "s2", "s3"]),
    ...     },
    ... )
    >>> step.name
    'naa_regression'
    """

    name: str
    func: Callable[..., AnalysisResult]
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunConfig:
    """Full description of a batch run.

    Attributes
    ----------
    output_dir : str or pathlib.Path
        Base directory under which this run's ``<run_id>`` directory
        (containing ``manifest.json``, ``resolved_config.json``, and
        the ``metrics``/``figures`` subdirectories) is created.
    steps : tuple of AnalysisStep
        The analysis steps to execute, in order.
    fail_fast : bool, optional
        If ``True``, abort the whole run on the first step failure. If
        ``False`` (default), record the failure for that step (as
        ``"failed_exception"`` bookkeeping in the run manifest) and
        continue with the remaining steps, so one broken optional
        analysis does not destroy every other valid result.
    random_seed : int or None, optional
        Random seed recorded in the manifest for provenance/audit.
        Default is ``None``. Not passed to ``steps`` automatically —
        include it explicitly in a step's ``kwargs`` if a step itself
        needs it.
    run_id : str or None, optional
        Identifier for this run, used as the run's output subdirectory
        name. Default is ``None``, in which case
        :func:`~mrsvalidate.pipeline.run_analysis_plan` generates a
        timestamp-based identifier.

    Examples
    --------
    >>> import tempfile
    >>> import numpy as np
    >>> from mrsvalidate.analysis.regression import regression_metrics
    >>> from mrsvalidate.pipeline import AnalysisStep, RunConfig
    >>> step = AnalysisStep(
    ...     name="naa_regression",
    ...     func=regression_metrics,
    ...     kwargs={
    ...         "reference": np.array([1.0, 2.0, 3.0]),
    ...         "prediction": np.array([1.1, 2.1, 2.9]),
    ...         "sample_ids": np.array(["s1", "s2", "s3"]),
    ...     },
    ... )
    >>> config = RunConfig(output_dir=tempfile.mkdtemp(), steps=(step,))
    >>> config.fail_fast
    False
    """

    output_dir: str | Path
    steps: tuple[AnalysisStep, ...]
    fail_fast: bool = False
    random_seed: int | None = None
    run_id: str | None = None
