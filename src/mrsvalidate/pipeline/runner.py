"""Declarative batch execution: ``validate -> align -> analyze ->
serialize -> plot``.

Notes
-----
:func:`run_analysis_plan` itself only orchestrates already-built
:class:`~mrsvalidate.pipeline.config.AnalysisStep` calls (the caller is
responsible for validating/aligning inputs before constructing steps,
typically via :mod:`mrsvalidate.alignment` — the pipeline does not
hide that sequence behind implicit magic). One failed optional
analysis does not destroy the rest of the run unless
``fail_fast=True``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from ..results import AnalysisResult
from ..io.exporters import export_result_json, export_results_csv
from ..io.manifest import build_manifest
from .config import RunConfig
from .progress import NullProgressReporter, ProgressReporter

__all__ = ["run_analysis_plan", "RunResult"]


@dataclass(frozen=True)
class RunResult:
    """Outcome of one batch run.

    Attributes
    ----------
    run_dir : pathlib.Path
        Directory this run wrote its output to. Self-describing on
        disk: contains ``manifest.json``, ``resolved_config.json``,
        ``metrics/*.json`` (one per successful step), and
        ``metrics/summary.csv``.
    results : Mapping[str, AnalysisResult]
        Mapping from step name to its computed
        :class:`~mrsvalidate.results.AnalysisResult`, for steps that
        did not raise. A step whose function raised an exception has
        no entry here (see ``errors``).
    statuses : Mapping[str, str]
        Mapping from step name to its outcome status string — either
        the result's ``meta.status.value`` or ``"failed_exception"``
        if the step's function raised.
    errors : Mapping[str, str]
        Mapping from step name to the exception message, for steps
        whose function raised an exception.

    Examples
    --------
    >>> import tempfile
    >>> import numpy as np
    >>> from mrsvalidate.analysis.regression import regression_metrics
    >>> from mrsvalidate.pipeline import AnalysisStep, RunConfig, run_analysis_plan
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
    >>> run_result = run_analysis_plan(config)
    >>> run_result.statuses["naa_regression"]
    'valid'
    """

    run_dir: Path
    results: Mapping[str, AnalysisResult]
    statuses: Mapping[str, str]
    errors: Mapping[str, str]


def run_analysis_plan(config: RunConfig, *, progress: ProgressReporter | None = None) -> RunResult:
    """Execute a declarative batch of analysis steps and serialize the results.

    Runs each :class:`~mrsvalidate.pipeline.config.AnalysisStep` in
    ``config.steps`` in order, exports each successful result to
    ``metrics/<step_name>.json`` and a combined ``metrics/summary.csv``,
    and writes a run manifest (``manifest.json``) and the resolved
    configuration (``resolved_config.json``) under a fresh, timestamped
    or explicitly named run directory.

    Parameters
    ----------
    config : RunConfig
        Declarative description of the run: output location, steps to
        execute, and failure/seed/run-id policy.
    progress : ProgressReporter or None, optional
        Progress callback invoked before/after each step. Default is
        ``None``, in which case a
        :class:`~mrsvalidate.pipeline.progress.NullProgressReporter`
        is used (no output).

    Returns
    -------
    RunResult
        The collected results, per-step statuses, per-step error
        messages, and the run's output directory.

    Raises
    ------
    Exception
        Re-raised from a step's function when ``config.fail_fast`` is
        ``True`` and that step raises; otherwise every exception is
        caught, recorded in the returned ``RunResult.errors``, and
        execution continues with the remaining steps.

    Notes
    -----
    A step's function is called exactly as configured
    (``step.func(**step.kwargs)``); this function never validates or
    aligns inputs on the caller's behalf — that must happen before
    the :class:`~mrsvalidate.pipeline.config.AnalysisStep` is
    constructed, typically via :mod:`mrsvalidate.alignment`. When
    ``config.fail_fast`` is ``False`` (default), one failing step does
    not prevent the rest of the run, or the export of every other
    step's result, from completing.

    Examples
    --------
    >>> import tempfile
    >>> import numpy as np
    >>> from mrsvalidate.analysis.regression import regression_metrics
    >>> from mrsvalidate.pipeline import AnalysisStep, RunConfig, run_analysis_plan
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
    >>> run_result = run_analysis_plan(config)
    >>> sorted(run_result.results.keys())
    ['naa_regression']
    >>> (run_result.run_dir / "manifest.json").exists()
    True
    """

    progress = progress or NullProgressReporter()
    run_id = config.run_id or f"run-{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H%M%SZ')}"
    run_dir = Path(config.output_dir) / run_id
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)

    results: dict[str, AnalysisResult] = {}
    statuses: dict[str, str] = {}
    errors: dict[str, str] = {}

    for i, step in enumerate(config.steps, 1):
        progress.on_step_start(step.name, i, len(config.steps))
        try:
            result = step.func(**step.kwargs)
        except Exception as exc:
            statuses[step.name] = "failed_exception"
            errors[step.name] = str(exc)
            progress.on_step_end(step.name, "failed_exception")
            if config.fail_fast:
                raise
            continue

        results[step.name] = result
        statuses[step.name] = result.meta.status.value
        export_result_json(result, run_dir / "metrics" / f"{step.name}.json")
        progress.on_step_end(step.name, statuses[step.name])

    if results:
        export_results_csv(results, run_dir / "metrics" / "summary.csv")

    resolved_configuration = {
        "steps": [{"name": s.name, "kwargs_keys": sorted(s.kwargs.keys())} for s in config.steps],
        "fail_fast": config.fail_fast,
    }
    manifest = build_manifest(
        step_names=[s.name for s in config.steps],
        step_statuses=statuses,
        resolved_configuration=resolved_configuration,
        random_seed=config.random_seed,
    )
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    with open(run_dir / "resolved_config.json", "w") as f:
        json.dump(resolved_configuration, f, indent=2)

    return RunResult(run_dir=run_dir, results=results, statuses=statuses, errors=errors)
