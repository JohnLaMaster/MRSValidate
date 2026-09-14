"""mrsvalidate.pipeline: declarative batch runner.

Describe a batch of analysis steps with a
:class:`~mrsvalidate.pipeline.config.RunConfig`, then execute it with
:func:`run_analysis_plan`, which serializes each result and writes a
self-describing, reproducible run manifest.

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

from __future__ import annotations

from .config import AnalysisStep, RunConfig
from .progress import NullProgressReporter, PrintProgressReporter, ProgressReporter
from .runner import RunResult, run_analysis_plan

__all__ = [
    "AnalysisStep",
    "NullProgressReporter",
    "PrintProgressReporter",
    "ProgressReporter",
    "RunConfig",
    "RunResult",
    "run_analysis_plan",
]
