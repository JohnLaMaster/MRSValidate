"""mrsvalidate: scientific analysis and visualization for evaluating
fitted MRS/metabolite-estimation models.

The public top-level namespace is deliberately small: schema and
alignment types used across every analysis, plus the universal result
envelope and the declarative batch pipeline. Domain-specific analysis
functions live in ``mrsvalidate.analysis.*`` submodules; plots live in
``mrsvalidate.viz``; I/O helpers live in ``mrsvalidate.io``.

Notes
-----
See ``LIBRARY_IMPLEMENTATION_SPEC.md`` for the full scientific
contract this package implements.

Examples
--------
>>> import mrsvalidate
>>> isinstance(mrsvalidate.__version__, str)
True
"""

from __future__ import annotations

from .alignment import AlignedComparison, AlignmentReport, align_observations
from .schema import ObservationSet, ParameterRole, ParameterSchema, ParameterSpec
from .results import AnalysisMeta, AnalysisResult, ResultStatus
from .pipeline import run_analysis_plan

__version__ = "0.1.0.dev0"

__all__ = [
    "__version__",
    "ObservationSet",
    "ParameterSpec",
    "ParameterSchema",
    "ParameterRole",
    "ResultStatus",
    "AnalysisMeta",
    "AnalysisResult",
    "align_observations",
    "AlignedComparison",
    "AlignmentReport",
    "run_analysis_plan",
]
