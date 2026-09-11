"""mrsvalidate.viz: manuscript-oriented scientific visualization.

Every plotting function here accepts a precomputed, typed analysis
result object (from ``mrsvalidate.analysis.*``) and only renders it —
it never computes or recomputes a scientific quantity itself.

Notes
-----
This subpackage must **never** import or call into
``mrsvalidate.analysis.*`` — see ``tests/plots/test_viz_boundary.py``
for the static (AST-level import scan) and dynamic (monkeypatch)
boundary tests that enforce this across every module in this
subpackage. Geometry-only calculations intrinsic to
rendering (axis limits, bin-center positions, color normalization) are
fine; recomputing a confidence interval, regression, test statistic,
calibration gap, or binned success probability is not.

Requires the ``viz`` extra (``matplotlib``). Matplotlib is imported
lazily inside each plotting function, so ``import mrsvalidate.viz``
itself never requires it — only actually calling a ``plot_*`` function
without the extra installed raises ``ImportError``.

Examples
--------
>>> import matplotlib
>>> matplotlib.use("Agg")
>>> import numpy as np
>>> from mrsvalidate.analysis.regression import regression_metrics
>>> import mrsvalidate.viz as viz
>>> reference = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
>>> prediction = np.array([1.1, 2.2, 2.9, 4.1, 5.2])
>>> sample_ids = np.array([f"s{i}" for i in range(5)])
>>> result = regression_metrics(reference, prediction, sample_ids)
>>> fig, ax = viz.plot_regression_result(result)
>>> type(fig).__name__
'Figure'
"""

from __future__ import annotations

from .agreement import plot_bland_altman
from .calibration import plot_calibration_result
from .information import plot_operating_envelope
from .response_curves import plot_response_curve
from .scatter import plot_regression_result

__all__ = [
    "plot_regression_result",
    "plot_bland_altman",
    "plot_calibration_result",
    "plot_operating_envelope",
    "plot_response_curve",
]
