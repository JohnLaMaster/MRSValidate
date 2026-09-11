"""Static and dynamic tests enforcing the viz/analysis architectural
boundary: ``mrsvalidate.viz.*`` must never import or call
``mrsvalidate.analysis.*``.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import numpy as np
import pytest

import mrsvalidate.viz as viz_pkg
from mrsvalidate.analysis.regression import regression_metrics
from mrsvalidate.analysis.method_comparison import bland_altman
from mrsvalidate.analysis.calibration import expected_calibration_error
from mrsvalidate.analysis.operating_envelope import operating_envelope
from mrsvalidate.analysis.response_curves import response_curve


def _viz_source_files() -> list[Path]:
    viz_dir = Path(viz_pkg.__file__).parent
    return sorted(viz_dir.glob("*.py"))


def test_viz_modules_never_import_analysis_statically():
    """AST-level check: no `import mrsvalidate.analysis...` or
    `from mrsvalidate.analysis... import ...` anywhere under viz/,
    including inside function bodies (deferred imports don't escape
    this check either)."""

    offending: list[str] = []
    for path in _viz_source_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("mrsvalidate.analysis") or alias.name == "mrsvalidate.analysis":
                        offending.append(f"{path.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith("mrsvalidate.analysis") or module == "..analysis" or module.startswith("..analysis"):
                    offending.append(f"{path.name}: from {module} import ...")

    assert not offending, f"viz modules must not import analysis: {offending}"


def test_plot_regression_result_does_not_invoke_analysis(monkeypatch):
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = np.array([1.1, 2.2, 2.9, 4.1, 5.2])
    ids = np.array([f"s{i}" for i in range(5)])
    known_result = regression_metrics(ref, pred, ids)  # computed BEFORE booby-trapping analysis

    def _boom(*args, **kwargs):
        raise AssertionError("analysis invoked by plot")

    monkeypatch.setattr("mrsvalidate.analysis.regression.regression_metrics", _boom)
    monkeypatch.setattr("mrsvalidate.analysis.regression.pearson_r", _boom)
    monkeypatch.setattr("mrsvalidate.analysis.regression.theil_sen", _boom)

    # Must still render because every scientific number is precomputed.
    from mrsvalidate.viz import plot_regression_result

    fig, ax = plot_regression_result(known_result)
    assert fig is not None
    assert ax is not None


def test_plot_bland_altman_does_not_invoke_analysis(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("analysis invoked by plot")

    monkeypatch.setattr("mrsvalidate.analysis.method_comparison.bland_altman", _boom)

    ref = np.array([1.0, 2.0, 3.0, 4.0])
    cand = ref + 1.0
    ids = np.array([f"s{i}" for i in range(4)])
    known_result = bland_altman(ref, cand, ids)

    from mrsvalidate.viz import plot_bland_altman

    fig, ax = plot_bland_altman(known_result)
    assert fig is not None


def test_plot_calibration_result_does_not_invoke_analysis(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("analysis invoked by plot")

    monkeypatch.setattr("mrsvalidate.analysis.calibration.expected_calibration_error", _boom)

    pit = np.random.default_rng(0).uniform(size=200)
    levels = np.array([0.1, 0.5, 0.9])
    ids = np.array([f"s{i}" for i in range(200)])
    known_result = expected_calibration_error(pit, levels, ids)

    from mrsvalidate.viz import plot_calibration_result

    fig, ax = plot_calibration_result(known_result)
    assert fig is not None


def test_plot_operating_envelope_does_not_invoke_analysis(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("analysis invoked by plot")

    monkeypatch.setattr("mrsvalidate.analysis.operating_envelope.operating_envelope", _boom)

    conc = np.random.default_rng(0).uniform(size=100)
    info = np.random.default_rng(1).uniform(size=100)
    success = np.random.default_rng(2).random(100) < 0.9
    ids = np.array([f"s{i}" for i in range(100)])
    known_result = operating_envelope(
        conc, info, success, ids,
        bin_edges=np.array([0.0, 0.5, 1.0]),
        stratum_edges=np.array([0.0, 0.5, 1.0]),
        target_reliability=0.8,
    )

    from mrsvalidate.viz import plot_operating_envelope

    fig, ax = plot_operating_envelope(known_result)
    assert fig is not None


def test_plot_response_curve_does_not_invoke_analysis(monkeypatch):
    outcome = np.array([1.0, 2.0, 3.0, 10.0])
    snr = np.array([5.0, 5.0, 50.0, 50.0])
    ids = np.array([f"s{i}" for i in range(4)])
    known_result = response_curve(
        outcome, {"mSNR": snr}, ids, bin_edges={"mSNR": np.array([0.0, 10.0, 100.0])}
    )

    def _boom(*args, **kwargs):
        raise AssertionError("analysis invoked by plot")

    monkeypatch.setattr("mrsvalidate.analysis.response_curves.response_curve", _boom)

    from mrsvalidate.viz import plot_response_curve

    fig, ax = plot_response_curve(known_result)
    assert fig is not None
