"""End-to-end synthetic run: validate -> align -> analyze -> serialize.

Exercises the pipeline runner against a small synthetic dataset and
checks that the run directory is self-describing and deterministic.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import mrsvalidate as mrs
from mrsvalidate.analysis.errors import rmse
from mrsvalidate.analysis.regression import regression_metrics
from mrsvalidate.analysis.agreement import lin_ccc
from mrsvalidate.pipeline import AnalysisStep, RunConfig, run_analysis_plan


def _synthetic_observation_sets():
    rng = np.random.default_rng(0)
    n = 60
    schema = mrs.ParameterSchema(
        (mrs.ParameterSpec("NAA", mrs.ParameterRole.CONCENTRATION, "mM"),)
    )
    sample_ids = np.array([f"s{i}" for i in range(n)])
    true_values = rng.uniform(1.0, 10.0, size=n)

    reference = mrs.ObservationSet(
        sample_ids=sample_ids, values=true_values.reshape(-1, 1), parameters=schema
    )
    predicted = true_values + rng.normal(scale=0.3, size=n)
    candidate = mrs.ObservationSet(
        sample_ids=sample_ids, values=predicted.reshape(-1, 1), parameters=schema, model_id="modelA"
    )
    return reference, candidate


def test_end_to_end_synthetic_run(tmp_path):
    reference, candidate = _synthetic_observation_sets()
    aligned = mrs.align_observations(reference, candidate, pairing="paired")
    assert aligned.report.n_final == 60  # every sample matched by ID

    ref_col = aligned.reference.column("NAA")
    cand_col = aligned.candidate.column("NAA")
    ids = aligned.reference.sample_ids

    config = RunConfig(
        output_dir=tmp_path,
        steps=(
            AnalysisStep("rmse_naa", rmse, {"reference": ref_col, "prediction": cand_col, "sample_ids": ids}),
            AnalysisStep(
                "regression_naa", regression_metrics,
                {"reference": ref_col, "prediction": cand_col, "sample_ids": ids},
            ),
            AnalysisStep("lin_ccc_naa", lin_ccc, {"reference": ref_col, "prediction": cand_col, "sample_ids": ids}),
        ),
        random_seed=42,
        run_id="run-test-fixed-id",
    )

    result = run_analysis_plan(config)

    assert result.run_dir.exists()
    assert (result.run_dir / "manifest.json").exists()
    assert (result.run_dir / "resolved_config.json").exists()
    assert (result.run_dir / "metrics" / "summary.csv").exists()
    assert (result.run_dir / "metrics" / "rmse_naa.json").exists()
    assert (result.run_dir / "metrics" / "regression_naa.json").exists()
    assert (result.run_dir / "metrics" / "lin_ccc_naa.json").exists()

    manifest = json.loads((result.run_dir / "manifest.json").read_text())
    assert manifest["random_seed"] == 42
    assert manifest["library_version"] == mrs.__version__
    assert set(manifest["step_statuses"].keys()) == {"rmse_naa", "regression_naa", "lin_ccc_naa"}
    assert all(s == "valid" for s in manifest["step_statuses"].values())

    rmse_json = json.loads((result.run_dir / "metrics" / "rmse_naa.json").read_text())
    assert rmse_json["value"] == pytest.approx(result.results["rmse_naa"].value)


def test_end_to_end_run_is_reproducible(tmp_path):
    reference, candidate = _synthetic_observation_sets()
    aligned = mrs.align_observations(reference, candidate, pairing="paired")
    ref_col = aligned.reference.column("NAA")
    cand_col = aligned.candidate.column("NAA")
    ids = aligned.reference.sample_ids

    def make_config(run_id):
        return RunConfig(
            output_dir=tmp_path,
            steps=(
                AnalysisStep("rmse_naa", rmse, {"reference": ref_col, "prediction": cand_col, "sample_ids": ids}),
            ),
            run_id=run_id,
        )

    r1 = run_analysis_plan(make_config("run-a"))
    r2 = run_analysis_plan(make_config("run-b"))
    assert r1.results["rmse_naa"].value == r2.results["rmse_naa"].value


def test_pipeline_continues_after_one_step_fails(tmp_path):
    def _broken(**kwargs):
        raise RuntimeError("boom")

    reference, candidate = _synthetic_observation_sets()
    ref_col = reference.column("NAA")
    ids = reference.sample_ids

    config = RunConfig(
        output_dir=tmp_path,
        steps=(
            AnalysisStep("broken_step", _broken, {}),
            AnalysisStep("rmse_naa", rmse, {"reference": ref_col, "prediction": ref_col, "sample_ids": ids}),
        ),
        fail_fast=False,
    )
    result = run_analysis_plan(config)
    assert "broken_step" in result.errors
    assert "rmse_naa" in result.results
    assert result.results["rmse_naa"].value == pytest.approx(0.0)
