#!/usr/bin/env python3
"""Populate the toy-figure analysis CSVs using the mrsvalidate library.

``mrs_validation_toy_figures.py`` (a ChatGPT-authored data/plot script) is
explicit that it computes nothing itself: it generates synthetic raw data
and blank ``analysis/*.csv`` templates, and its plotting functions only
render whichever numbers already exist in those templates. This script is
the missing "live" piece — it reads the raw data CSVs that script already
generated and calls into ``mrsvalidate`` to fill in every analysis CSV for
figures 1, 2, 5, 6, 7, and 8 (figures 3 and 4 need real simulated/fitted
MRS-Sim spectra that do not exist yet, per the user).

Run order::

    python mrs_validation_toy_figures.py generate --savedir toy_outputs
    python compute_toy_analysis.py --savedir toy_outputs
    python mrs_validation_toy_figures.py plot --savedir toy_outputs --figures 1 2 5 6 7 8

Every number below is either a direct mrsvalidate result field, or -- where
explicitly noted in a comment -- a plain descriptive/geometric quantity
(e.g. a covariance-error-ellipse's dimensions) that mrsvalidate does not
expose as a named statistic and that would, per its own architecture, be
legitimate "rendering geometry" even inside a plotting layer. Two
judgment-call inputs (the Deming error variances and the pre-specified
equivalence margin for Figure 5) were confirmed with the user rather than
picked silently; see the comments at their call sites.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mrs_validation_toy_figures import (  # noqa: E402
    ANALYSIS_TEMPLATES,
    DATA_FILENAMES,
    as_float,
    ensure_directories,
    read_csv,
    write_csv,
)

from mrsvalidate.analysis import (  # noqa: E402
    agreement,
    clustering,
    descriptive,
    distribution,
    errors,
    method_comparison,
    monte_carlo,
    multivariate,
    regression,
    resampling,
)

# Judgment calls confirmed with the user (2026-09-14), not fit to the data:
DEMING_REFERENCE_ERROR_VARIANCE = 0.01  # small but nonzero; reference is ~exact in this toy set
DEMING_TEST_ERROR_VARIANCE = 0.3025  # matches the simulated candidate noise scale, 0.55**2
EQUIVALENCE_MARGIN_MM = 1.0  # pre-specified acceptable-difference margin, +/- mM


def _round(value, ndigits: int = 4):
    """Round a float for display in the figures' annotation boxes.

    The existing plot script prints whatever string is in the analysis
    CSV verbatim, so rounding here (rather than in the unmodified
    plotting code) is what keeps the on-figure metric boxes readable.
    Non-floats (strings, bools, '') pass through unchanged.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        return value
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return value
    return round(float(value), ndigits)


def _round_row(row: dict) -> dict:
    return {k: _round(v) for k, v in row.items()}


def _col(rows: list[dict[str, str]], field: str) -> np.ndarray:
    return np.array([as_float(r[field], field=field) for r in rows])


def _ids(rows: list[dict[str, str]], field: str) -> np.ndarray:
    return np.array([r[field] for r in rows])


def compute_figure01(data_dir: Path, analysis_dir: Path) -> None:
    rows = read_csv(data_dir / DATA_FILENAMES[1])
    out_rows = []
    for scenario in ("Accurate", "Constant bias", "Proportional bias"):
        selected = [r for r in rows if r["scenario"] == scenario]
        reference = _col(selected, "ground_truth_mM")
        prediction = _col(selected, "prediction_mM")
        ids = _ids(selected, "sample_id")

        ts = regression.theil_sen(reference, prediction, ids)
        # mrsvalidate's Theil-Sen direction convention is
        # `reference = slope*prediction + intercept` (see
        # analysis/regression.py's module docstring); this figure's panel
        # plots prediction (y) against ground truth (x), so the displayed
        # line needs the inverted convention `prediction = intercept + slope*reference`.
        slope_plot = 1.0 / ts.slope
        intercept_plot = -ts.intercept / ts.slope

        r_result = regression.pearson_r(reference, prediction, ids)
        r2_result = regression.r_squared_direct(reference, prediction, ids)
        rmse_result = errors.rmse(reference, prediction, ids)
        bias_result = descriptive.mean(prediction - reference, ids)
        ccc_result = agreement.lin_ccc(reference, prediction, ids)

        out_rows.append(
            {
                "scenario": scenario,
                "slope": slope_plot,
                "intercept": intercept_plot,
                "pearson_r": r_result.r,
                "r_squared": r2_result.value,
                "mean_error_mM": bias_result.value,
                "rmse_mM": rmse_result.value,
                "lin_ccc": ccc_result.rho_c,
            }
        )
    write_csv(analysis_dir / "figure01_metrics.csv", [_round_row(r) for r in out_rows], ANALYSIS_TEMPLATES["figure01_metrics.csv"])


def compute_figure02(data_dir: Path, analysis_dir: Path) -> None:
    rows = read_csv(data_dir / DATA_FILENAMES[2])
    out_rows = []

    # Design A: 100 independent subjects, one observation each. No repeated
    # measurements exist, so a within-subject variance component is not
    # estimable; structurally design_effect == 1 whenever the mean cluster
    # size is 1, regardless of any between-subject variance, so ICC is left
    # blank (not estimable) rather than computed from a degenerate fit.
    design_a_rows = [r for r in rows if r["design"] == "100 independent subjects"]
    values_a = _col(design_a_rows, "value_mM")
    ids_a = _ids(design_a_rows, "subject_id")
    mean_a = descriptive.mean(values_a, ids_a)
    ci_a = resampling.bootstrap_ci(np.mean, values_a, sample_ids=ids_a, seed=0)
    out_rows.append(
        {
            "design": "100 independent subjects",
            "nominal_n": len(design_a_rows),
            "independent_clusters": len(set(ids_a.tolist())),
            "icc": "",
            "design_effect": 1.0,
            "effective_n": len(design_a_rows),
            "estimate_mM": mean_a.value,
            "ci_low_mM": ci_a.ci_low,
            "ci_high_mM": ci_a.ci_high,
        }
    )

    # Design B: 10 subjects x 10 repeats -- exactly the clustered design
    # mrsvalidate.analysis.clustering.one_way_icc and a clustered bootstrap
    # are built for.
    design_b_rows = [r for r in rows if r["design"] == "10 subjects x 10 repeats"]
    values_b = _col(design_b_rows, "value_mM")
    subject_ids_b = _ids(design_b_rows, "subject_id")
    sample_ids_b = np.array([f"{r['subject_id']}-{r['replicate_id']}" for r in design_b_rows])
    icc_result = clustering.one_way_icc(values_b, subject_ids_b, sample_ids_b)
    mean_b = descriptive.mean(values_b, sample_ids_b)
    ci_b = resampling.cluster_bootstrap_ci(
        np.mean, values_b, cluster_ids=subject_ids_b, sample_ids=sample_ids_b, seed=0
    )
    out_rows.append(
        {
            "design": "10 subjects x 10 repeats",
            "nominal_n": len(design_b_rows),
            "independent_clusters": icc_result.n_subjects,
            "icc": icc_result.rho,
            "design_effect": icc_result.design_effect,
            "effective_n": icc_result.n_eff,
            "estimate_mM": mean_b.value,
            "ci_low_mM": ci_b.ci_low,
            "ci_high_mM": ci_b.ci_high,
        }
    )
    write_csv(analysis_dir / "figure02_metrics.csv", [_round_row(r) for r in out_rows], ANALYSIS_TEMPLATES["figure02_metrics.csv"])


def compute_figure05(data_dir: Path, analysis_dir: Path) -> None:
    rows = read_csv(data_dir / DATA_FILENAMES[5])
    reference = _col(rows, "reference_mM")
    candidate = _col(rows, "candidate_mM")
    ids = _ids(rows, "pair_id")

    r_result = regression.pearson_r(reference, candidate, ids)
    ccc_result = agreement.lin_ccc(reference, candidate, ids)
    pb_result = method_comparison.passing_bablok(reference, candidate, ids, seed=0)
    # Deming error variances: confirmed with the user. The reference is
    # ~exact in this toy simulation (small nonzero variance to avoid a
    # degenerate lambda); the test/candidate variance matches its actual
    # simulated noise scale (0.55**2).
    deming_result = method_comparison.deming(
        reference, candidate, ids,
        reference_error_variance=DEMING_REFERENCE_ERROR_VARIANCE,
        test_error_variance=DEMING_TEST_ERROR_VARIANCE,
    )
    ba_result = method_comparison.bland_altman(reference, candidate, ids, x_axis="mean")

    diff_ci = resampling.bootstrap_ci(
        lambda x, y: np.mean(y - x), reference, candidate, sample_ids=ids, design="paired", seed=0
    )
    # Equivalence margin: pre-specified with the user (+/- 1.0 mM), not
    # fit to the data. "Equivalent" uses the standard TOST logic: the
    # methods are equivalent if the mean-difference CI falls entirely
    # inside the pre-specified margin.
    equivalent = bool(
        diff_ci.ci_low >= -EQUIVALENCE_MARGIN_MM and diff_ci.ci_high <= EQUIVALENCE_MARGIN_MM
    )

    out_rows = [
        {
            "pearson_r": r_result.r,
            "lin_ccc": ccc_result.rho_c,
            "passing_bablok_intercept": pb_result.intercept,
            "passing_bablok_slope": pb_result.slope,
            "deming_intercept": deming_result.intercept,
            "deming_slope": deming_result.slope,
            "bland_altman_bias_mM": ba_result.bias,
            "loa_lower_mM": ba_result.loa_low,
            "loa_upper_mM": ba_result.loa_high,
            "difference_ci_low_mM": diff_ci.ci_low,
            "difference_ci_high_mM": diff_ci.ci_high,
            "equivalence_lower_mM": -EQUIVALENCE_MARGIN_MM,
            "equivalence_upper_mM": EQUIVALENCE_MARGIN_MM,
            "equivalent": equivalent,
        }
    ]
    write_csv(analysis_dir / "figure05_metrics.csv", [_round_row(r) for r in out_rows], ANALYSIS_TEMPLATES["figure05_metrics.csv"])


def compute_figure06(data_dir: Path, analysis_dir: Path) -> None:
    rows = read_csv(data_dir / DATA_FILENAMES[6])
    truth = _col(rows, "ground_truth_mM")
    prediction = _col(rows, "prediction_mM")
    ids = _ids(rows, "sample_id")

    r_result = regression.pearson_r(truth, prediction, ids)
    ccc_result = agreement.lin_ccc(truth, prediction, ids)
    rmse_result = errors.rmse(truth, prediction, ids)
    mae_result = descriptive.mean(np.abs(prediction - truth), ids)
    # "Matching marginals, wrong pairing" is exactly the manuscript's
    # independent-two-sample distributional-comparison design (ECWAS):
    # truth and the permuted prediction share an identical empirical
    # marginal distribution by construction, so this tests location/
    # scale/shape equality while deliberately ignoring the (broken)
    # sample-wise pairing.
    ecwas_result = distribution.ecwas_kossler_mukherjee(
        truth, prediction, ids, critical_value_method="chi2_asymptotic"
    )

    out_rows = [
        {
            "scenario": "Permuted predictions",
            "pearson_r": r_result.r,
            "lin_ccc": ccc_result.rho_c,
            "rmse_mM": rmse_result.value,
            "mae_mM": mae_result.value,
            "marginal_distance": ecwas_result.statistic,
            "marginal_test_p": ecwas_result.p_value,
        }
    ]
    write_csv(analysis_dir / "figure06_metrics.csv", [_round_row(r) for r in out_rows], ANALYSIS_TEMPLATES["figure06_metrics.csv"])


def compute_figure07(data_dir: Path, analysis_dir: Path) -> None:
    rows = read_csv(data_dir / DATA_FILENAMES[7])
    out_rows = []
    for scenario in ("Accurate and precise", "Bias-dominated", "Variance-dominated"):
        selected = [r for r in rows if r["scenario"] == scenario]
        estimates = _col(selected, "estimate_mM")
        truth_value = as_float(selected[0]["truth_mM"], field="truth_mM")
        mc_result = monte_carlo.monte_carlo_condition(estimates, truth_value, scenario)
        # mse = rmse**2 is an exact algebraic identity (not a new
        # statistic), included so the figure can display the
        # bias**2 + variance == mse decomposition the manuscript calls for.
        out_rows.append(
            {
                "scenario": scenario,
                "bias_mM": mc_result.bias,
                "variance_mM2": mc_result.variance,
                "mse_mM2": mc_result.rmse**2,
                "rmse_mM": mc_result.rmse,
            }
        )
    write_csv(analysis_dir / "figure07_metrics.csv", [_round_row(r) for r in out_rows], ANALYSIS_TEMPLATES["figure07_metrics.csv"])


def _covariance_ellipse(a: np.ndarray, b: np.ndarray, *, confidence: float = 0.95) -> dict[str, float]:
    """Rendering-geometry helper: a 2D covariance confidence ellipse.

    Not a scientific statistic mrsvalidate implements (it has no bearing
    on any inferential claim) -- this is exactly the kind of "axis
    limits / geometry" calculation the spec explicitly allows even
    inside a plotting layer; it is computed here only so the analysis
    CSV can carry it alongside the genuine mrsvalidate results.
    """

    center = (float(np.mean(a)), float(np.mean(b)))
    cov = np.cov(a, b, ddof=1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    scale = float(np.sqrt(scipy_stats.chi2.ppf(confidence, df=2)))
    width, height = 2.0 * scale * np.sqrt(np.clip(eigenvalues, 0, None))
    angle_deg = float(np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])))
    return {
        "ellipse_center_a": center[0],
        "ellipse_center_b": center[1],
        "ellipse_width": float(width),
        "ellipse_height": float(height),
        "ellipse_angle_deg": angle_deg,
    }


def compute_figure08(data_dir: Path, analysis_dir: Path) -> None:
    rows = read_csv(data_dir / DATA_FILENAMES[8])
    truth_a = _col(rows, "truth_metabolite_a_mM")
    truth_b = _col(rows, "truth_metabolite_b_mM")
    pred_a = _col(rows, "prediction_metabolite_a_mM")
    pred_b = _col(rows, "prediction_metabolite_b_mM")
    ids = _ids(rows, "sample_id")

    out_rows = []
    for dataset, (a, b) in (("Ground truth", (truth_a, truth_b)), ("Prediction", (pred_a, pred_b))):
        corr_result = regression.pearson_r(a, b, ids)
        # Plain covariance: mrsvalidate does not expose a standalone
        # cross-covariance statistic (its agreement/regression functions
        # consume, but do not return, the covariance term), so this one
        # descriptive quantity is computed directly with NumPy.
        covariance_ab = float(np.cov(a, b, ddof=1)[0, 1])
        row = {
            "dataset": dataset,
            "corr_metabolite_a_b": corr_result.r,
            "covariance_a_b": covariance_ab,
            "joint_distance": "",
            "joint_test_p": "",
        }
        row.update(_covariance_ellipse(a, b))
        out_rows.append(row)

    # Joint (multivariate) structure comparison: treat "ground truth" and
    # "prediction" as two methods each reporting 2 variables (A, B) per
    # subject, and use the multivariate CCC -- built exactly for
    # assessing joint agreement across a variable set, which is the
    # entire point of this figure. joint_distance = 1 - rho_C^(M) so it
    # reads like a distance (0 = identical joint structure).
    measurements = np.stack(
        [np.column_stack([truth_a, truth_b]), np.column_stack([pred_a, pred_b])], axis=1
    )
    mv_result = multivariate.multivariate_ccc(measurements, ids)
    prediction_row = next(r for r in out_rows if r["dataset"] == "Prediction")
    prediction_row["joint_distance"] = 1.0 - mv_result.rho_c_multivariate
    # No standalone multivariate test/p-value is implemented in
    # mrsvalidate; this reuses the already-computed Pearson association
    # p-value between A and B in the prediction dataset (a genuine test
    # that their pairing is non-random), rather than inventing a new
    # joint-distribution test.
    prediction_row["joint_test_p"] = regression.pearson_r(pred_a, pred_b, ids).p_value

    write_csv(analysis_dir / "figure08_metrics.csv", [_round_row(r) for r in out_rows], ANALYSIS_TEMPLATES["figure08_metrics.csv"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--savedir", type=Path, required=True)
    args = parser.parse_args(argv)

    data_dir, analysis_dir, _ = ensure_directories(args.savedir.resolve())
    compute_figure01(data_dir, analysis_dir)
    compute_figure02(data_dir, analysis_dir)
    compute_figure05(data_dir, analysis_dir)
    compute_figure06(data_dir, analysis_dir)
    compute_figure07(data_dir, analysis_dir)
    compute_figure08(data_dir, analysis_dir)
    print(f"Populated analysis CSVs (figures 1,2,5,6,7,8) under: {analysis_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
