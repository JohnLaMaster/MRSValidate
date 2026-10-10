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
limits of acceptability for Figure 5) were confirmed with the user rather than
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
ACCEPTABLE_LIMIT_MM = 1.0  # pre-specified limits of acceptability for the limits of agreement, +/- mM


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
    """Method comparison for an *in silico* paired dataset (exact reference).

    Follows the manuscript's Method Comparison section for the in silico
    regime: Bland-Altman against the reference (Krouwer), robust
    regression-based limits of agreement with bootstrap confidence bands,
    judged against limits of acceptability fixed in advance;
    Passing-Bablok as the primary regression; and the Silveira three-part
    protocol (Tests 1-3).
    """
    rows = read_csv(data_dir / DATA_FILENAMES[5])
    reference = _col(rows, "reference_mM")
    candidate = _col(rows, "candidate_mM")
    ids = _ids(rows, "pair_id")

    r_result = regression.pearson_r(reference, candidate, ids)
    ccc_result = agreement.lin_ccc(reference, candidate, ids)
    # Linearity: bootstrap-calibrated CUSUM, with the error-variance ratio
    # known from the simulation (same values as the Deming fit below).
    pb = method_comparison.passing_bablok(
        reference, candidate, ids,
        cusum_method="bootstrap",
        cusum_bootstrap_variance_ratio=DEMING_REFERENCE_ERROR_VARIANCE / DEMING_TEST_ERROR_VARIANCE,
        seed=0,
    )
    ba = method_comparison.bland_altman(reference, candidate, ids, x_axis="reference", seed=0)

    # Limits of acceptability: pre-specified with the user (+/- 1.0 mM),
    # not fit to the data. The limits of agreement are acceptable only if
    # their confidence bands lie inside these limits across the whole
    # concentration range.
    lower_band_low = ba.conditional_loa_low_band[0]
    upper_band_high = ba.conditional_loa_high_band[1]
    loa_within_acceptable = bool(
        np.min(lower_band_low) >= -ACCEPTABLE_LIMIT_MM
        and np.max(upper_band_high) <= ACCEPTABLE_LIMIT_MM
    )

    # Silveira protocol. Test 1 decides whether a translation correction
    # (subtracting the estimated bias) is applied before Tests 2 and 3.
    t1 = method_comparison.equivalence_test_1(reference, candidate, ids, seed=0)
    correction = t1.alpha if t1.reject_null_bootstrap else None
    t2 = method_comparison.equivalence_test_2(reference, candidate, ids, bias_correction=correction, seed=0)
    # Deming error variances: confirmed with the user. The reference is
    # ~exact in this toy simulation (small nonzero variance to avoid a
    # degenerate lambda); the candidate variance matches its simulated
    # noise scale (0.55**2).
    t3 = method_comparison.equivalence_test_3(
        reference, candidate, ids,
        reference_error_variance=DEMING_REFERENCE_ERROR_VARIANCE,
        test_error_variance=DEMING_TEST_ERROR_VARIANCE,
        bias_correction=correction, seed=0,
    )

    out_rows = [
        {
            "pearson_r": r_result.r,
            "lin_ccc": ccc_result.rho_c,
            "pb_intercept_mM": pb.intercept,
            "pb_intercept_ci_low_mM": pb.intercept_ci[0],
            "pb_intercept_ci_high_mM": pb.intercept_ci[1],
            "pb_slope": pb.slope,
            "pb_slope_ci_low": pb.slope_ci[0],
            "pb_slope_ci_high": pb.slope_ci[1],
            "pb_ci_method": pb.ci_method,
            "pb_cusum_method": pb.cusum_method,
            "pb_cusum_p_value": pb.cusum_p_value,
            "pb_linearity_rejected": pb.cusum_reject_linearity,
            "ba_x_axis": ba.x_axis,
            "ba_location": ba.location,
            "ba_scale": ba.scale_method,
            "ba_confidence_level": ba.confidence_level,
            "acceptable_lower_mM": -ACCEPTABLE_LIMIT_MM,
            "acceptable_upper_mM": ACCEPTABLE_LIMIT_MM,
            "loa_within_acceptable": loa_within_acceptable,
        }
    ]
    write_csv(analysis_dir / "figure05_metrics.csv", [_round_row(r) for r in out_rows], ANALYSIS_TEMPLATES["figure05_metrics.csv"])

    # Fitted conditional bias and limits of agreement at each observed
    # reference value (bias(c) = b0 + b1 * c from the library's fitted
    # coefficients; the limits are the library's own per-sample values).
    b0, b1 = ba.conditional_bias_coefficients
    order = np.argsort(ba.x_values)
    curve_rows = [
        {
            "reference_mM": ba.x_values[i],
            "bias_mM": b0 + b1 * ba.x_values[i],
            "loa_lower_mM": ba.conditional_loa_low[i],
            "loa_upper_mM": ba.conditional_loa_high[i],
        }
        for i in order
    ]
    write_csv(analysis_dir / "figure05_ba_curves.csv", [_round_row(r) for r in curve_rows], ANALYSIS_TEMPLATES["figure05_ba_curves.csv"])

    band_rows = [
        {
            "reference_mM": c,
            "bias_ci_low_mM": ba.conditional_bias_band[0][k],
            "bias_ci_high_mM": ba.conditional_bias_band[1][k],
            "loa_lower_ci_low_mM": ba.conditional_loa_low_band[0][k],
            "loa_lower_ci_high_mM": ba.conditional_loa_low_band[1][k],
            "loa_upper_ci_low_mM": ba.conditional_loa_high_band[0][k],
            "loa_upper_ci_high_mM": ba.conditional_loa_high_band[1][k],
        }
        for k, c in enumerate(ba.ci_grid)
    ]
    write_csv(analysis_dir / "figure05_ba_bands.csv", [_round_row(r) for r in band_rows], ANALYSIS_TEMPLATES["figure05_ba_bands.csv"])

    silveira_rows = [
        {"test": "1 (accuracy)", "parameter": "alpha (mM)", "estimate": t1.alpha,
         "ci_low": t1.alpha_ci_bootstrap[0], "ci_high": t1.alpha_ci_bootstrap[1],
         "ci_level": t1.confidence_level, "null_value": 0.0, "rejected": t1.reject_null_bootstrap},
        {"test": "2 (precision)", "parameter": "beta", "estimate": t2.beta,
         "ci_low": t2.beta_ci_bootstrap[0], "ci_high": t2.beta_ci_bootstrap[1],
         "ci_level": t2.confidence_level, "null_value": 0.0, "rejected": t2.reject_null_bootstrap},
        {"test": "3 (agreement)", "parameter": "alpha (mM)", "estimate": t3.alpha,
         "ci_low": t3.alpha_ci[0], "ci_high": t3.alpha_ci[1],
         "ci_level": t3.confidence_level_per_parameter, "null_value": 0.0,
         "rejected": not (t3.alpha_ci[0] <= 0.0 <= t3.alpha_ci[1])},
        {"test": "3 (agreement)", "parameter": "beta", "estimate": t3.beta,
         "ci_low": t3.beta_ci[0], "ci_high": t3.beta_ci[1],
         "ci_level": t3.confidence_level_per_parameter, "null_value": 1.0,
         "rejected": not (t3.beta_ci[0] <= 1.0 <= t3.beta_ci[1])},
    ]
    write_csv(analysis_dir / "figure05_silveira.csv", [_round_row(r) for r in silveira_rows], ANALYSIS_TEMPLATES["figure05_silveira.csv"])

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
