#!/usr/bin/env python3
"""Generate data and standalone per-subplot figures for eight validation toy examples.

This is a companion to ``mrs_validation_toy_figures.py`` that does NOT render
the eight composite multi-panel figures (that script already does, and owns
those combined PNGs). Instead, every individual subplot ("panel") of each
composite figure is rendered as its own self-contained Matplotlib figure --
with its own axis labels, legend, and title, independent of any sibling
panel's shared axes -- and exported as both SVG and PDF under
``<savedir>/panels/figureNN/<panel-slug>.{svg,pdf}``. This is meant for
importing individual panels into LaTeX (e.g. via ``subcaption``) so each one
can be captioned, referenced, and resized on its own.

This is intentionally a DATA-GENERATION AND PLOTTING script, not a statistics
implementation.  It does not calculate Pearson r, R-squared, Lin's CCC, RMSE,
ICC, effective sample size, confidence intervals, CRLBs, Bland--Altman limits,
Passing--Bablok/Deming regression, equivalence tests, covariance ellipses, or
operating-range confidence bounds.  Those results must come from the existing
statistics library and be supplied in the analysis files described below.

Quick start
-----------
Generate all numerical toy datasets, MRS-Sim job manifests, and blank analysis
templates in a directory selected by the caller (shared with the composite
script -- both can point at the same ``--savedir``)::

    python mrs_validation_toy_figures.py generate --savedir ./toy_outputs

Export the standalone panels for figures whose raw-data panels do not require
external results::

    python mrs_validation_toy_figures_individually.py plot --savedir ./toy_outputs \
        --figures 1 2 6 7 8

After the statistics library has populated the files in
``toy_outputs/analysis/``, export panels for all eight figures::

    python mrs_validation_toy_figures_individually.py plot --savedir ./toy_outputs \
        --figures all

Print the metric and MRS-Sim contracts::

    python mrs_validation_toy_figures_individually.py describe

MRS-Sim requirements
--------------------
Figure 3 (CRLB is not an accuracy certificate):
    A noise-free spectrum is useful as a clean anchor and for diagnosing
    deterministic model mismatch, but it is not sufficient for the intended
    figure.  Each condition also needs repeated, independently noised spectra
    (preferably 200--500 realizations; 100 is a pragmatic minimum) at a known,
    non-zero noise level.  Fit every realization with identical settings and
    export ground truth, fitted concentration, externally calculated CRLB (or
    reported standard error), the CRLB representation and units, SNR definition,
    linewidth, fit status, and sample/condition/replicate IDs.  Include at least
    a well-specified condition and a controlled model-mismatch condition.  The
    supplied stats library then calculates empirical bias/dispersion and the
    comparisons with the externally supplied CRLB.

Figure 4 (evidence-supported operating range):
    Use noisy Monte Carlo spectra over a COWS-informed concentration x SNR x
    linewidth grid, preferably 200--500 independent realizations per cell.
    Preserve ground truth, acquisition/fitting configuration, noise model and
    seed for every realization.  Pre-specify the acceptable error threshold and
    the desired success probability before analysis.  The stats library must
    calculate cell-wise success rates and one-sided lower confidence bounds.
    Noise-free templates may be retained once per grid cell for provenance, but
    a separate noise-free spectrum for every replicate is unnecessary.

Analysis files expected from the statistics library
---------------------------------------------------
All files are CSV and live in ``<savedir>/analysis``. Blank templates are made
by the ``generate`` command. Empty cells mean "not yet analyzed".

* figure01_metrics.csv: scenario, slope, intercept, pearson_r, r_squared,
  mean_error_mM, rmse_mM, lin_ccc.
* figure02_metrics.csv: design, nominal_n, independent_clusters, icc,
  design_effect, effective_n, estimate_mM, ci_low_mM, ci_high_mM.
* figure03_crlb_summary.csv: condition, metabolite, n_replicates, bias_mM,
  empirical_sd_mM, median_crlb_sd_mM, median_abs_error_mM,
  median_abs_standardized_error.
* figure04_operating_range.csv: metabolite, linewidth_hz, concentration_mM,
  snr, n_total, n_success, success_probability, success_lcb,
  acceptable_abs_error_mM, evidence_status.
* figure05_metrics.csv: one row containing pearson_r, lin_ccc, the
  Passing--Bablok intercept/slope with confidence intervals and CUSUM
  linearity result (pb_*), the Bland--Altman settings (ba_*), and the
  pre-specified acceptable limits with the loa_within_acceptable decision.
* figure05_ba_curves.csv: reference_mM, bias_mM, loa_lower_mM,
  loa_upper_mM (regression-based bias and limits of agreement).
* figure05_ba_bands.csv: reference_mM grid with confidence bands for the
  bias and both limits of agreement.
* figure05_silveira.csv: one row per Silveira test parameter: test,
  parameter, estimate, ci_low, ci_high, ci_level, null_value, rejected.
* figure06_metrics.csv: scenario, pearson_r, lin_ccc, rmse_mM, mae_mM,
  marginal_distance, marginal_test_p.
* figure07_metrics.csv: scenario, bias_mM, variance_mM2, mse_mM2, rmse_mM.
* figure08_metrics.csv: dataset, corr_metabolite_a_b, covariance_a_b,
  joint_distance, joint_test_p, ellipse_center_a, ellipse_center_b,
  ellipse_width, ellipse_height, ellipse_angle_deg.

Relevant metrics (state them; calculate them elsewhere)
-------------------------------------------------------
1. Association versus agreement: Pearson r, R-squared under an explicitly
   stated regression, slope, intercept, mean error/bias, MAE or RMSE, Lin's $\rho_C$.
2. Nominal versus effective N: ICC, design effect, effective sample size, naive
   and cluster-aware or subject-bootstrap confidence intervals; optionally a
   mixed-effects variance decomposition.
3. CRLB interpretation: empirical bias, empirical SD/variance, externally
   supplied CRLB SD/variance, empirical-variance-to-CRLB-variance ratio,
   standardized absolute error, interval coverage/calibration, and failure rate.
4. Operating range: absolute/relative error, a pre-specified success indicator,
   cell-wise success probability, one-sided confidence lower bound, sample count,
   and an evidence status that distinguishes validated, unacceptable, and
   insufficiently sampled cells.
5. Method comparison (in silico, exact reference): Pearson r and Lin's $\rho_C$;
   Bland--Altman against the reference with robust regression-based limits of
   agreement and their confidence bands, judged against pre-specified acceptable
   limits; Passing--Bablok slope/intercept with confidence intervals plus the
   CUSUM linearity test; and the Silveira three-part accuracy/precision/agreement
   protocol.
6. Marginal equality versus pairing: a marginal distribution distance/test plus
   paired MAE/RMSE, Pearson r, and Lin's $\rho_C$.
7. Bias versus variance: empirical bias, variance, MSE, RMSE, and the externally
   verified bias-squared-plus-variance decomposition.
8. Multivariate structure: per-variable marginal summaries/distances, covariance
   and correlation matrices, plus a justified joint-distribution statistic such
   as energy distance or MMD and, if used, uncertainty for those quantities.

The plotting functions consume the results above. They never derive them.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # File export only; avoids requiring a GUI/Qt backend.
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Ellipse, Patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mrs_validation_toy_figures import (  # noqa: E402
    _draw_figure05_association,
    _draw_figure05_bland_altman,
    _draw_figure05_fit_for_use,
    _draw_figure05_passing_bablok,
    _figure05_inputs,
)

PANEL_FORMATS = (".svg", ".pdf")


# A color-blind-conscious, manuscript-friendly palette.
NAVY = "#173F5F"
BLUE = "#2878B5"
TEAL = "#2A9D8F"
PURPLE = "#7B61A8"
ORANGE = "#E07A5F"
GOLD = "#E9C46A"
GREEN = "#4C956C"
RED = "#C44536"
LIGHT_BLUE = "#DCEAF4"
LIGHT_PURPLE = "#E8E0F0"
LIGHT_GREY = "#EFF2F5"
DARK_GREY = "#344054"


DATA_FILENAMES = {
    1: "figure01_correlation_vs_agreement.csv",
    2: "figure02_nominal_vs_effective_sample_size.csv",
    5: "figure05_method_comparison_pairs.csv",
    6: "figure06_matching_marginals_wrong_pairing.csv",
    7: "figure07_bias_variance_equal_rmse.csv",
    8: "figure08_correct_marginals_wrong_joint_structure.csv",
}


ANALYSIS_TEMPLATES: dict[str, list[str]] = {
    "figure01_metrics.csv": [
        "scenario", "slope", "intercept", "pearson_r", "r_squared",
        "mean_error_mM", "rmse_mM", "lin_ccc",
    ],
    "figure02_metrics.csv": [
        "design", "nominal_n", "independent_clusters", "icc",
        "design_effect", "effective_n", "estimate_mM", "ci_low_mM",
        "ci_high_mM",
    ],
    "figure03_crlb_summary.csv": [
        "condition", "metabolite", "n_replicates", "bias_mM",
        "empirical_sd_mM", "median_crlb_sd_mM", "median_abs_error_mM",
        "median_abs_standardized_error",
    ],
    "figure04_operating_range.csv": [
        "metabolite", "linewidth_hz", "concentration_mM", "snr",
        "n_total", "n_success", "success_probability", "success_lcb",
        "acceptable_abs_error_mM", "evidence_status",
    ],
    "figure05_metrics.csv": [
        "pearson_r", "lin_ccc",
        "pb_intercept_mM", "pb_intercept_ci_low_mM", "pb_intercept_ci_high_mM",
        "pb_slope", "pb_slope_ci_low", "pb_slope_ci_high", "pb_ci_method",
        "pb_cusum_method", "pb_cusum_p_value", "pb_linearity_rejected",
        "ba_x_axis", "ba_location", "ba_scale", "ba_confidence_level",
        "acceptable_lower_mM", "acceptable_upper_mM", "loa_within_acceptable",
    ],
    "figure05_ba_curves.csv": [
        "reference_mM", "bias_mM", "loa_lower_mM", "loa_upper_mM",
    ],
    "figure05_ba_bands.csv": [
        "reference_mM", "bias_ci_low_mM", "bias_ci_high_mM",
        "loa_lower_ci_low_mM", "loa_lower_ci_high_mM",
        "loa_upper_ci_low_mM", "loa_upper_ci_high_mM",
    ],
    "figure05_silveira.csv": [
        "test", "parameter", "estimate", "ci_low", "ci_high",
        "ci_level", "null_value", "rejected",
    ],
    "figure06_metrics.csv": [
        "scenario", "pearson_r", "lin_ccc", "rmse_mM", "mae_mM",
        "marginal_distance", "marginal_test_p",
    ],
    "figure07_metrics.csv": [
        "scenario", "bias_mM", "variance_mM2", "mse_mM2", "rmse_mM",
    ],
    "figure08_metrics.csv": [
        "dataset", "corr_metabolite_a_b", "covariance_a_b",
        "joint_distance", "joint_test_p", "ellipse_center_a",
        "ellipse_center_b", "ellipse_width", "ellipse_height",
        "ellipse_angle_deg",
    ],
}


@dataclass(frozen=True)
class Settings:
    """Reproducible dataset and export settings."""

    seed: int = 20260914
    dpi: int = 300
    metabolite: str = "NAA"
    concentration_unit: str = "mM"


def configure_matplotlib() -> None:
    """Apply a consistent publication-oriented style."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.edgecolor": DARK_GREY,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": DARK_GREY,
            "ytick.color": DARK_GREY,
            "text.color": DARK_GREY,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def ensure_directories(savedir: Path) -> tuple[Path, Path, Path, Path]:
    data_dir = savedir / "data"
    analysis_dir = savedir / "analysis"
    figures_dir = savedir / "figures"
    panels_dir = savedir / "panels"
    for directory in (data_dir, analysis_dir, figures_dir, panels_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return data_dir, analysis_dir, figures_dir, panels_dir


def write_csv(path: Path, rows: Iterable[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path, *, required: bool = True) -> list[dict[str, str]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required file not found: {path}")
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str | None, *, field: str, allow_blank: bool = False) -> float:
    if value is None or value.strip() == "":
        if allow_blank:
            return math.nan
        raise ValueError(f"Missing required numeric value for '{field}'.")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric value for '{field}': {value!r}") from exc


def index_rows(rows: Sequence[Mapping[str, str]], key: str) -> dict[str, Mapping[str, str]]:
    return {str(row[key]): row for row in rows if row.get(key)}


def metric_text(row: Mapping[str, str] | None, fields: Sequence[tuple[str, str]]) -> str:
    """Format supplied values only; no metric is calculated here."""
    if not row:
        return ""
    entries: list[tuple[str, str]] = []
    for field, label in fields:
        value = row.get(field, "")
        if value not in (None, ""):
            entries.append((label, value))
    if not entries:
        return ""
    # add_metric_box right-justifies each line independently (ha="right"),
    # so the "=" only lands in the same place on every line when the text
    # following it is the same width on every line -- the label width is
    # irrelevant under right justification. Padding values to a common
    # width (rather than labels) is what actually keeps the "=" aligned.
    value_width = max(len(value) for _, value in entries)
    return "\n".join(f"{label} = {value.rjust(value_width)}" for label, value in entries)


def add_metric_box(ax: plt.Axes, text: str, x: float = 0.97, y: float = 0.97, ha: str="right", va: str="top") -> None:
    if not text:
        return
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9,
              "edgecolor": "#CBD5E1"},
    )


def save_panel(
    panels_dir: Path,
    figure_number: int,
    slug: str,
    draw: Callable[[plt.Axes], None],
    *,
    figsize: tuple[float, float],
    dpi: int,
    legend: bool = False,
    fixed_size: bool = False,
) -> list[Path]:
    """Render one self-contained subplot to its own SVG and PDF file.

    ``draw`` receives a fresh :class:`~matplotlib.axes.Axes` and must set
    its own title, x/y labels, and (if ``legend`` is True) call
    ``ax.legend()`` -- panels are meant to stand alone once dropped into a
    LaTeX ``subfigure``/``subcaption`` layout, so nothing here should
    depend on sibling panels for context (e.g. a shared y-axis label).

    By default the saved file is cropped tightly to its contents, so its
    final size depends on label/legend extents. With ``fixed_size`` the
    full ``figsize`` canvas is kept, so panels sharing a ``figsize`` are
    saved at exactly the same size and aspect ratio.
    """
    fig, ax = plt.subplots(figsize=figsize)
    draw(ax)
    if legend and ax.get_legend() is None and ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False)
    fig.tight_layout()
    out_dir = panels_dir / f"figure{figure_number:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for suffix in PANEL_FORMATS:
        path = out_dir / f"{slug}{suffix}"
        # bbox_inches=None would fall back to rcParams["savefig.bbox"]
        # ("tight"), so pass the full figure box explicitly instead.
        fig.savefig(path, dpi=dpi, bbox_inches=fig.bbox_inches if fixed_size else "tight")
        saved.append(path)
    plt.close(fig)
    return saved


def generate_figure01(rng: np.random.Generator, path: Path) -> None:
    n = 140
    truth = rng.uniform(1.0, 12.0, n)
    shared_noise = rng.normal(0.0, 0.30, n)
    scenarios = {
        "Accurate": truth + shared_noise,
        "Constant bias": truth + 2.0 + shared_noise,
        "Proportional bias": 1.35 * truth + shared_noise,
    }
    rows = []
    for scenario, prediction in scenarios.items():
        for i, (reference, estimate) in enumerate(zip(truth, prediction, strict=True)):
            rows.append(
                {
                    "sample_id": f"F01-{i:04d}",
                    "scenario": scenario,
                    "ground_truth_mM": f"{reference:.8f}",
                    "prediction_mM": f"{estimate:.8f}",
                }
            )
    write_csv(path, rows, ["sample_id", "scenario", "ground_truth_mM", "prediction_mM"])


def generate_figure02(rng: np.random.Generator, path: Path) -> None:
    rows: list[dict[str, object]] = []
    # Design A: 100 independent subjects.
    independent = rng.normal(5.0, 1.0, 100)
    for i, value in enumerate(independent):
        rows.append(
            {
                "design": "100 independent subjects",
                "subject_id": f"IND-{i:03d}",
                "replicate_id": 0,
                "value_mM": f"{value:.8f}",
                "nominal_n": 100,
                "independent_clusters_by_design": 100,
            }
        )
    # Design B: 10 subjects x 10 repeated observations, with strong clustering.
    subject_effect = rng.normal(0.0, 0.95, 10)
    residual = rng.normal(0.0, 0.28, (10, 10))
    clustered = 5.0 + subject_effect[:, None] + residual
    for subject in range(10):
        for replicate in range(10):
            rows.append(
                {
                    "design": "10 subjects x 10 repeats",
                    "subject_id": f"REP-{subject:03d}",
                    "replicate_id": replicate,
                    "value_mM": f"{clustered[subject, replicate]:.8f}",
                    "nominal_n": 100,
                    "independent_clusters_by_design": 10,
                }
            )
    write_csv(
        path,
        rows,
        ["design", "subject_id", "replicate_id", "value_mM", "nominal_n",
         "independent_clusters_by_design"],
    )


def generate_figure03_manifest(path: Path, settings: Settings) -> None:
    """Write requested MRS-Sim jobs; this function does not simulate spectra."""
    rows: list[dict[str, object]] = []
    conditions = (
        ("well_specified", "matched_basis_and_baseline"),
        ("controlled_model_mismatch", "predeclared_basis_or_baseline_mismatch"),
    )
    n_noisy_replicates = 250
    for condition, fit_model in conditions:
        # One clean anchor per condition.
        rows.append(
            {
                "job_id": f"F03-{condition}-clean",
                "condition": condition,
                "replicate": -1,
                "is_noise_free_anchor": True,
                "add_noise": False,
                "noise_seed": "",
                "metabolite": settings.metabolite,
                "truth_concentration_mM": 5.0,
                "target_snr": "",
                "linewidth_hz": 6.0,
                "noise_model": "none",
                "simulation_model": "same_ground_truth_forward_model",
                "fit_model_variant": fit_model,
            }
        )
        for replicate in range(n_noisy_replicates):
            rows.append(
                {
                    "job_id": f"F03-{condition}-{replicate:04d}",
                    "condition": condition,
                    "replicate": replicate,
                    "is_noise_free_anchor": False,
                    "add_noise": True,
                    "noise_seed": settings.seed + 300_000 + replicate,
                    "metabolite": settings.metabolite,
                    "truth_concentration_mM": 5.0,
                    "target_snr": 20.0,
                    "linewidth_hz": 6.0,
                    "noise_model": "complex_zero_mean_gaussian_define_variance_convention",
                    "simulation_model": "same_ground_truth_forward_model",
                    "fit_model_variant": fit_model,
                }
            )
    fields = [
        "job_id", "condition", "replicate", "is_noise_free_anchor", "add_noise",
        "noise_seed", "metabolite", "truth_concentration_mM", "target_snr",
        "linewidth_hz", "noise_model", "simulation_model", "fit_model_variant",
    ]
    write_csv(path, rows, fields)


def generate_figure04_manifest(path: Path, settings: Settings) -> None:
    """Write a COWS-informed-style MRS-Sim grid; adjust ranges before running."""
    rows: list[dict[str, object]] = []
    concentrations = (1.0, 2.0, 3.5, 5.0, 7.0, 9.5, 12.0)
    snrs = (5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 40.0)
    linewidths = (4.0, 7.0, 10.0)
    replicates_per_cell = 250
    counter = 0
    for linewidth_hz in linewidths:
        for concentration_mM in concentrations:
            for snr in snrs:
                for replicate in range(replicates_per_cell):
                    rows.append(
                        {
                            "job_id": f"F04-{counter:06d}",
                            "cell_id": f"LW{linewidth_hz:g}-C{concentration_mM:g}-SNR{snr:g}",
                            "replicate": replicate,
                            "noise_seed": settings.seed + 400_000 + counter,
                            "metabolite": settings.metabolite,
                            "truth_concentration_mM": concentration_mM,
                            "target_snr": snr,
                            "linewidth_hz": linewidth_hz,
                            "noise_model": "complex_zero_mean_gaussian_define_variance_convention",
                            "acceptable_abs_error_mM": 0.5,
                            "required_success_probability": 0.95,
                            "confidence_level_for_lcb": 0.95,
                        }
                    )
                    counter += 1
    fields = [
        "job_id", "cell_id", "replicate", "noise_seed", "metabolite",
        "truth_concentration_mM", "target_snr", "linewidth_hz", "noise_model",
        "acceptable_abs_error_mM", "required_success_probability",
        "confidence_level_for_lcb",
    ]
    write_csv(path, rows, fields)


def generate_figure05(rng: np.random.Generator, path: Path) -> None:
    n = 150
    reference = rng.uniform(1.0, 12.0, n)
    candidate = 0.35 + 1.10 * reference + rng.normal(0.0, 0.55, n)
    rows = []
    for i, (x, y) in enumerate(zip(reference, candidate, strict=True)):
        rows.append(
            {
                "pair_id": f"F05-{i:04d}",
                "reference_mM": f"{x:.8f}",
                "candidate_mM": f"{y:.8f}",
                # These are plot coordinates, not inferential statistics.
                "pair_mean_mM": f"{0.5 * (x + y):.8f}",
                "candidate_minus_reference_mM": f"{y - x:.8f}",
            }
        )
    write_csv(
        path,
        rows,
        ["pair_id", "reference_mM", "candidate_mM", "pair_mean_mM",
         "candidate_minus_reference_mM"],
    )


def generate_figure06(rng: np.random.Generator, path: Path) -> None:
    n = 140
    truth = np.sort(rng.gamma(shape=5.0, scale=1.1, size=n))
    wrong = rng.permutation(truth)
    rows = []
    for i, (reference, prediction) in enumerate(zip(truth, wrong, strict=True)):
        rows.append(
            {
                "sample_id": f"F06-{i:04d}",
                "scenario": "Permuted predictions",
                "ground_truth_mM": f"{reference:.8f}",
                "prediction_mM": f"{prediction:.8f}",
            }
        )
    write_csv(path, rows, ["sample_id", "scenario", "ground_truth_mM", "prediction_mM"])


def generate_figure07(rng: np.random.Generator, path: Path) -> None:
    n = 1200
    truth = 5.0
    scenarios = {
        "Accurate and precise": rng.normal(truth, 0.18, n),
        "Bias-dominated": rng.normal(truth + 0.75, 0.18, n),
        "Variance-dominated": rng.normal(truth, 0.77, n),
    }
    rows = []
    for scenario, estimates in scenarios.items():
        for i, estimate in enumerate(estimates):
            rows.append(
                {
                    "sample_id": f"F07-{scenario[:3].upper()}-{i:05d}",
                    "scenario": scenario,
                    "truth_mM": truth,
                    "estimate_mM": f"{estimate:.8f}",
                }
            )
    write_csv(path, rows, ["sample_id", "scenario", "truth_mM", "estimate_mM"])


def generate_figure08(rng: np.random.Generator, path: Path) -> None:
    n = 600
    mean = np.array([5.0, 7.0])
    covariance = np.array([[1.4, 1.05], [1.05, 1.8]])
    truth = rng.multivariate_normal(mean, covariance, size=n)
    # Permuting B preserves its exact marginal sample but destroys its pairing with A.
    prediction = np.column_stack([truth[:, 0].copy(), rng.permutation(truth[:, 1])])
    rows = []
    for i in range(n):
        rows.append(
            {
                "sample_id": f"F08-{i:04d}",
                "truth_metabolite_a_mM": f"{truth[i, 0]:.8f}",
                "truth_metabolite_b_mM": f"{truth[i, 1]:.8f}",
                "prediction_metabolite_a_mM": f"{prediction[i, 0]:.8f}",
                "prediction_metabolite_b_mM": f"{prediction[i, 1]:.8f}",
            }
        )
    write_csv(
        path,
        rows,
        ["sample_id", "truth_metabolite_a_mM", "truth_metabolite_b_mM",
         "prediction_metabolite_a_mM", "prediction_metabolite_b_mM"],
    )


def write_mrs_fit_export_template(path: Path) -> None:
    fields = [
        "job_id", "sample_id", "condition", "cell_id", "replicate",
        "metabolite", "truth_concentration_mM", "estimate_mM", "crlb_value",
        "crlb_representation", "crlb_unit", "crlb_reference",
        "target_snr", "measured_snr", "snr_definition", "linewidth_hz",
        "noise_model", "noise_scale", "basis_set_id", "fit_software",
        "fit_software_version", "fit_config_id", "fit_status", "failure_reason",
        "noise_free_anchor_id",
    ]
    write_csv(path, [], fields)


def write_analysis_templates(analysis_dir: Path) -> None:
    for filename, fields in ANALYSIS_TEMPLATES.items():
        path = analysis_dir / filename
        if not path.exists():
            write_csv(path, [], fields)


def generate_all(savedir: Path, settings: Settings) -> None:
    data_dir, analysis_dir, _figures_dir, _panels_dir = ensure_directories(savedir)
    rng = np.random.default_rng(settings.seed)
    generate_figure01(rng, data_dir / DATA_FILENAMES[1])
    generate_figure02(rng, data_dir / DATA_FILENAMES[2])
    generate_figure03_manifest(data_dir / "figure03_mrssim_manifest.csv", settings)
    generate_figure04_manifest(data_dir / "figure04_mrssim_manifest.csv", settings)
    generate_figure05(rng, data_dir / DATA_FILENAMES[5])
    generate_figure06(rng, data_dir / DATA_FILENAMES[6])
    generate_figure07(rng, data_dir / DATA_FILENAMES[7])
    generate_figure08(rng, data_dir / DATA_FILENAMES[8])
    write_mrs_fit_export_template(data_dir / "figures03_04_fitter_export_template.csv")
    write_analysis_templates(analysis_dir)
    provenance = {
        "generator": Path(__file__).name,
        "seed": settings.seed,
        "numpy_rng": "numpy.random.Generator(PCG64 default)",
        "metabolite": settings.metabolite,
        "concentration_unit": settings.concentration_unit,
        "statistics_calculated_by_this_script": False,
        "note": "MRS-Sim manifests are requests, not simulated spectra.",
    }
    (savedir / "generation_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


def plot_figure01(data_dir: Path, analysis_dir: Path, panels_dir: Path, settings: Settings) -> list[Path]:
    rows = read_csv(data_dir / DATA_FILENAMES[1])
    metric_rows = read_csv(analysis_dir / "figure01_metrics.csv", required=False)
    metrics = index_rows(metric_rows, "scenario")
    scenarios = ("Accurate", "Constant bias", "Proportional bias")
    slugs = ("accurate", "constant_bias", "proportional_bias")
    colors = (TEAL, ORANGE, PURPLE)
    all_values = [as_float(r["ground_truth_mM"], field="ground_truth_mM") for r in rows]
    all_values += [as_float(r["prediction_mM"], field="prediction_mM") for r in rows]
    low, high = math.floor(min(all_values)), math.ceil(max(all_values))

    def draw_panel(ax: plt.Axes, scenario: str, color: str) -> None:
        selected = [r for r in rows if r["scenario"] == scenario]
        x = np.array([as_float(r["ground_truth_mM"], field="ground_truth_mM") for r in selected])
        y = np.array([as_float(r["prediction_mM"], field="prediction_mM") for r in selected])
        ax.scatter(x, y, s=19, color=color, alpha=0.68, edgecolor="white", linewidth=0.3)
        ax.plot([low, high], [low, high], color=DARK_GREY, linewidth=1.2,
                linestyle="--", label="Identity")
        row = metrics.get(scenario)
        if row and row.get("slope") and row.get("intercept"):
            slope = as_float(row["slope"], field="slope")
            intercept = as_float(row["intercept"], field="intercept")
            ax.plot([low, high], [intercept + slope * low, intercept + slope * high],
                    color=color, linewidth=2.0, label="Supplied fit")
        add_metric_box(
            ax,
            metric_text(row, (("pearson_r", "r"), ("rmse_mM", "RMSE"),
                              ("lin_ccc", r'$\rho_C$'))),
            x=0.97, y=0.2, ha="right", va="top",
        )
        # ax.set_title(scenario, color=color, fontweight="bold")
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color=LIGHT_GREY, linewidth=0.7)
        ax.set_xlabel("Ground truth (mM)")
        ax.set_ylabel("Prediction (mM)")

    saved: list[Path] = []
    for scenario, slug, color in zip(scenarios, slugs, colors, strict=True):
        saved += save_panel(
            panels_dir, 1, slug,
            lambda ax, scenario=scenario, color=color: draw_panel(ax, scenario, color),
            figsize=(4.2, 3.9), dpi=settings.dpi, legend=True,
        )
    return saved


def plot_figure02(data_dir: Path, analysis_dir: Path, panels_dir: Path, settings: Settings) -> list[Path]:
    rows = read_csv(data_dir / DATA_FILENAMES[2])
    metric_rows = read_csv(analysis_dir / "figure02_metrics.csv", required=False)
    metrics = index_rows(metric_rows, "design")
    designs = ("100 independent subjects", "10 subjects x 10 repeats")
    slugs = ("independent_subjects", "clustered_repeats")
    colors = (BLUE, PURPLE)

    def draw_panel(ax: plt.Axes, design: str, color: str) -> None:
        selected = [r for r in rows if r["design"] == design]
        subject_ids = sorted({r["subject_id"] for r in selected})
        x_map = {subject_id: i for i, subject_id in enumerate(subject_ids)}
        for subject_id in subject_ids:
            subject_rows = [r for r in selected if r["subject_id"] == subject_id]
            x = np.full(len(subject_rows), x_map[subject_id], dtype=float)
            if len(subject_rows) > 1:
                # Deterministic horizontal jitter for visibility only.
                x += np.linspace(-0.22, 0.22, len(subject_rows))
            values = np.array([as_float(r["value_mM"], field="value_mM") for r in subject_rows])
            ax.scatter(x, values, s=20, color=color, alpha=0.72, edgecolor="white", linewidth=0.3)
            if len(values) > 1:
                ax.plot(x, values, color=color, alpha=0.20, linewidth=0.7)
        row = metrics.get(design)
        text = metric_text(
            row,
            (("nominal_n", "Nominal N"), ("independent_clusters", "Clusters"),
             ("icc", "ICC"), ("effective_n", "Effective N")),
        )
        if not text:
            text = f"Nominal N = 100\nIndependent units = {len(subject_ids)}"
        add_metric_box(ax, text)
        # ax.set_title(design, color=color, fontweight="bold")
        ax.set_xlabel("Independent sampling unit")
        ax.set_ylabel("Repeated measurement (mM)")
        ax.set_xticks([])
        ax.grid(axis="y", color=LIGHT_GREY, linewidth=0.7)

    saved: list[Path] = []
    for design, slug, color in zip(designs, slugs, colors, strict=True):
        saved += save_panel(
            panels_dir, 2, slug,
            lambda ax, design=design, color=color: draw_panel(ax, design, color),
            figsize=(5.2, 4.3), dpi=settings.dpi,
        )
    return saved


def plot_figure03(analysis_dir: Path, panels_dir: Path, settings: Settings) -> list[Path]:
    rows = read_csv(analysis_dir / "figure03_crlb_summary.csv")
    if not rows:
        raise ValueError("figure03_crlb_summary.csv is empty; run the stats library first.")
    labels = [f"{r['condition'].replace('_', ' ')}\n{r['metabolite']}" for r in rows]
    empirical_sd = np.array([as_float(r["empirical_sd_mM"], field="empirical_sd_mM") for r in rows])
    crlb_sd = np.array([as_float(r["median_crlb_sd_mM"], field="median_crlb_sd_mM") for r in rows])
    bias = np.array([as_float(r["bias_mM"], field="bias_mM") for r in rows])
    standardized = np.array([
        as_float(r["median_abs_standardized_error"], field="median_abs_standardized_error")
        for r in rows
    ])
    x = np.arange(len(rows))
    width = 0.36

    def draw_precision(ax: plt.Axes) -> None:
        ax.bar(x - width / 2, empirical_sd, width, color=BLUE, label="Empirical SD")
        ax.bar(x + width / 2, crlb_sd, width, color=GOLD, label="Median supplied CRLB SD")
        ax.set_ylabel("Dispersion / bound scale (mM)")
        # ax.set_title("Precision comparison")
        ax.set_xticks(x, labels, rotation=15, ha="right")
        ax.grid(axis="y", color=LIGHT_GREY, linewidth=0.7)

    def draw_accuracy(ax: plt.Axes) -> None:
        ax.bar(x, np.abs(bias), 0.56, color=ORANGE)
        ax.set_ylabel("Absolute empirical bias (mM)")
        # ax.set_title("Accuracy")
        ax.set_xticks(x, labels, rotation=15, ha="right")
        ax.grid(axis="y", color=LIGHT_GREY, linewidth=0.7)

    def draw_relative_error(ax: plt.Axes) -> None:
        ax.bar(x, standardized, 0.56, color=PURPLE)
        ax.set_ylabel("Median |error| / supplied CRLB SD")
        # ax.set_title("Error relative to the supplied bound")
        ax.set_xticks(x, labels, rotation=15, ha="right")
        ax.grid(axis="y", color=LIGHT_GREY, linewidth=0.7)

    saved: list[Path] = []
    for slug, draw in (
        ("precision_comparison", draw_precision),
        ("accuracy", draw_accuracy),
        ("error_relative_to_bound", draw_relative_error),
    ):
        saved += save_panel(panels_dir, 3, slug, draw, figsize=(4.6, 4.2), dpi=settings.dpi, legend=True)
    return saved


def plot_figure04(analysis_dir: Path, panels_dir: Path, settings: Settings,
                  metabolite: str | None, linewidth_hz: float | None) -> list[Path]:
    rows = read_csv(analysis_dir / "figure04_operating_range.csv")
    if not rows:
        raise ValueError("figure04_operating_range.csv is empty; run the stats library first.")
    selected_metabolite = metabolite or rows[0]["metabolite"]
    candidate_lw = sorted({as_float(r["linewidth_hz"], field="linewidth_hz")
                           for r in rows if r["metabolite"] == selected_metabolite})
    if not candidate_lw:
        raise ValueError(f"No operating-range rows for metabolite {selected_metabolite!r}.")
    selected_lw = linewidth_hz if linewidth_hz is not None else candidate_lw[len(candidate_lw) // 2]
    subset = [
        r for r in rows
        if r["metabolite"] == selected_metabolite
        and math.isclose(as_float(r["linewidth_hz"], field="linewidth_hz"), selected_lw)
    ]
    concentrations = sorted({as_float(r["concentration_mM"], field="concentration_mM") for r in subset})
    snrs = sorted({as_float(r["snr"], field="snr") for r in subset})
    if not subset or not concentrations or not snrs:
        raise ValueError("Selected figure-4 slice has no complete grid.")
    lcb = np.full((len(snrs), len(concentrations)), np.nan)
    status = np.full((len(snrs), len(concentrations)), -1, dtype=int)
    status_code = {"outside_validated_range": 0, "unacceptable": 0,
                   "insufficient_evidence": 1, "validated": 2}
    for row in subset:
        i = snrs.index(as_float(row["snr"], field="snr"))
        j = concentrations.index(as_float(row["concentration_mM"], field="concentration_mM"))
        lcb[i, j] = as_float(row["success_lcb"], field="success_lcb")
        supplied_status = row.get("evidence_status", "").strip().lower()
        if supplied_status not in status_code:
            raise ValueError(
                "evidence_status must be validated, insufficient_evidence, "
                "unacceptable, or outside_validated_range."
            )
        status[i, j] = status_code[supplied_status]
    title_suffix = f"{selected_metabolite}, linewidth {selected_lw:g} Hz"

    def format_axes(ax: plt.Axes) -> None:
        ax.set_xticks(np.arange(len(concentrations)), [f"{v:g}" for v in concentrations])
        ax.set_yticks(np.arange(len(snrs)), [f"{v:g}" for v in snrs])
        ax.set_xlabel("Ground-truth concentration (mM)")
        ax.set_ylabel("SNR")

    def draw_evidence_strength(ax: plt.Axes) -> None:
        im = ax.imshow(lcb, origin="lower", aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
        ax.figure.colorbar(im, ax=ax, label="Supplied one-sided lower confidence bound")
        # ax.set_title(f"Strength of evidence: {title_suffix}")
        format_axes(ax)

    def draw_operating_status(ax: plt.Axes) -> None:
        cmap = ListedColormap(["#D95D39", "#E9C46A", "#2A9D8F"])
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
        ax.imshow(status, origin="lower", aspect="auto", cmap=cmap, norm=norm)
        # ax.set_title(f"Evidence-supported operating status: {title_suffix}")
        ax.legend(
            handles=[Patch(color="#2A9D8F", label="Validated"),
                     Patch(color="#E9C46A", label="Insufficient evidence"),
                     Patch(color="#D95D39", label="Unacceptable / outside")],
            frameon=False,
            fontsize=8,
            loc="lower right",
        )
        format_axes(ax)

    saved = save_panel(panels_dir, 4, "evidence_strength", draw_evidence_strength,
                       figsize=(5.6, 4.3), dpi=settings.dpi)
    saved += save_panel(panels_dir, 4, "operating_status", draw_operating_status,
                        figsize=(5.6, 4.3), dpi=settings.dpi)
    return saved


def plot_figure05(data_dir: Path, analysis_dir: Path, panels_dir: Path, settings: Settings) -> list[Path]:
    # The panel drawing is shared with the composite script so the two
    # renderings of Figure 5 cannot drift apart.
    d = _figure05_inputs(data_dir, analysis_dir)

    def with_grid(draw: Callable[[plt.Axes, Mapping[str, object]], None]) -> Callable[[plt.Axes], None]:
        def wrapped(ax: plt.Axes) -> None:
            draw(ax, d)
            ax.grid(color=LIGHT_GREY, linewidth=0.7)
            ax.legend(frameon=False, fontsize=8, loc="upper left")
        return wrapped

    # All four panels share one canvas size (no tight cropping) so they
    # line up when placed side by side as LaTeX subfigures.
    saved: list[Path] = []
    for slug, draw in (
        ("a_association", with_grid(_draw_figure05_association)),
        ("b_bland_altman", with_grid(_draw_figure05_bland_altman)),
        ("c_passing_bablok", with_grid(_draw_figure05_passing_bablok)),
        ("d_fit_for_use", lambda ax: _draw_figure05_fit_for_use(ax, d)),
    ):
        saved += save_panel(panels_dir, 5, slug, draw, figsize=(5.6, 4.6),
                            dpi=settings.dpi, fixed_size=True)
    return saved


def plot_figure06(data_dir: Path, analysis_dir: Path, panels_dir: Path, settings: Settings) -> list[Path]:
    rows = read_csv(data_dir / DATA_FILENAMES[6])
    metric_rows = read_csv(analysis_dir / "figure06_metrics.csv", required=False)
    metric = metric_rows[0] if metric_rows else None
    truth = np.array([as_float(r["ground_truth_mM"], field="ground_truth_mM") for r in rows])
    prediction = np.array([as_float(r["prediction_mM"], field="prediction_mM") for r in rows])
    limits = (math.floor(min(truth.min(), prediction.min())), math.ceil(max(truth.max(), prediction.max())))
    bins = np.linspace(limits[0], limits[1], 20)

    def draw_marginals(ax: plt.Axes) -> None:
        ax.hist(truth, bins=bins, color=TEAL, alpha=0.52, label="Ground truth")
        ax.hist(prediction, bins=bins, histtype="step", color=PURPLE, linewidth=2,
                label="Permuted predictions")
        ax.set(xlabel="Concentration (mM)", ylabel="Count", 
            #    title="Identical marginal samples"
               )
        ax.grid(color=LIGHT_GREY, linewidth=0.7)

    def draw_pairing(ax: plt.Axes) -> None:
        ax.scatter(truth, prediction, s=20, color=PURPLE, alpha=0.62)
        ax.plot(limits, limits, linestyle="--", color=DARK_GREY)
        ax.set(xlabel="Ground truth (mM)", ylabel="Prediction (mM)",
            #    title="Wrong sample-wise pairing", 
               xlim=limits, ylim=limits)
        ax.set_aspect("equal", adjustable="box")
        add_metric_box(
            ax,
            metric_text(metric, (("pearson_r", "r"), ("rmse_mM", "RMSE"),
                                 ("lin_ccc", r'$\rho_C$'))),
        )
        ax.grid(color=LIGHT_GREY, linewidth=0.7)

    saved = save_panel(panels_dir, 6, "marginals", draw_marginals, figsize=(4.8, 3.9),
                       dpi=settings.dpi, legend=True)
    saved += save_panel(panels_dir, 6, "pairing", draw_pairing, figsize=(4.8, 4.4), dpi=settings.dpi)
    return saved


def plot_figure07(data_dir: Path, analysis_dir: Path, panels_dir: Path, settings: Settings) -> list[Path]:
    rows = read_csv(data_dir / DATA_FILENAMES[7])
    metric_rows = read_csv(analysis_dir / "figure07_metrics.csv", required=False)
    metrics = index_rows(metric_rows, "scenario")
    scenarios = ("Accurate and precise", "Bias-dominated", "Variance-dominated")
    slugs = ("accurate_and_precise", "bias_dominated", "variance_dominated")
    colors = (TEAL, ORANGE, PURPLE)
    all_estimates = np.array([as_float(r["estimate_mM"], field="estimate_mM") for r in rows])
    bins = np.linspace(all_estimates.min(), all_estimates.max(), 42)

    def draw_panel(ax: plt.Axes, scenario: str, color: str) -> None:
        selected = [as_float(r["estimate_mM"], field="estimate_mM")
                    for r in rows if r["scenario"] == scenario]
        ax.hist(selected, bins=bins, density=True, color=color, alpha=0.72)
        ax.axvline(5.0, color=DARK_GREY, linestyle="--", linewidth=1.4, label="Truth")
        add_metric_box(
            ax,
            metric_text(metrics.get(scenario), (("bias_mM", "Bias"),
                                                ("variance_mM2", "Variance"),
                                                ("rmse_mM", "RMSE"))),
        )
        # ax.set_title(scenario, color=color, fontweight="bold")
        ax.set_xlabel("Repeated estimate (mM)")
        ax.set_ylabel("Density")
        ax.grid(axis="y", color=LIGHT_GREY, linewidth=0.7)

    saved: list[Path] = []
    for scenario, slug, color in zip(scenarios, slugs, colors, strict=True):
        saved += save_panel(
            panels_dir, 7, slug,
            lambda ax, scenario=scenario, color=color: draw_panel(ax, scenario, color),
            figsize=(4.4, 3.7), dpi=settings.dpi, legend=True,
        )
    return saved


def add_supplied_ellipse(ax: plt.Axes, row: Mapping[str, str], color: str,
                         label: str) -> None:
    required = ("ellipse_center_a", "ellipse_center_b", "ellipse_width",
                "ellipse_height", "ellipse_angle_deg")
    if not all(row.get(field, "") for field in required):
        return
    ellipse = Ellipse(
        xy=(as_float(row["ellipse_center_a"], field="ellipse_center_a"),
            as_float(row["ellipse_center_b"], field="ellipse_center_b")),
        width=as_float(row["ellipse_width"], field="ellipse_width"),
        height=as_float(row["ellipse_height"], field="ellipse_height"),
        angle=as_float(row["ellipse_angle_deg"], field="ellipse_angle_deg"),
        fill=False,
        color=color,
        linewidth=2.2,
        label=label,
    )
    ax.add_patch(ellipse)


def plot_figure08(data_dir: Path, analysis_dir: Path, panels_dir: Path, settings: Settings) -> list[Path]:
    rows = read_csv(data_dir / DATA_FILENAMES[8])
    metric_rows = read_csv(analysis_dir / "figure08_metrics.csv", required=False)
    metrics = index_rows(metric_rows, "dataset")
    ta = np.array([as_float(r["truth_metabolite_a_mM"], field="truth_metabolite_a_mM") for r in rows])
    tb = np.array([as_float(r["truth_metabolite_b_mM"], field="truth_metabolite_b_mM") for r in rows])
    pa = np.array([as_float(r["prediction_metabolite_a_mM"], field="prediction_metabolite_a_mM") for r in rows])
    pb = np.array([as_float(r["prediction_metabolite_b_mM"], field="prediction_metabolite_b_mM") for r in rows])
    bins_a = np.linspace(min(ta.min(), pa.min()), max(ta.max(), pa.max()), 25)
    bins_b = np.linspace(min(tb.min(), pb.min()), max(tb.max(), pb.max()), 25)
    # Shared explicitly (rather than via sharex/sharey) so the two joint-structure
    # panels stay on identical axes once split into separate standalone files.
    joint_xlim = (min(ta.min(), pa.min()), max(ta.max(), pa.max()))
    joint_ylim = (min(tb.min(), pb.min()), max(tb.max(), pb.max()))

    def draw_marginal_a(ax: plt.Axes) -> None:
        ax.hist(ta, bins=bins_a, color=TEAL, alpha=0.5, label="Truth")
        ax.hist(pa, bins=bins_a, histtype="step", color=PURPLE, linewidth=2, label="Prediction")
        ax.set_title("Metabolite A: matching marginal")
        ax.set_xlabel("Metabolite A (mM)")
        ax.set_ylabel("Count")
        ax.grid(color=LIGHT_GREY, linewidth=0.7)

    def draw_marginal_b(ax: plt.Axes) -> None:
        ax.hist(tb, bins=bins_b, color=TEAL, alpha=0.5, label="Truth")
        ax.hist(pb, bins=bins_b, histtype="step", color=PURPLE, linewidth=2, label="Prediction")
        ax.set_title("Metabolite B: matching marginal")
        ax.set_xlabel("Metabolite B (mM)")
        ax.set_ylabel("Count")
        ax.grid(color=LIGHT_GREY, linewidth=0.7)

    def draw_ground_truth_joint(ax: plt.Axes) -> None:
        ax.scatter(ta, tb, s=13, color=TEAL, alpha=0.38)
        add_supplied_ellipse(ax, metrics.get("Ground truth", {}), TEAL, "Supplied ellipse")
        ax.set(xlabel="Metabolite A (mM)", ylabel="Metabolite B (mM)",
            #    title="Ground-truth joint structure", 
               xlim=joint_xlim, ylim=joint_ylim)
        add_metric_box(ax, metric_text(metrics.get("Ground truth"),
                                       (("corr_metabolite_a_b", "Correlation"),)))
        ax.grid(color=LIGHT_GREY, linewidth=0.7)

    def draw_prediction_joint(ax: plt.Axes) -> None:
        ax.scatter(pa, pb, s=13, color=PURPLE, alpha=0.38)
        add_supplied_ellipse(ax, metrics.get("Prediction", {}), PURPLE, "Supplied ellipse")
        ax.set(xlabel="Metabolite A (mM)", ylabel="Metabolite B (mM)",
            #    title="Prediction: dependence destroyed", 
               xlim=joint_xlim, ylim=joint_ylim)
        add_metric_box(ax, metric_text(metrics.get("Prediction"),
                                       (("corr_metabolite_a_b", "Correlation"),
                                        ("joint_distance", "Joint distance"))))
        ax.grid(color=LIGHT_GREY, linewidth=0.7)

    saved = save_panel(panels_dir, 8, "marginal_a", draw_marginal_a, figsize=(4.6, 3.6),
                       dpi=settings.dpi, legend=True)
    saved += save_panel(panels_dir, 8, "marginal_b", draw_marginal_b, figsize=(4.6, 3.6),
                        dpi=settings.dpi, legend=True)
    saved += save_panel(panels_dir, 8, "ground_truth_joint", draw_ground_truth_joint,
                        figsize=(4.8, 4.6), dpi=settings.dpi, legend=True)
    saved += save_panel(panels_dir, 8, "prediction_joint", draw_prediction_joint,
                        figsize=(4.8, 4.6), dpi=settings.dpi, legend=True)
    return saved


VALID_FIGURES = frozenset(range(1, 9))


def parse_figure_selection(values: Sequence[str]) -> list[int]:
    if len(values) == 1 and values[0].lower() == "all":
        return list(range(1, 9))
    figures: list[int] = []
    for value in values:
        figure = int(value)
        if figure not in VALID_FIGURES:
            raise ValueError("Figure numbers must be between 1 and 8.")
        figures.append(figure)
    return sorted(set(figures))


def plot_selected(savedir: Path, figures: Sequence[int], settings: Settings,
                  metabolite: str | None, linewidth_hz: float | None) -> list[Path]:
    data_dir, analysis_dir, _figures_dir, panels_dir = ensure_directories(savedir)
    configure_matplotlib()
    plotters: dict[int, Callable[[], list[Path]]] = {
        1: lambda: plot_figure01(data_dir, analysis_dir, panels_dir, settings),
        2: lambda: plot_figure02(data_dir, analysis_dir, panels_dir, settings),
        3: lambda: plot_figure03(analysis_dir, panels_dir, settings),
        4: lambda: plot_figure04(analysis_dir, panels_dir, settings, metabolite, linewidth_hz),
        5: lambda: plot_figure05(data_dir, analysis_dir, panels_dir, settings),
        6: lambda: plot_figure06(data_dir, analysis_dir, panels_dir, settings),
        7: lambda: plot_figure07(data_dir, analysis_dir, panels_dir, settings),
        8: lambda: plot_figure08(data_dir, analysis_dir, panels_dir, settings),
    }
    made: list[Path] = []
    skipped: list[int] = []
    for figure in figures:
        # A figure whose external analysis results are not yet available is
        # skipped (with a warning) so the remaining figures still render.
        try:
            made += plotters[figure]()
        except (FileNotFoundError, ValueError) as exc:
            print(f"Warning: skipping figure {figure}: {exc}", file=sys.stderr)
            skipped.append(figure)
    if skipped and len(skipped) == len(figures):
        raise ValueError("No figures could be plotted; see warnings above.")
    return made


def describe() -> None:
    print(__doc__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate datasets/manifests and export standalone per-subplot panels "
                     "for eight MRS validation toy figures."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate", help="Generate toy data and templates.")
    generate_parser.add_argument("--savedir", type=Path, required=True,
                                 help="Output root selected by the caller.")
    generate_parser.add_argument("--seed", type=int, default=Settings.seed)
    generate_parser.add_argument("--metabolite", default=Settings.metabolite)
    plot_parser = subparsers.add_parser(
        "plot", help="Export standalone SVG/PDF panels under <savedir>/panels/."
    )
    plot_parser.add_argument("--savedir", type=Path, required=True,
                             help="Root containing data/ and analysis/ directories.")
    plot_parser.add_argument("--figures", nargs="+", default=["all"],
                             help="Figure numbers 1..8 or 'all'.")
    plot_parser.add_argument("--dpi", type=int, default=Settings.dpi)
    plot_parser.add_argument("--metabolite", default=None,
                             help="Figure-4 metabolite slice; default is first available.")
    plot_parser.add_argument("--linewidth-hz", type=float, default=None,
                             help="Figure-4 linewidth slice; default is middle available.")
    subparsers.add_parser("describe", help="Print metric and MRS-Sim contracts.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "describe":
        describe()
        return 0
    if args.command == "generate":
        settings = Settings(seed=args.seed, metabolite=args.metabolite)
        generate_all(args.savedir.resolve(), settings)
        print(f"Generated datasets, manifests, and templates under: {args.savedir.resolve()}")
        return 0
    if args.command == "plot":
        settings = Settings(dpi=args.dpi)
        figures = parse_figure_selection(args.figures)
        made = plot_selected(args.savedir.resolve(), figures, settings,
                             args.metabolite, args.linewidth_hz)
        for path in made:
            print(path)
        return 0
    raise AssertionError("Unreachable command state")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
