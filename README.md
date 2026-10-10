# MRSValidate

`mrsvalidate` is a Python library for the statistical evaluation of
quantification methods in magnetic resonance spectroscopy (MRS), with a
focus on deep-learning-based spectral fitting and metabolite
quantification. It implements the validation framework described in
*Laying the Foundation for the Clinical Translation of Deep
Learning-based Spectral Fitting Models in Magnetic Resonance
Spectroscopy* (LaMaster et al., 2026).

Given model estimates, reference (ground-truth) values and, optionally,
uncertainty estimates or Cramér–Rao lower bounds (CRLBs), the library
provides metrics and tests for accuracy, agreement, uncertainty
calibration and operating range. Every analysis returns a typed result
object that records which samples were included, which were excluded and
why, and whether the result is statistically usable.

## Features

- **Error and descriptive statistics:** bias, MPE/MePE, MAPE/MeAPE,
  RMSE, coefficient of variation, and robust (median-based) alternatives.
- **Regression and agreement:** Pearson's *r*, direct and adjusted
  *R*², Theil–Sen regression, Lin's concordance correlation coefficient
  (univariate and multivariate), and concise performance scores.
- **Method comparison:** Bland–Altman analysis (including
  heteroscedastic limits of agreement), Passing–Bablok and Deming
  regression, and a three-part equivalence protocol for accuracy,
  precision and agreement.
- **Inference with dependent data:** paired, independent and clustered
  bootstrap; permutation tests; ICC, design effect and effective sample
  size; mixed-effects models.
- **Distributions:** normality tests (Shapiro–Wilk, Kolmogorov–Smirnov,
  Lilliefors, Anderson–Darling, D'Agostino–Pearson) and ECWAS
  distribution-comparison tests.
- **Uncertainty and information bounds:** NLPD, expected calibration
  error, error consistency, CRLB-standardized errors, Mahalanobis error,
  and Monte Carlo failure-mode characterization.
- **Operating range:** cell-wise success probabilities with
  Clopper–Pearson lower bounds over, e.g., SNR × concentration grids,
  and response curves against metabolite SNR.
- **Reproducibility:** a declarative batch pipeline
  (`run_analysis_plan`) with CSV/JSON/NPZ export and a run manifest.
- **Plotting (optional):** regression, agreement, calibration and
  response-curve plots built from the result objects.

## Installation

`mrsvalidate` requires Python 3.11 or later. It is not yet published on
PyPI. Install the latest version from GitHub:

```bash
pip install "mrsvalidate @ git+https://github.com/JohnLaMaster/MRSValidate.git"
```

Optional extras: `viz` (plotting, adds Matplotlib) and `pandas`. For
development, clone the repository and install it in editable mode:

```bash
git clone https://github.com/JohnLaMaster/MRSValidate.git
cd MRSValidate
pip install -e ".[dev]"
pytest
```

## Quick example

Compare simulated NAA estimates against their ground truth:

```python
import numpy as np
from mrsvalidate import ObservationSet, ParameterSchema, ParameterSpec, ParameterRole, align_observations
from mrsvalidate.analysis.regression import regression_metrics
from mrsvalidate.analysis.agreement import lin_ccc

# One concentration parameter (NAA, in mM) for 100 samples.
schema = ParameterSchema((ParameterSpec("NAA", ParameterRole.CONCENTRATION, "mM"),))
sample_ids = np.array([f"s{i}" for i in range(100)])
reference_values = np.random.default_rng(0).uniform(1, 10, size=100)
predicted_values = reference_values + np.random.default_rng(1).normal(scale=0.3, size=100)

reference = ObservationSet(sample_ids=sample_ids, values=reference_values.reshape(-1, 1), parameters=schema)
candidate = ObservationSet(sample_ids=sample_ids, values=predicted_values.reshape(-1, 1), parameters=schema)

# Pair estimates with reference values by sample ID.
aligned = align_observations(reference, candidate, pairing="paired")
ref_col = aligned.reference.column("NAA")
pred_col = aligned.candidate.column("NAA")
ids = aligned.reference.sample_ids

# Association: Pearson's r, direct R², and the Theil-Sen fit
# (reference = slope * prediction + intercept).
result = regression_metrics(ref_col, pred_col, ids)
print(result.r, result.r_squared_direct, result.slope, result.intercept)

# Agreement with the identity line: Lin's concordance correlation coefficient.
concordance = lin_ccc(ref_col, pred_col, ids)
print(concordance.rho_c)
```

## Project status

`mrsvalidate` is in **pre-release (version 0.1.0.dev0, alpha)**. The
statistical methods listed above are implemented and covered by unit,
property-based, reference and integration tests. The reference tests
check results against SciPy and Statsmodels where an equivalent exists.
CI runs on Python 3.11 and 3.12 (Linux), and on Python 3.11 (macOS and
Windows).

Until the first stable release:

- the public API may change between versions;
- the documentation at <https://mrsvalidate.readthedocs.io> is
  incomplete;
- plotting covers only part of the figure types used in the manuscript;
- the package is not yet available on PyPI.

Where the implementation intentionally departs from a formula as
printed in the manuscript (for example, Deming regression), the
difference is documented in [`CHANGELOG.md`](CHANGELOG.md) and in the
relevant docstring.

## Reporting bugs and contributing

If you find an error in a statistical method, a bug, or a discrepancy
with the manuscript, please report it by opening a pull request on
[GitHub](https://github.com/JohnLaMaster/MRSValidate). A pull request
should include:

- a clear description of the problem and the expected behavior;
- a minimal, self-contained example that reproduces it;
- your Python version, operating system, and the versions of
  `mrsvalidate`, NumPy and SciPy;
- for statistical issues, a reference (publication, textbook, or
  established implementation) supporting the expected result;
- where possible, the proposed fix together with a test that fails
  without it.

## Citation

If you use `mrsvalidate` in your research, please cite the accompanying
article. Citation metadata is provided in [`CITATION.cff`](CITATION.cff),
and GitHub's "Cite this repository" button generates the corresponding
reference.

## License

`mrsvalidate` is released under the [MIT License](LICENSE). Licenses of
third-party dependencies are listed in
[`THIRD_PARTY_LICENSES.md`](assets/dev/THIRD_PARTY_LICENSES.md).
