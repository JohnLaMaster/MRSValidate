# Changelog

All notable changes to `mrsvalidate` are documented in this file.
Scientific-definition changes (a metric's formula, denominator
convention, or status semantics) are called out explicitly, per
`LIBRARY_IMPLEMENTATION_SPEC.md`.

## [0.1.0.dev0] — Unreleased

Initial implementation pass, covering the Foundation, Core Numerical,
Inference, Probabilistic/Information, and Method Comparison milestones
from `LIBRARY_IMPLEMENTATION_SPEC.md`.

### Added

- `schema`, `validation`, `alignment`, `missingness`, `results`,
  `_numerics`: the package's structural/scientific contract
  (`ObservationSet`, `ParameterSchema`, `align_observations`,
  `ResultStatus`/`AnalysisMeta`, near-zero policy).
- `analysis.descriptive`, `analysis.errors`: mean/median/variance/
  robust variance/IQR/CV, MPE/MePE/MAPE/MeAPE (manuscript-literal and
  corrected/conventional definition modes), RMSE/RMeSE.
- `analysis.snr`: spectral and power SNR (manuscript first-equality
  RMS-power definition).
- `analysis.standardization`: reference-derived normal/robust
  standardization.
- `analysis.regression`, `analysis.agreement`: Pearson r (SciPy
  reference-tested) and median-centered r, direct/adjusted R²,
  original/manuscript concise scores, Theil-Sen (SciPy `method="joint"`,
  reference-tested), Lin's CCC and the manuscript's experimental robust
  CCC.
- `analysis.resampling`: paired/independent bootstrap (SciPy),
  clustered bootstrap, and permutation testing with an explicit
  `design` argument selecting SciPy's `permutation_type`.
- `analysis.clustering`, `analysis.sample_size`: one-way and
  cross-classified REML ICC / design effect / effective sample size.
- `analysis.distribution`: the ECWAS family (Kössler-Mukherjee and the
  manuscript's Funato-style max-shape extension), implemented from
  first-principles linear rank-statistic theory.
- `analysis.probabilistic`, `analysis.error_consistency`,
  `analysis.calibration`, `analysis.uncertainty`: NLPD (marginal, macro,
  and full-covariance joint), error consistency, expected calibration
  error, and cobeau (with Fisher-z macro aggregation).
- `analysis.information`, `analysis.operating_envelope`: the external
  CRLB contract (`UncertaintyRepresentation`), standardized error,
  Mahalanobis squared error (Cholesky solve, optional pseudoinverse
  extension), and the robust operating envelope (Clopper-Pearson,
  per-stratum allocation).
- `analysis.method_comparison`, `analysis.mixed_effects`,
  `analysis.multivariate`: Bland-Altman (constant and conditional
  heteroscedastic limits of agreement), Passing-Bablok, Deming, the
  Silveira et al. three-part equivalence protocol (per-metabolite and
  model-level/mixed-effects), and the multivariate CCC.
- `viz.scatter`, `viz.agreement`, `viz.calibration`, `viz.information`:
  plots that consume precomputed result objects only, with a static
  AST check and dynamic monkeypatch tests enforcing that `viz` never
  calls into `analysis`.
- `io`, `pipeline`: CSV/JSON/NPZ export, a self-describing run
  manifest, and a declarative batch runner (`run_analysis_plan`).

### Added (follow-up pass)

- `analysis.normality`: Shapiro-Wilk, Kolmogorov-Smirnov (fully
  specified null), Lilliefors, Anderson-Darling, and D'Agostino-Pearson,
  each wrapping a mature SciPy/Statsmodels primitive and reporting a
  critical value — from a bundled table (Anderson-Darling via SciPy's
  own Stephens-1974 values; Lilliefors via Statsmodels' internal
  10M-replicate simulated table) where one exists, from an exact
  equation (D'Agostino-Pearson: `chi2.ppf`; classical KS: the
  asymptotic Kolmogorov distribution) where the null distribution is
  known analytically, and by Monte Carlo simulation (Shapiro-Wilk, no
  bundled table available) otherwise.
- ECWAS (`analysis.distribution`): both `ecwas_*` functions now report
  `critical_value`, `p_value`, and `reject_null`, selectable via
  `critical_value_method` — `"chi2_asymptotic"` (`chi2.ppf(.., df=3)`,
  the standard reference for a sum of three asymptotically independent
  squared standardized statistics) or `"monte_carlo"` (a joint
  permutation null of the whole composite, the default for the
  max-shape variant since its shape component is non-normal even
  asymptotically).
- `analysis.monte_carlo`: the manuscript's hierarchical Monte Carlo
  failure-mode characterization (`generate_noise_realizations`,
  `monte_carlo_condition`, `monte_carlo_characterization`), producing
  per-condition bias/variance/RMSE and (given an external CRLB
  reference) the variability ratio/excess variance, retained
  per-condition before aggregation for response-curve analysis.
- `analysis.response_curves` / `viz.response_curves`: 1D/2D
  response-curve binning and plotting for information-standardized
  failure-mode characterization (e.g. outcome vs. mSNR, or vs. mSNR and
  concentration jointly), with the axis/axes specified explicitly by
  the caller.
- `schema.ParameterSchema.index_by_role_and_metabolite` /
  `ObservationSet.column_for_metabolite_role`: look up one metabolite's
  parameter of a given role (e.g. NAA's Lorentzian linewidth) by schema
  metadata rather than an assumed column position.

### Known scientific-definition notes

- **Deming regression**: the manuscript's displayed slope equation, if
  combined literally with its own displayed intercept equation, does
  not reproduce a consistent line even for noiseless data. This
  implementation uses the algebraically self-consistent form (verified
  against the noiseless closed-form solution) rather than the literal
  juxtaposition of both displayed equations — see the docstring in
  `analysis/method_comparison.py::deming`.
- **Theil-Sen direction**: implemented as `reference = slope *
  prediction + intercept` (SciPy `theilslopes(reference, prediction,
  method="joint")`), matching the manuscript's stated x/y roles for
  that equation, which is the opposite of the "prediction vs. true"
  direction used by some plotting conventions elsewhere in the
  manuscript. Recorded explicitly in `RegressionResult.method_metadata`.

### Not yet implemented

- Full visualization coverage (correlation heatmap, violin
  distributions, SNR-dependence plots, uncertainty/entropy plots,
  Passing-Bablok plot) — the architectural pattern and boundary
  enforcement are in place; additional figure families follow the same
  pattern.
- Narrative documentation site (`docs/`) and worked examples
  (`examples/`).
- PyPI publication / OIDC Trusted Publishing configuration (requires
  repository-admin action).
- Final `LICENSE` selection (deferred pending the provenance/dependency
  review described in `THIRD_PARTY_CODE.md`).
