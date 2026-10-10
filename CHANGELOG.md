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

### Changed (manuscript-consistency corrections)

Each of these changes numeric results under default settings. The
corresponding manuscript edits are tracked in
`assets/dev/MANUSCRIPT_UPDATES.md`.

- **Cross-classified design effect** (`clustering.cross_classified_icc`):
  each ICC is now weighted by the size of the clusters sharing that
  random effect (observations per metabolite and per subject,
  size-weighted mean `sum(N_c^2) / N_tot`), and `n_eff = N_tot / deff`.
  The previous formula (copied from the manuscript's `eqn:deff_cc`)
  paired each ICC with the wrong cluster size and overstated `n_eff`
  about ten-fold in simulation. New result fields
  `metabolite_cluster_size`, `subject_cluster_size`.
- **ECWAS critical values** (`distribution.ecwas_kossler_mukherjee`):
  the default is now the joint permutation null
  (`critical_value_method="monte_carlo"`, 10 000 relabelings, also the
  new default for `ecwas_manuscript_max_shape`). The three components
  are correlated under H0, so the former chi2(3) default was
  anti-conservative (7.4 % type-I error at nominal 5 %, against 4.6 %
  for the permutation null). chi2(3) remains available for comparison.
- **Bias-corrected MAPE** (`errors.bias_corrected_mape`): now
  `100 * mean|e_n - MPE/100|`, centering the signed relative error
  before the absolute value. The previous definitions subtracted MPE
  from the absolute error, which reduces to MAPE - MPE and doubles a
  negative bias. The `definition` argument was removed.
- **MeAPE** (`errors.meape`): the default is now `"conventional"`
  (`100 * median|e_n|`), without the stray `1/N` factor. The
  bias-corrected median is `definition="bias_corrected"`,
  `100 * median|e_n - MePE/100|`. The `manuscript_literal` and
  `manuscript_literal_bias_corrected` definitions were removed.
- **Bootstrap intervals** (`resampling.bootstrap_ci`): the default
  `method` is now `"percentile"`, the manuscript's interval; `"BCa"` and
  `"basic"` remain available.

### Changed (model-level mixed-effects equivalence tests)

- **Small-sample cluster-robust standard errors** (`mixed_effects`
  Tests 1-2): new `small_sample={"auto","CR0","CR1","CR2"}`, default
  `"auto"`: CR2 with Bell-McCaffrey degrees of freedom below 50 clusters,
  CR1 with a t(G-1) reference otherwise. p-values now use t(df) instead
  of a normal z (CR0 keeps z). Fields `alpha_z`/`beta_z` renamed to
  `alpha_t`/`beta_t`; new `se_method`, `df`; Test 2 also reports
  `tau_alpha_sq`. At 10 clusters, rejection of a true null at nominal 5 %
  fell from 11-18 % (CR0 + z) to 4-5 % (CR2).
- **Test 3 is now a two-stage mixed-effects Deming model**
  (`mixed_equivalence_test_3`, breaking signature change): per-metabolite
  York regression with per-observation error variances (reduces exactly
  to Deming for constant variances), REML random-effects synthesis of
  intercepts and slopes, and a cluster bootstrap over spectra/subjects.
  Reports the fixed `alpha`, `beta`, `tau_alpha_sq`, `tau_beta_sq` with
  intervals and the per-metabolite estimates. New required argument
  `metabolite_ids`; error variances accept scalars or per-observation
  arrays. The previous implementation was a single pooled Deming fit
  that ignored metabolite.
- **Inference scope** (`mixed_equivalence_test_3(inference=...)`):
  default `"panel"` (inference conditional on the evaluated joint
  metabolite set; tau^2 descriptive); `"population"` uses a two-way
  pigeonhole bootstrap over metabolites and clusters for generalisation
  beyond the panel (conservative when tau^2 is small).

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
- Narrative documentation site (`docs/`); two gallery examples exist
  in `docs/source/examples/`.
- PyPI publication / OIDC Trusted Publishing configuration (requires
  repository-admin action).
