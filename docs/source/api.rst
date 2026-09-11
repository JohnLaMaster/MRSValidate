.. _api:

###############
API Reference
###############

This reference is organized by subpackage, following the package's own
architecture: the **core** contract (schema, validation, alignment,
result types) that every analysis is built on, the **analysis** layer
(the actual statistics — never called from ``viz``), the
**visualization** layer (renders precomputed results only), and
**I/O + pipeline** (serialization and batch orchestration).

Core
======

.. currentmodule:: mrsvalidate

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ObservationSet
   ParameterSpec
   ParameterSchema
   ParameterRole
   ResultStatus
   AnalysisMeta
   AnalysisResult
   align_observations
   AlignedComparison
   AlignmentReport
   run_analysis_plan

.. currentmodule:: mrsvalidate.results

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ScalarMetricResult
   IntervalResult
   PerSampleResult
   MetricCollectionResult
   build_meta
   is_usable

.. currentmodule:: mrsvalidate.validation

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ValidationReport
   validate_observation_set
   find_duplicate_ids
   require_role
   require_non_empty

.. currentmodule:: mrsvalidate._numerics

.. autosummary::
   :toctree: generated/
   :nosignatures:

   near_zero_threshold
   is_near_zero
   robust_variance
   robust_sd
   iqr
   tukey_outlier_range

Analysis: descriptive & error metrics
========================================

.. currentmodule:: mrsvalidate.analysis.descriptive

.. autosummary::
   :toctree: generated/
   :nosignatures:

   mean
   median
   variance
   robust_variance
   robust_sd
   iqr
   tukey_outlier_range
   cv
   robust_cv

.. currentmodule:: mrsvalidate.analysis.errors

.. autosummary::
   :toctree: generated/
   :nosignatures:

   mpe
   mepe
   mape
   bias_corrected_mape
   meape
   rmse
   rmese

.. currentmodule:: mrsvalidate.analysis.snr

.. autosummary::
   :toctree: generated/
   :nosignatures:

   spectral_snr
   power_snr
   spectral_snr_from_spectra

.. currentmodule:: mrsvalidate.analysis.standardization

.. autosummary::
   :toctree: generated/
   :nosignatures:

   standardize
   StandardizationResult

Analysis: regression, agreement & multivariate
=================================================

.. currentmodule:: mrsvalidate.analysis.regression

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pearson_r
   median_centered_correlation
   r_squared_direct
   adjusted_r_squared
   concise_score_original
   concise_score_manuscript
   theil_sen
   regression_metrics
   CorrelationResult
   TheilSenResult
   RegressionResult

.. currentmodule:: mrsvalidate.analysis.agreement

.. autosummary::
   :toctree: generated/
   :nosignatures:

   lin_ccc
   robust_ccc
   ConcordanceResult

.. currentmodule:: mrsvalidate.analysis.multivariate

.. autosummary::
   :toctree: generated/
   :nosignatures:

   multivariate_ccc
   MultivariateResult

Analysis: resampling, clustering & distribution
==================================================

.. currentmodule:: mrsvalidate.analysis.resampling

.. autosummary::
   :toctree: generated/
   :nosignatures:

   bootstrap_ci
   cluster_bootstrap_ci
   permutation_test
   BootstrapResult
   PermutationResult

.. currentmodule:: mrsvalidate.analysis.clustering

.. autosummary::
   :toctree: generated/
   :nosignatures:

   one_way_icc
   cross_classified_icc
   EffectiveSampleSizeResult
   CrossClassifiedEffectiveSampleSizeResult

.. currentmodule:: mrsvalidate.analysis.sample_size

.. autosummary::
   :toctree: generated/
   :nosignatures:

   check_minimum_sample_size
   check_minimum_clusters

.. currentmodule:: mrsvalidate.analysis.distribution

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ecwas_kossler_mukherjee
   ecwas_manuscript_max_shape
   ECWASResult

.. currentmodule:: mrsvalidate.analysis.normality

.. autosummary::
   :toctree: generated/
   :nosignatures:

   shapiro_wilk
   kolmogorov_smirnov
   lilliefors
   anderson_darling
   dagostino_pearson
   normality_test_suite
   NormalityTestResult

Analysis: probabilistic, calibration & uncertainty
=====================================================

.. currentmodule:: mrsvalidate.analysis.probabilistic

.. autosummary::
   :toctree: generated/
   :nosignatures:

   nlpd
   nlpd_macro
   joint_nlpd_full_covariance

.. currentmodule:: mrsvalidate.analysis.error_consistency

.. autosummary::
   :toctree: generated/
   :nosignatures:

   error_consistency
   ErrorConsistencyResult

.. currentmodule:: mrsvalidate.analysis.calibration

.. autosummary::
   :toctree: generated/
   :nosignatures:

   expected_calibration_error
   CalibrationResult

.. currentmodule:: mrsvalidate.analysis.uncertainty

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cobeau
   cobeau_fisher_aggregate
   CobeauResult

.. currentmodule:: mrsvalidate.analysis.information

.. autosummary::
   :toctree: generated/
   :nosignatures:

   UncertaintyRepresentation
   to_std
   crlb_sd_from_covariance
   standardized_error
   variance_ratio
   mahalanobis_squared_error
   VarianceRatioResult

.. currentmodule:: mrsvalidate.analysis.operating_envelope

.. autosummary::
   :toctree: generated/
   :nosignatures:

   operating_envelope
   OperatingEnvelopeResult

Analysis: method comparison
==============================

.. currentmodule:: mrsvalidate.analysis.method_comparison

.. autosummary::
   :toctree: generated/
   :nosignatures:

   bland_altman
   passing_bablok
   deming
   equivalence_test_1
   equivalence_test_2
   equivalence_test_3
   BlandAltmanResult
   PassingBablokResult
   DemingResult
   EquivalenceTest1Result
   EquivalenceTest2Result
   EquivalenceTest3Result

.. currentmodule:: mrsvalidate.analysis.mixed_effects

.. autosummary::
   :toctree: generated/
   :nosignatures:

   mixed_equivalence_test_1
   mixed_equivalence_test_2
   mixed_equivalence_test_3
   MixedTest1Result
   MixedTest2Result
   MixedTest3Result

Analysis: Monte Carlo & response curves
===========================================

.. currentmodule:: mrsvalidate.analysis.monte_carlo

.. autosummary::
   :toctree: generated/
   :nosignatures:

   generate_noise_realizations
   monte_carlo_condition
   monte_carlo_characterization
   condition_spectral_snr
   MonteCarloConditionResult
   MonteCarloCharacterizationResult

.. currentmodule:: mrsvalidate.analysis.response_curves

.. autosummary::
   :toctree: generated/
   :nosignatures:

   response_curve
   ResponseCurveResult

Visualization
===============

.. currentmodule:: mrsvalidate.viz

.. autosummary::
   :toctree: generated/
   :nosignatures:

   plot_regression_result
   plot_bland_altman
   plot_calibration_result
   plot_operating_envelope
   plot_response_curve

I/O
=====

.. currentmodule:: mrsvalidate.io

.. autosummary::
   :toctree: generated/
   :nosignatures:

   observation_set_from_columns
   observation_set_from_csv
   export_result_json
   export_results_csv
   export_arrays_npz
   result_to_jsonable
   build_manifest
   config_digest

Pipeline
==========

.. currentmodule:: mrsvalidate.pipeline

.. autosummary::
   :toctree: generated/
   :nosignatures:

   AnalysisStep
   RunConfig
   RunResult
   run_analysis_plan
   ProgressReporter
   NullProgressReporter
   PrintProgressReporter
