.. mrsvalidate documentation master file

##################################################
mrsvalidate: MRS/Metabolite Model Evaluation
##################################################

**mrsvalidate** is a scientific analysis and visualization library for
rigorously evaluating deep-learning-based model fitting and metabolite
quantification in Magnetic Resonance Spectroscopy (MRS). It accompanies
the *"Laying the Foundations"* manuscript on validation methodology for
DL-based MRS quantification.

The library is built around one non-negotiable architectural rule:

    The scientific analysis layer computes all metrics, statistical
    tests, confidence intervals, uncertainty diagnostics, agreement
    analyses, operating envelopes, and standardized quantities. The
    visualization layer receives those precomputed result objects and
    renders them — it never recomputes a scientific quantity.

This separation exists so that the number reported in a table,
serialized to disk, or shown on a figure always comes from the exact
same analysis result object, which is essential for reproducibility in
research and clinical evaluation contexts.

Who this is for
================

- **MRS/MRI researchers** evaluating a new deep-learning quantification
  or spectral-fitting model against ground truth or an established
  reference method.
- **Clinical scientists** assessing whether a model's outputs agree
  sufficiently with a conventional fitting algorithm for a specific
  intended use (method-comparison, equivalence testing).
- **Method developers** who need reproducible, auditable statistics —
  every result carries explicit provenance (status, exclusions,
  configuration, random seed) rather than a bare number.

What it computes
==================

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Descriptive & error statistics
      :link: api
      :link-type: doc

      Mean/median/robust variance, IQR, CV, MPE/MAPE/MeAPE (with
      manuscript-literal and corrected definition modes), RMSE/RMeSE.

   .. grid-item-card:: Regression & agreement
      :link: api
      :link-type: doc

      Pearson *r*, Theil-Sen, direct/adjusted :math:`R^2`, Lin's
      concordance correlation coefficient, multivariate CCC.

   .. grid-item-card:: Inference & distribution testing
      :link: api
      :link-type: doc

      Bootstrap/permutation resampling, effective sample size (ICC),
      the ECWAS distributional-comparison family, a full normality-test
      suite.

   .. grid-item-card:: Method comparison
      :link: api
      :link-type: doc

      Bland-Altman, Passing-Bablok, Deming regression, and the Silveira
      et al. equivalence-testing protocol (per-metabolite and
      model-level).

   .. grid-item-card:: Uncertainty & information
      :link: api
      :link-type: doc

      NLPD, calibration, cobeau, external-CRLB-standardized error, the
      robust operating envelope.

   .. grid-item-card:: Monte Carlo & response curves
      :link: api
      :link-type: doc

      Hierarchical Monte Carlo failure-mode characterization and 1D/2D
      response-curve analysis (e.g. error vs. mSNR and concentration).

Quick example
===============

.. code-block:: python

    import numpy as np
    from mrsvalidate import (
        ObservationSet, ParameterSchema, ParameterSpec, ParameterRole,
        align_observations,
    )
    from mrsvalidate.analysis.regression import regression_metrics
    from mrsvalidate.analysis.agreement import lin_ccc

    schema = ParameterSchema((ParameterSpec("NAA", ParameterRole.CONCENTRATION, "mM"),))
    sample_ids = np.array([f"s{i}" for i in range(100)])

    reference = ObservationSet(sample_ids=sample_ids, values=ref_values, parameters=schema)
    candidate = ObservationSet(sample_ids=sample_ids, values=pred_values, parameters=schema)

    aligned = align_observations(reference, candidate, pairing="paired")
    result = regression_metrics(
        aligned.reference.column("NAA"), aligned.candidate.column("NAA"), aligned.reference.sample_ids,
    )
    print(result.r, result.r_squared_direct)

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: User Guide

   installation
   auto_examples/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Reference

   api

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Project

   GitHub Repository <https://github.com/JohnLaMaster/MRSValidate>
   Changelog <https://github.com/JohnLaMaster/MRSValidate/blob/main/CHANGELOG.md>

Indices
========

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
