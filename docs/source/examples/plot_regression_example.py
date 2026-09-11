"""
Regression metrics and the predicted-vs-reference scatter plot
================================================================

This example simulates a small synthetic metabolite-concentration
dataset, computes the full regression/agreement metric set via
:func:`mrsvalidate.analysis.regression.regression_metrics`, and renders
it with :func:`mrsvalidate.viz.plot_regression_result` — which consumes
the already-computed result object and recomputes nothing.
"""

# %%
# Simulate a synthetic reference/prediction pair
# -----------------------------------------------
#
# A small amount of proportional and additive noise is added to a
# noise-free "ground truth" concentration to stand in for a model's
# predictions.

import numpy as np

from mrsvalidate.analysis.regression import regression_metrics
from mrsvalidate.viz import plot_regression_result

rng = np.random.default_rng(0)
n_samples = 120
reference = rng.uniform(1.0, 10.0, size=n_samples)
prediction = 0.97 * reference + rng.normal(scale=0.35, size=n_samples)
sample_ids = np.array([f"s{i}" for i in range(n_samples)])

# %%
# Compute the regression result
# --------------------------------
#
# A single call returns Pearson's r, direct/adjusted :math:`R^2`, the
# Theil-Sen fit, and both concise-score formulations together, with
# full provenance in ``result.meta``.

result = regression_metrics(reference, prediction, sample_ids)
print(f"r = {result.r:.3f}, R^2 = {result.r_squared_direct:.3f}, "
      f"slope = {result.slope:.3f}, intercept = {result.intercept:.3f}")

# %%
# Render the result
# --------------------
#
# The plot draws the identity line and the fitted Theil-Sen line from
# fields already present on ``result`` — it never recomputes the
# regression.

fig, ax = plot_regression_result(result)
fig.suptitle("Synthetic NAA concentration: prediction vs. reference")
