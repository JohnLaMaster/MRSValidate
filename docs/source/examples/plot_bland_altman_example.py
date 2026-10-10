"""
Bland-Altman method comparison
================================

Demonstrates :func:`mrsvalidate.analysis.method_comparison.bland_altman`
and its plot, on a synthetic pair of methods with a small proportional
bias — the kind of concentration-dependent disagreement a scalar
agreement statistic alone would not reveal.
"""

# %%
# Simulate two "methods" with a proportional bias
# --------------------------------------------------

import numpy as np

from mrsvalidate.analysis.method_comparison import bland_altman
from mrsvalidate.viz import plot_bland_altman

rng = np.random.default_rng(1)
n_samples = 150
reference = rng.uniform(1.0, 10.0, size=n_samples)
candidate = 1.05 * reference + rng.normal(scale=0.2 + 0.05 * reference, size=n_samples)
sample_ids = np.array([f"s{i}" for i in range(n_samples)])

# %%
# Constant-variance limits of agreement
# ----------------------------------------

result = bland_altman(reference, candidate, sample_ids, conditional=False, seed=0)
print(f"bias = {result.bias:.3f}, LoA = [{result.loa_low:.3f}, {result.loa_high:.3f}]")

fig, ax = plot_bland_altman(result)
fig.suptitle("Bland-Altman: constant-variance limits of agreement")

# %%
# Heteroscedastic (concentration-dependent) limits of agreement
# ------------------------------------------------------------------
#
# The default, ``conditional=True``, fits linear bias and scale models
# against the reference instead of assuming constant variance, which is
# appropriate here, since the simulated noise scale grows with
# concentration. Both analyses use the robust estimators (median bias,
# normal-consistent MAD) by default.

result_conditional = bland_altman(reference, candidate, sample_ids, seed=0)
fig2, ax2 = plot_bland_altman(result_conditional)
fig2.suptitle("Bland-Altman: concentration-dependent limits of agreement")
