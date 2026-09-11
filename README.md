# MRSValidate

Statistical analysis and visualization library that accompanies the "Laying the Foundations" paper.

`mrsvalidate` turns model outputs, ground/reference values, uncertainty
estimates, and externally supplied information bounds (e.g. CRLBs) into
reproducible scientific analyses for evaluating fitted MRS/metabolite
deep-learning models. The scientific analysis layer computes every
metric, statistical test, confidence interval, and standardized
quantity; the visualization layer only renders those precomputed
results — it never recomputes a statistic. See
[`LIBRARY_IMPLEMENTATION_SPEC.md`](LIBRARY_IMPLEMENTATION_SPEC.md) for
the full scientific contract and implementation specification, and
[`CHANGELOG.md`](CHANGELOG.md) for what is implemented so far.

## Installation (development)

```bash
pip install -e ".[dev]"
```

## Quick example

```python
import numpy as np
from mrsvalidate import ObservationSet, ParameterSchema, ParameterSpec, ParameterRole, align_observations
from mrsvalidate.analysis.regression import regression_metrics
from mrsvalidate.analysis.agreement import lin_ccc

schema = ParameterSchema((ParameterSpec("NAA", ParameterRole.CONCENTRATION, "mM"),))
sample_ids = np.array([f"s{i}" for i in range(100)])
reference_values = np.random.default_rng(0).uniform(1, 10, size=100)
predicted_values = reference_values + np.random.default_rng(1).normal(scale=0.3, size=100)

reference = ObservationSet(sample_ids=sample_ids, values=reference_values.reshape(-1, 1), parameters=schema)
candidate = ObservationSet(sample_ids=sample_ids, values=predicted_values.reshape(-1, 1), parameters=schema)

aligned = align_observations(reference, candidate, pairing="paired")
ref_col = aligned.reference.column("NAA")
pred_col = aligned.candidate.column("NAA")
ids = aligned.reference.sample_ids

result = regression_metrics(ref_col, pred_col, ids)
print(result.r, result.r_squared_direct, result.slope, result.intercept)

concordance = lin_ccc(ref_col, pred_col, ids)
print(concordance.rho_c)
```

## Status

Implementation status, scientific-definition notes, and known gaps are
tracked in [`CHANGELOG.md`](CHANGELOG.md). Third-party dependency and
code-provenance records are in [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md)
and [`THIRD_PARTY_CODE.md`](THIRD_PARTY_CODE.md). The project license
has not yet been finalized (see `THIRD_PARTY_CODE.md`).
