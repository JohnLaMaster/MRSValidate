# Third-Party Licenses

This file inventories mrsvalidate's software **dependencies** — packages
installed and imported through the Python packaging ecosystem. It is
distinct from `THIRD_PARTY_CODE.md`, which tracks any external
*implementation source* copied, adapted, or translated into this
repository (see that file: as of this writing, none).

Versions below are those resolved in the development environment at
the time of writing; the versions actually used for a given release
are recorded in that release's `manifest.json` / `environment.json`
and should be treated as authoritative for that release.

## Runtime dependencies

| Package | Version (dev) | License | SPDX | Project URL | Attribution/notice required | Redistribution constraint |
|---|---|---|---|---|---|---|
| NumPy | 2.3.1 | BSD 3-Clause | `BSD-3-Clause` | https://numpy.org | Copyright notice | None material |
| SciPy | 1.15.3 | BSD 3-Clause | `BSD-3-Clause` | https://scipy.org | Copyright notice | None material |
| Statsmodels | 0.15.0 | BSD 3-Clause (majority; some modules carry compatible permissive notices) | `BSD-3-Clause` | https://www.statsmodels.org | Copyright notice | None material |

## Optional (`viz`, `pandas` extras)

| Package | Version (dev) | License | SPDX | Project URL | Attribution/notice required | Redistribution constraint |
|---|---|---|---|---|---|---|
| Matplotlib | 3.11.0 | Matplotlib License (BSD-style, PSF-derived) | `PSF-2.0`-like custom license (see project `LICENSE/LICENSE.txt`) | https://matplotlib.org | Copyright/license notice per Matplotlib's own license text | None material |
| pandas | 3.0.2 | BSD 3-Clause | `BSD-3-Clause` | https://pandas.pydata.org | Copyright notice | None material |

## Development-only dependencies (never shipped in the released wheel/sdist)

| Package | Version (dev) | License | SPDX | Project URL |
|---|---|---|---|---|
| pytest | 9.1.1 | MIT | `MIT` | https://pytest.org |
| Hypothesis | 6.168.0 | Mozilla Public License 2.0 | `MPL-2.0` | https://hypothesis.readthedocs.io |
| build | (release tooling) | Apache-2.0 | `Apache-2.0` | https://pypa-build.readthedocs.io |

Hypothesis is MPL-2.0, a weak-copyleft license — this is a
development-only dependency (property/adversarial tests), never
imported by the released package, so its copyleft terms do not extend
to `mrsvalidate` itself. No modifications to Hypothesis's own source
are made or distributed here.

## Notes

- None of the packages above are physically incorporated into this
  repository; they are ordinary installed dependencies resolved via
  `pyproject.toml`. See `THIRD_PARTY_CODE.md` for the separate question
  of incorporated/adapted source code.
- This inventory should be regenerated (or at least spot-checked)
  before each release, since dependency versions and, occasionally,
  licenses change between releases.
- No license-compatibility conflict is currently known between these
  dependencies and a permissive project license (MIT/BSD-3-Clause/
  Apache-2.0); this has **not** been reviewed by counsel. See
  `LIBRARY_IMPLEMENTATION_SPEC.md` §31 for the license-selection
  process this project follows before finalizing `LICENSE`.
