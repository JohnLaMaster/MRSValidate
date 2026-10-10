# Third-Party Code Provenance

This file is the authoritative, human-readable record of external
*implementation source code* — as opposed to ordinary installed
dependencies (see `THIRD_PARTY_LICENSES.md`) — incorporated, adapted,
translated, or substantially derived into this repository, per
`LIBRARY_IMPLEMENTATION_SPEC.md` §31.

## Summary

```text
External source code incorporated: No
External source code adapted or translated: No
External implementations used for validation only: No (not yet performed;
    see "Planned validation" below)
Unresolved licensing issues: No
```

## Statement

As of this writing, every statistical method in `src/mrsvalidate/` has
been implemented **independently** from:

- the mathematical specification in `LIBRARY_IMPLEMENTATION_SPEC.md`;
- the accompanying manuscript, `Laying_the_Foundations_compiled.tex`;
- the scientific publications the manuscript itself cites for each
  method (Lin 1989, Passing & Bablok 1983, Linnet 1998 (Deming),
  Shukla 1973, Silveira et al., Kössler & Mukherjee, Funato et al.,
  Kuleshov et al. 2018, Choi et al. 2025, etc.);
- and standard, general statistical theory (e.g. Hájek-Šidák linear
  rank-statistic moments for the Wilcoxon/Ansari-Bradley/Savage
  components of ECWAS).

No code from an external repository, package, notebook, or
supplementary-code archive has been copied, translated, or
substantially derived into this project. Reuse of mature third-party
**dependencies** (NumPy, SciPy, Statsmodels) for their own tested
primitives (Pearson correlation, Theil-Sen, permutation-test
machinery, mixed-effects/REML fitting, cluster-robust covariance) is
tracked as ordinary dependency usage in `THIRD_PARTY_LICENSES.md`, not
as incorporated source, per the spec's own distinction between the two
categories.

## Planned validation (not yet performed)

Per §31.6 of the spec, an independently written implementation may
still be *compared against* an external reference implementation
purely for validation, without that reference being incorporated or
adapted. Candidate future validation targets (to be recorded here, not
copied, if used):

- R's `mcr` package or MedCalc, for Passing-Bablok and Deming
  regression fixtures;
- a reference Lin's CCC implementation (e.g. R's `epiR::epi.ccc`) for
  published numerical examples.

Should any such comparison be performed, this file will be updated
with: source name/version, retrieval date, license, and an explicit
`Reuse status: reference/validation only` / `Source incorporated: no`
declaration, per the spec's required record format.

## Maintenance

This file must be updated *before* any future change that copies,
translates, adapts, or substantially derives implementation code from
an external source — see `LIBRARY_IMPLEMENTATION_SPEC.md` §31.2-§31.6
for the required record fields and license-verification rules.
