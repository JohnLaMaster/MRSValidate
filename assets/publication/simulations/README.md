# Supplementary simulation studies

Reproduction scripts for the Monte Carlo results in
`supplement_simulations.tex` (rows e–h of section 5.2 in
`assets/dev/MANUSCRIPT_UPDATES.md`). Each script has a fixed seed, its
replicate count as a constant at the top (`N_REPS`, ...), and writes the
rates, their Monte Carlo SE `sqrt(p(1-p)/R)` and every parameter to
`outputs/<claim>.csv` (plus provenance columns: library/NumPy/Python
versions, runtime). Every simulation cell has its own random stream
(`SeedSequence(seed).spawn`), so results do not depend on `--workers`.

Run from the repository root:

```bash
PY=/home/john/anaconda3/envs/python3.12/bin/python
PYTHONPATH=src $PY assets/publication/simulations/<script> [--quick] [--workers K] [--reps R]
```

`--quick` is a smoke test (few replicates, sometimes a reduced grid) and
writes `outputs/<claim>_quick.csv`; its numbers are not for publication.

| Supplement section | Script | Output | Publication replicates | Quick run (measured) | Full run (estimated) |
|---|---|---|---|---|---|
| `supp:sim:ecwas_type1` (row e) | `sim_ecwas_type1.py` | `ecwas_type1.csv` | R = 10 000 per n; P = 20 000 relabelings | 7 s (1 worker) | ~4 min, 1 worker |
| `supp:sim:pb_cusum_calibration` (row f) | `sim_pb_cusum_calibration.py` | `pb_cusum_calibration.csv` | R = 5000 per cell, B = 199 | 15 s (4 workers; 33 CPU-s) | ~3.5 CPU-h; ~1 h with `--workers 4` |
| `supp:sim:cusum_lambda` (row g) | `sim_cusum_lambda.py --part lambda` | `cusum_lambda.csv` | R = 2000 per cell, B = 199 | 18 s (4 workers) | ~19 CPU-h; ~5 h with `--workers 4` |
| `supp:sim:cusum_lambda` (Theil–Sen) | `sim_cusum_lambda.py --part theil_sen` | `theil_sen_validation.csv` | R = 5000 per cell, B = 199 | 8 s (4 workers) | ~10 min with `--workers 4` |
| `supp:sim:cluster_robust` (row h) | `sim_cluster_robust.py` | `cluster_robust.csv` | R = 10 000 per G | 3 s (1 worker) | ~3 min, 1 worker |

Full-run estimates scale the measured per-dataset cost on this machine
(single core): one Passing–Bablok/Theil–Sen fit with B = 199 bootstrap
refits takes about 0.04 s at n = 60, 0.1 s at n = 200, 0.47 s at n = 500
and 1.4 s at n = 1000. The lambda study is dominated by its 20 cells at
n = 1000 (about 48 min each); R = 2000 gives an MC SE of about
0.5 percentage points at 5 %. Use `--workers` up to the number of cores.

Notes

- The ECWAS script applies one permutation critical value per sample size
  to all null datasets (valid without ties, where the permutation null
  depends only on the group sizes), as the unit test does.
- The cluster-robust script calls the private helpers
  `mixed_effects._cluster_robust_inference` and `_t_test`, as the unit test
  does; everything else uses the public API.
- In `cusum_lambda`, Passing–Bablok *in silico* uses lambda = 0.01 because
  `passing_bablok` rejects `cusum_bootstrap_variance_ratio=0`
  (MANUSCRIPT_UPDATES.md 2.4, "Library (to do)"). Supplying lambda = 1 is
  identical to the default `None`.
- After a full run, replace the `% PROVISIONAL` / `\textbf{TBD}` entries in
  `supplement_simulations.tex` with the CSV values.
