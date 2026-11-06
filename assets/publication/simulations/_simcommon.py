"""Shared helpers for the supplementary simulation scripts.

Each ``sim_<claim>.py`` script defines a list of simulation *cells* (one
parameter combination each) and a function that runs one cell. This module
gives every cell its own independent random stream derived from the
script's fixed seed, runs the cells (optionally in parallel; the results do
not depend on the number of worker processes), adds the Monte Carlo
standard error of each rate, and writes the results to
``outputs/<claim>.csv``.
"""

from __future__ import annotations

import argparse
import csv
import math
import platform
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def mc_se(rate: float, n_reps: int) -> float:
    """Monte Carlo standard error of a rejection/coverage rate, sqrt(p(1-p)/R)."""

    if n_reps <= 0 or not np.isfinite(rate):
        return float("nan")
    return math.sqrt(rate * (1.0 - rate) / n_reps)


def cell_seeds(seed: int, n_cells: int) -> list[np.random.SeedSequence]:
    """One independent SeedSequence per cell, in cell order."""

    return np.random.SeedSequence(seed).spawn(n_cells)


def parse_args(
    doc: str, argv: Sequence[str] | None = None, parts: Sequence[str] | None = None
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="smoke test: few replicates (and a reduced grid where noted); "
        "writes outputs/<claim>_quick.csv instead of the publication CSV",
    )
    parser.add_argument("--workers", type=int, default=1, help="parallel worker processes (default 1)")
    parser.add_argument("--reps", type=int, default=None, help="override the replicate count")
    if parts:
        parser.add_argument("--part", choices=[*parts, "all"], default="all", help="run only one part")
    return parser.parse_args(argv)


def run_cells(
    run_one: Callable[[dict, np.random.SeedSequence], dict],
    cells: list[dict],
    seed: int,
    workers: int,
) -> list[dict]:
    """Run ``run_one(cell, seed_sequence)`` for every cell, preserving order."""

    seeds = cell_seeds(seed, len(cells))
    t0 = time.perf_counter()
    if workers <= 1:
        out = []
        for i, (cell, ss) in enumerate(zip(cells, seeds)):
            out.append(run_one(cell, ss))
            print(f"[{i + 1}/{len(cells)}] {_fmt(out[-1])}  ({time.perf_counter() - t0:.0f} s)", flush=True)
        return out
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, cell, ss) for cell, ss in zip(cells, seeds)]
        out = []
        for i, fut in enumerate(futures):
            out.append(fut.result())
            print(f"[{i + 1}/{len(cells)}] {_fmt(out[-1])}  ({time.perf_counter() - t0:.0f} s)", flush=True)
        return out


def _fmt(row: dict | list) -> str:
    if isinstance(row, list):
        return "; ".join(_fmt(r) for r in row)
    return ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items())


def write_results(claim: str, rows: list[dict], quick: bool, elapsed_s: float) -> Path:
    """Write ``rows`` to ``outputs/<claim>.csv`` (``_quick`` suffix for smoke tests)."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{claim}{'_quick' if quick else ''}.csv"
    fields: list[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    import mrsvalidate  # noqa: PLC0415

    provenance = {
        "mrsvalidate_version": getattr(mrsvalidate, "__version__", "unknown"),
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "quick_run": quick,
        "elapsed_s": round(elapsed_s, 1),
    }
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields + list(provenance))
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, **provenance})
    print(f"wrote {path} ({len(rows)} rows, {elapsed_s:.0f} s)")
    return path


def ensure_src_on_path() -> None:
    """Allow running from a checkout without installing the package."""

    src = Path(__file__).resolve().parents[3] / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
