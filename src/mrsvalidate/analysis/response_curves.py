"""1D/2D response-curve analysis for information-standardized
failure-mode characterization.

Implements the manuscript's "Monte Carlo Characterization of Failure
Modes" instruction that bias/variance/RMSE/``R_j``/``Delta_V_j`` be
"examined as functions of the controlled experimental variables ...
selected multidimensional analyses can be used to examine interactions
between parameters" — e.g. mSNR vs. concentration, or mSNR vs.
linewidth.

Notes
-----
The axis (or axes) to compare are specified explicitly by the caller
when defining the analysis — never inferred or hard-coded. Pass one
named covariate array for a 1D response curve (``axes={"mSNR": ...}``)
or two for a 2D response surface
(``axes={"mSNR": ..., "concentration": ...}``). This module only bins
and aggregates an already-computed outcome (bias, RMSE, |error|, R_j,
D^2, ...) against the caller-specified covariate(s); it never computes
the outcome itself — that comes from
:mod:`mrsvalidate.analysis.monte_carlo` or any other analysis module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np
from numpy.typing import NDArray

from ..results import AnalysisResult, ResultStatus, build_meta

__all__ = ["response_curve", "ResponseCurveResult"]

ResponseStatistic = Literal["mean", "median", "rmse"]
DispersionMethod = Literal["sd", "se"]


@dataclass(frozen=True)
class ResponseCurveResult(AnalysisResult):
    """Result of a :func:`response_curve` binned aggregation.

    Attributes
    ----------
    meta : AnalysisMeta
        Provenance and inclusion/exclusion record for this computation.
    axis_names : tuple of str
        Name of each binning axis, in the order they appear in `values`
        and every other per-axis array (1 name for a response curve, 2
        for a response surface).
    bin_edges : tuple of ndarray of float
        Bin edges supplied for each axis, in `axis_names` order; axis
        ``i`` has ``len(bin_edges[i]) - 1`` bins.
    bin_centers : tuple of ndarray of float
        Midpoint of each bin for each axis, in `axis_names` order.
    statistic_name : {"mean", "median", "rmse"}
        Which summary statistic was applied within each bin.
    values : ndarray of float, shape (n_bins,) or (n_bins_0, n_bins_1)
        Per-bin `statistic_name` of `outcome`; NaN for bins with fewer
        than ``min_bin_n`` observations.
    dispersion : ndarray of float, same shape as `values`
        Per-bin dispersion of `outcome` (SD or SE, per the `dispersion`
        argument); NaN under the same conditions as `values`.
    counts : ndarray of int, same shape as `values`
        Number of valid, in-range observations that fell into each bin
        (reported even for bins below ``min_bin_n``).
    sufficient_data_mask : ndarray of bool, same shape as `values`
        True for bins with at least ``min_bin_n`` observations, i.e.
        where `values`/`dispersion` are not NaN.
    """

    axis_names: tuple[str, ...]
    bin_edges: tuple[NDArray[np.float64], ...]
    bin_centers: tuple[NDArray[np.float64], ...]
    statistic_name: ResponseStatistic
    values: NDArray[np.float64]
    dispersion: NDArray[np.float64]
    counts: NDArray[np.int64]
    sufficient_data_mask: NDArray[np.bool_]


def _apply_statistic(values: NDArray[np.float64], statistic: ResponseStatistic) -> float:
    if statistic == "mean":
        return float(np.mean(values))
    if statistic == "median":
        return float(np.median(values))
    if statistic == "rmse":
        return float(np.sqrt(np.mean(values**2)))
    raise ValueError(f"Unknown statistic: {statistic!r}")


def _apply_dispersion(values: NDArray[np.float64], method: DispersionMethod) -> float:
    if values.shape[0] < 2:
        return float("nan")
    sd = float(np.std(values, ddof=1))
    if method == "sd":
        return sd
    if method == "se":
        return sd / np.sqrt(values.shape[0])
    raise ValueError(f"Unknown dispersion method: {method!r}")


def response_curve(
    outcome: NDArray[np.float64],
    axes: Mapping[str, NDArray[np.float64]],
    sample_ids: NDArray[np.str_],
    *,
    bin_edges: Mapping[str, NDArray[np.float64]],
    statistic: ResponseStatistic = "mean",
    dispersion: DispersionMethod = "sd",
    min_bin_n: int = 1,
) -> ResponseCurveResult:
    """Bin an outcome by 1 or 2 named covariates and summarize each bin.

    Groups `outcome` by the value of each covariate in `axes` (using the
    caller-supplied `bin_edges`) and reduces each bin to a single
    `statistic` value plus a `dispersion` estimate — a 1D response curve
    for one axis, or a 2D response surface for two.

    Parameters
    ----------
    outcome : ndarray of float, shape (n_samples,)
        The already-computed per-sample outcome to summarize (e.g. bias,
        RMSE, ``|error|``, ``R_j``, ``D^2``). This function never
        computes the outcome itself.
    axes : Mapping of str to ndarray of float
        One (for a response curve) or two (for a response surface) named
        covariate arrays, each of shape ``(n_samples,)`` and aligned with
        `outcome`. Insertion order determines `axis_names` and the axis
        order of every returned per-axis array.
    sample_ids : ndarray of str, shape (n_samples,)
        Per-sample identifiers, used for inclusion/exclusion bookkeeping.
    bin_edges : Mapping of str to ndarray of float
        Bin edges for each axis; must have exactly the same keys as
        `axes` (one edge array per axis). Values outside the supplied
        edges are excluded, not clipped into the end bins.
    statistic : {"mean", "median", "rmse"}, default "mean"
        Summary statistic computed within each bin.
    dispersion : {"sd", "se"}, default "sd"
        Dispersion estimate computed within each bin: sample standard
        deviation (``ddof=1``) or standard error of the mean.
    min_bin_n : int, default 1
        Minimum number of valid observations a bin must contain for
        `statistic`/`dispersion` to be reported; bins below this
        threshold keep their `counts` and bin edges but get NaN
        `values`/`dispersion` and False `sufficient_data_mask` — never
        silently interpolated or dropped from the grid.

    Returns
    -------
    ResponseCurveResult
        Per-bin statistic, dispersion, counts, and sufficiency mask,
        plus the bin geometry. See :class:`ResponseCurveResult` for the
        full field list.

    Raises
    ------
    ValueError
        If `axes` does not have 1 or 2 entries, or if `bin_edges` does
        not have exactly the same keys as `axes`.

    Notes
    -----
    A sample is excluded before binning if `outcome` or any axis value is
    non-finite, or if any axis value falls outside its supplied
    `bin_edges` range; `meta.exclusion_reasons["nonfinite_or_out_of_bin_range"]`
    records how many samples were excluded for either reason. If no
    sample remains, ``meta.status = ResultStatus.INSUFFICIENT_DATA`` and
    every bin is NaN-filled with zero counts.

    Examples
    --------
    A 1D response curve of an outcome against mSNR:

    >>> import numpy as np
    >>> from mrsvalidate.analysis.response_curves import response_curve
    >>> snr = np.array([5., 5., 5., 5., 15., 15., 15., 15., 25., 25., 25., 25.])
    >>> outcome = np.array([2.1, 1.9, 2.05, 2.2, 0.9, 1.1, 1.0, 0.95, 0.4, 0.5, 0.45, 0.42])
    >>> ids = np.array([f"s{i}" for i in range(12)])
    >>> edges = np.array([0.0, 10.0, 20.0, 30.0])
    >>> result = response_curve(
    ...     outcome, {"mSNR": snr}, ids,
    ...     bin_edges={"mSNR": edges}, statistic="mean", dispersion="sd", min_bin_n=2,
    ... )
    >>> result.axis_names
    ('mSNR',)
    >>> print(result.bin_centers[0])
    [ 5. 15. 25.]
    >>> print(np.round(result.values, 4))
    [2.0625 0.9875 0.4425]
    >>> result.counts
    array([4, 4, 4])
    >>> result.sufficient_data_mask
    array([ True,  True,  True])

    A 2D response surface of the same outcome against mSNR and
    concentration:

    >>> concentration = np.array([1., 1., 1., 1., 1., 1., 1., 1., 2., 2., 2., 2.])
    >>> conc_edges = np.array([0.5, 1.5, 2.5])
    >>> surface = response_curve(
    ...     outcome, {"mSNR": snr, "concentration": concentration}, ids,
    ...     bin_edges={"mSNR": edges, "concentration": conc_edges},
    ...     statistic="rmse", min_bin_n=2,
    ... )
    >>> surface.axis_names
    ('mSNR', 'concentration')
    >>> surface.values.shape
    (3, 2)
    >>> surface.counts
    array([[4, 0],
           [4, 0],
           [0, 4]])
    """

    axis_names = tuple(axes.keys())
    if len(axis_names) not in (1, 2):
        raise ValueError(f"response_curve supports 1 or 2 axes, got {len(axis_names)}")
    if set(bin_edges.keys()) != set(axis_names):
        raise ValueError("bin_edges must have exactly the same keys as axes")

    all_ids = tuple(np.asarray(sample_ids, dtype=str).tolist())
    outcome_arr = np.asarray(outcome, dtype=np.float64)
    axis_arrays = [np.asarray(axes[name], dtype=np.float64) for name in axis_names]
    edges = [np.asarray(bin_edges[name], dtype=np.float64) for name in axis_names]

    finite = np.isfinite(outcome_arr)
    for arr in axis_arrays:
        finite &= np.isfinite(arr)

    n_bins = [e.shape[0] - 1 for e in edges]
    bin_idx = []
    in_range = finite.copy()
    for arr, e, nb in zip(axis_arrays, edges, n_bins):
        idx = np.digitize(arr, e) - 1
        in_range &= (idx >= 0) & (idx < nb)
        bin_idx.append(idx)

    included_ids = tuple(np.asarray(all_ids)[in_range].tolist())
    n_excluded = len(all_ids) - len(included_ids)
    status = ResultStatus.VALID if included_ids else ResultStatus.INSUFFICIENT_DATA

    meta = build_meta(
        status=status,
        all_ids=all_ids,
        included_ids=included_ids,
        exclusion_reasons={"nonfinite_or_out_of_bin_range": n_excluded} if n_excluded else {},
        method="response_curve",
        configuration={
            "axis_names": axis_names,
            "statistic": statistic,
            "dispersion": dispersion,
            "min_bin_n": min_bin_n,
        },
    )

    bin_centers = tuple((e[:-1] + e[1:]) / 2.0 for e in edges)

    if len(axis_names) == 1:
        shape = (n_bins[0],)
    else:
        shape = (n_bins[0], n_bins[1])

    values = np.full(shape, np.nan, dtype=np.float64)
    disp = np.full(shape, np.nan, dtype=np.float64)
    counts = np.zeros(shape, dtype=np.int64)
    sufficient = np.zeros(shape, dtype=bool)

    if status is ResultStatus.VALID:
        outcome_valid = outcome_arr[in_range]
        idx0_valid = bin_idx[0][in_range]
        if len(axis_names) == 1:
            for b in range(n_bins[0]):
                cell_mask = idx0_valid == b
                counts[b] = int(np.sum(cell_mask))
                if counts[b] >= min_bin_n:
                    cell_values = outcome_valid[cell_mask]
                    values[b] = _apply_statistic(cell_values, statistic)
                    disp[b] = _apply_dispersion(cell_values, dispersion)
                    sufficient[b] = True
        else:
            idx1_valid = bin_idx[1][in_range]
            for b0 in range(n_bins[0]):
                for b1 in range(n_bins[1]):
                    cell_mask = (idx0_valid == b0) & (idx1_valid == b1)
                    counts[b0, b1] = int(np.sum(cell_mask))
                    if counts[b0, b1] >= min_bin_n:
                        cell_values = outcome_valid[cell_mask]
                        values[b0, b1] = _apply_statistic(cell_values, statistic)
                        disp[b0, b1] = _apply_dispersion(cell_values, dispersion)
                        sufficient[b0, b1] = True

    return ResponseCurveResult(
        meta=meta,
        axis_names=axis_names,
        bin_edges=tuple(edges),
        bin_centers=bin_centers,
        statistic_name=statistic,
        values=values,
        dispersion=disp,
        counts=counts,
        sufficient_data_mask=sufficient,
    )
