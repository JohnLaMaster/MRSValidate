"""Robust operating-envelope plot.

Renders per-bin robust lower confidence bounds and the pass/fail
operating-envelope determination from a precomputed
:class:`~mrsvalidate.analysis.operating_envelope.OperatingEnvelopeResult`.

Notes
-----
This module must never import or call into ``mrsvalidate.analysis.*``
(enforced statically and dynamically by
``tests/plots/test_viz_boundary.py``). Every success rate, confidence
bound, and pass/fail bin determination is read directly from the
result; this module performs no binomial calculation of its own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ..analysis.operating_envelope import OperatingEnvelopeResult

__all__ = ["plot_operating_envelope"]


def plot_operating_envelope(
    result: OperatingEnvelopeResult, *, ax: Axes | None = None
) -> tuple[Figure, Axes]:
    """Plot per-bin robust lower confidence bounds against a reliability target.

    Renders one bar per predicted-concentration bin at
    ``result.robust_lower_bound``, colored green/red for
    inside/outside the operating envelope (gray where data is
    insufficient), plus a horizontal line at
    ``result.target_reliability``. This function only renders a
    precomputed
    :class:`~mrsvalidate.analysis.operating_envelope.OperatingEnvelopeResult`;
    it never recomputes a binomial confidence bound.

    Parameters
    ----------
    result : OperatingEnvelopeResult
        Precomputed operating-envelope result from
        :func:`mrsvalidate.analysis.operating_envelope.operating_envelope`.
        Bin edges, robust lower bounds, and the sufficient-data/
        operating-bin masks are read directly from this object.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on. If ``None`` (default), a new figure and axes
        are created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing ``ax`` (the caller's figure, if ``ax``
        was supplied).
    ax : matplotlib.axes.Axes
        The axes the bars and target-reliability line were drawn on.

    Notes
    -----
    This function never imports or calls into
    ``mrsvalidate.analysis.*`` — see the module docstring and
    :mod:`mrsvalidate.viz` for the architectural boundary this
    preserves. Bins where ``result.sufficient_data_mask`` is ``False``
    are colored gray regardless of their (possibly ``NaN``) robust
    lower bound, so a data-starved bin is never mistaken for a passing
    or failing one.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import numpy as np
    >>> from mrsvalidate.analysis.operating_envelope import operating_envelope
    >>> from mrsvalidate.viz import plot_operating_envelope
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> predicted_concentration = rng.uniform(0, 10, size=n)
    >>> information_metric = rng.uniform(0, 5, size=n)
    >>> success = rng.uniform(size=n) < 0.9
    >>> sample_ids = np.array([f"s{i}" for i in range(n)])
    >>> result = operating_envelope(
    ...     predicted_concentration, information_metric, success, sample_ids,
    ...     bin_edges=np.linspace(0, 10, 6),
    ...     stratum_edges=np.linspace(0, 5, 3),
    ...     target_reliability=0.8,
    ... )
    >>> fig, ax = plot_operating_envelope(result)
    >>> type(fig).__name__
    'Figure'
    """

    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    bin_centers = (result.bin_edges[:-1] + result.bin_edges[1:]) / 2.0
    colors = np.where(result.operating_bin_mask, "tab:green", "tab:red")
    colors = np.where(result.sufficient_data_mask, colors, "tab:gray")

    ax.bar(bin_centers, result.robust_lower_bound, width=np.diff(result.bin_edges), color=colors, alpha=0.7)
    ax.axhline(result.target_reliability, color="k", linestyle="--", linewidth=1, label="target reliability")
    ax.set_xlabel("predicted concentration bin")
    ax.set_ylabel("robust lower confidence bound")
    ax.set_ylim(0, 1)
    ax.legend(loc="best", fontsize=8)
    return fig, ax
