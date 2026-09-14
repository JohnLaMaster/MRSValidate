"""1D/2D response-curve plots.

Renders a 1D response curve (line + error bars) or a 2D response
surface (heatmap) from a precomputed
:class:`~mrsvalidate.analysis.response_curves.ResponseCurveResult`.

Notes
-----
This module must never import or call into ``mrsvalidate.analysis.*``
(enforced statically and dynamically by
``tests/plots/test_viz_boundary.py``). Every plotted value (bin
statistic, dispersion, sufficiency mask) is read directly from the
result; this module performs no binning or aggregation of its own —
only axis/marker/color layout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ..analysis.response_curves import ResponseCurveResult

__all__ = ["plot_response_curve"]


def plot_response_curve(
    result: ResponseCurveResult, *, ax: Axes | None = None
) -> tuple[Figure, Axes]:
    """Plot a 1D response curve or 2D response surface.

    Renders an error-bar line plot when ``result`` has one axis
    (``len(result.axis_names) == 1``), or a pcolormesh heatmap when it
    has two. This function only renders a precomputed
    :class:`~mrsvalidate.analysis.response_curves.ResponseCurveResult`;
    it never recomputes a bin statistic or dispersion.

    Parameters
    ----------
    result : ResponseCurveResult
        Precomputed response-curve result from
        :func:`mrsvalidate.analysis.response_curves.response_curve`.
        Bin centers/edges, values, dispersion, and the sufficient-data
        mask are read directly from this object.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on. If ``None`` (default), a new figure and axes
        are created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing ``ax`` (the caller's figure, if ``ax``
        was supplied).
    ax : matplotlib.axes.Axes
        The axes the curve or surface was drawn on.

    Notes
    -----
    This function never imports or calls into
    ``mrsvalidate.analysis.*`` — see the module docstring and
    :mod:`mrsvalidate.viz` for the architectural boundary this
    preserves. Bins where ``result.sufficient_data_mask`` is ``False``
    are plotted as ``NaN`` (a gap in the line, or a blank cell in the
    heatmap) rather than showing an unreliable value.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import numpy as np
    >>> from mrsvalidate.analysis.response_curves import response_curve
    >>> from mrsvalidate.viz import plot_response_curve
    >>> rng = np.random.default_rng(0)
    >>> n = 100
    >>> outcome = rng.normal(size=n)
    >>> msnr = rng.uniform(0, 20, size=n)
    >>> sample_ids = np.array([f"s{i}" for i in range(n)])
    >>> result = response_curve(
    ...     outcome, {"mSNR": msnr}, sample_ids,
    ...     bin_edges={"mSNR": np.linspace(0, 20, 5)},
    ... )
    >>> fig, ax = plot_response_curve(result)
    >>> type(fig).__name__
    'Figure'
    """

    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if len(result.axis_names) == 1:
        x = result.bin_centers[0]
        y = np.where(result.sufficient_data_mask, result.values, np.nan)
        yerr = np.where(result.sufficient_data_mask, result.dispersion, np.nan)
        ax.errorbar(x, y, yerr=yerr, marker="o", linestyle="-", capsize=3)
        ax.set_xlabel(result.axis_names[0])
        ax.set_ylabel(result.statistic_name)
    else:
        values = np.where(result.sufficient_data_mask, result.values, np.nan)
        x_edges, y_edges = result.bin_edges
        mesh = ax.pcolormesh(x_edges, y_edges, values.T, shading="flat")
        fig.colorbar(mesh, ax=ax, label=result.statistic_name)
        ax.set_xlabel(result.axis_names[0])
        ax.set_ylabel(result.axis_names[1])

    return fig, ax
