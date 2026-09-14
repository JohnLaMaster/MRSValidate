"""Predicted-vs-reference scatter plots.

Renders a scatter of model prediction against reference value,
annotated with regression diagnostics, from a precomputed
:class:`~mrsvalidate.analysis.regression.RegressionResult`.

Notes
-----
This module must never import or call into ``mrsvalidate.analysis.*``
(enforced statically and dynamically by
``tests/plots/test_viz_boundary.py``). No scientific quantity (slope,
intercept, r, R^2, ...) is recomputed here — every number displayed is
read directly from the result object; this module only lays out axes,
markers, and the lines implied by the already-computed
slope/intercept/x_range_for_line. Geometry-only calculations intrinsic
to rendering are fine; recomputing a statistic is not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ..analysis.regression import RegressionResult

__all__ = ["plot_regression_result"]


def plot_regression_result(
    result: RegressionResult, *, ax: Axes | None = None, identity_line: bool = True
) -> tuple[Figure, Axes]:
    """Plot a prediction-vs-reference scatter with fit lines and annotations.

    Renders the scatter of ``result.prediction`` against
    ``result.reference``, the identity line, and the already-computed
    Theil-Sen line, then annotates the axes with ``r``, ``R^2``, slope,
    intercept, the manuscript concise score, and the sample count. This
    function only renders a precomputed
    :class:`~mrsvalidate.analysis.regression.RegressionResult`; it never
    recomputes any of the plotted quantities.

    Parameters
    ----------
    result : RegressionResult
        Precomputed regression result from
        :func:`mrsvalidate.analysis.regression.regression_metrics`.
        Every plotted quantity (slope, intercept, r, R^2, concise
        score, x-range, prediction/reference arrays) is read directly
        from this object.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on. If ``None`` (default), a new figure and axes
        are created.
    identity_line : bool, optional
        Whether to draw the ``y = x`` identity line. Default is
        ``True``. The line is skipped automatically if
        ``result.x_range_for_line`` is not finite (e.g. for an
        insufficient-data result).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing ``ax`` (the caller's figure, if ``ax``
        was supplied).
    ax : matplotlib.axes.Axes
        The axes the scatter, fit lines, and annotation were drawn on.

    Notes
    -----
    This function never imports or calls into
    ``mrsvalidate.analysis.*`` — see the module docstring and
    :mod:`mrsvalidate.viz` for the architectural boundary this
    preserves. If ``result`` carries a non-finite slope/intercept (e.g.
    an ``UNDEFINED`` or ``INSUFFICIENT_DATA`` status), the
    corresponding fit line is simply omitted rather than raising.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import numpy as np
    >>> from mrsvalidate.analysis.regression import regression_metrics
    >>> from mrsvalidate.viz import plot_regression_result
    >>> reference = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> prediction = np.array([1.1, 2.2, 2.9, 4.1, 5.2])
    >>> sample_ids = np.array([f"s{i}" for i in range(5)])
    >>> result = regression_metrics(reference, prediction, sample_ids)
    >>> fig, ax = plot_regression_result(result)
    >>> type(fig).__name__
    'Figure'
    """

    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.scatter(result.reference, result.prediction, s=12, alpha=0.6, label="observations")

    x_lo, x_hi = result.x_range_for_line
    if identity_line and np.isfinite(x_lo) and np.isfinite(x_hi):
        ax.plot([x_lo, x_hi], [x_lo, x_hi], "k--", linewidth=1, label="identity")

    if np.isfinite(result.slope) and np.isfinite(result.intercept):
        line_x = np.array([x_lo, x_hi])
        line_y = result.intercept + result.slope * line_x
        ax.plot(line_x, line_y, "r-", linewidth=1.5, label="Theil-Sen fit")

    annotation = (
        f"r = {result.r:.3g}\n"
        f"R² = {result.r_squared_direct:.3g}\n"
        f"slope = {result.slope:.3g}\n"
        f"intercept = {result.intercept:.3g}\n"
        f"r³ (manuscript) = {result.concise_score_manuscript:.3g}\n"
        f"n = {result.meta.n_used}"
    )
    ax.text(
        0.02, 0.98, annotation, transform=ax.transAxes, va="top", ha="left", fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    ax.set_xlabel("reference")
    ax.set_ylabel("prediction")
    ax.legend(loc="lower right", fontsize=8)
    return fig, ax
