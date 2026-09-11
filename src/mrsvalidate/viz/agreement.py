"""Bland-Altman plot.

Renders a difference-vs-axis scatter with bias and limits-of-agreement
(or conditional location/scale lines) from a precomputed
:class:`~mrsvalidate.analysis.method_comparison.BlandAltmanResult`.

Notes
-----
This module must never import or call into ``mrsvalidate.analysis.*``
(enforced statically and dynamically by
``tests/plots/test_viz_boundary.py``). Bias, limits of agreement, and
conditional bias/scale coefficients are read directly from the result;
nothing here recomputes a difference, bias, or regression.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ..analysis.method_comparison import BlandAltmanResult

__all__ = ["plot_bland_altman"]


def plot_bland_altman(result: "BlandAltmanResult", *, ax: "Axes | None" = None) -> tuple["Figure", "Axes"]:
    """Plot a Bland-Altman difference scatter with bias and agreement limits.

    Renders ``result.differences`` against ``result.x_values``, plus
    either the constant bias and limits-of-agreement lines or, when
    ``result.conditional`` is set, the fitted linear bias(c)/SD(c)
    lines. This function only renders a precomputed
    :class:`~mrsvalidate.analysis.method_comparison.BlandAltmanResult`;
    it never recomputes a difference, bias, or regression.

    Parameters
    ----------
    result : BlandAltmanResult
        Precomputed method-comparison result from
        :func:`mrsvalidate.analysis.method_comparison.bland_altman`.
        Bias, limits of agreement, and (if present) conditional
        bias/scale coefficients are read directly from this object.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on. If ``None`` (default), a new figure and axes
        are created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing ``ax`` (the caller's figure, if ``ax``
        was supplied).
    ax : matplotlib.axes.Axes
        The axes the scatter and bias/agreement lines were drawn on.

    Notes
    -----
    This function never imports or calls into
    ``mrsvalidate.analysis.*`` — see the module docstring and
    :mod:`mrsvalidate.viz` for the architectural boundary this
    preserves. When ``result.conditional`` is ``True``, the plotted
    bias(c) and ±1.96·SD(c) bands are linear functions of the x-axis
    values rather than horizontal lines; both forms are read entirely
    from ``result``.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import numpy as np
    >>> from mrsvalidate.analysis.method_comparison import bland_altman
    >>> from mrsvalidate.viz import plot_bland_altman
    >>> reference = np.array([1.0, 2.0, 3.0, 4.0])
    >>> candidate = reference + 1.0
    >>> sample_ids = np.array([f"s{i}" for i in range(4)])
    >>> result = bland_altman(reference, candidate, sample_ids)
    >>> fig, ax = plot_bland_altman(result)
    >>> type(fig).__name__
    'Figure'
    """

    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.scatter(result.x_values, result.differences, s=12, alpha=0.6)

    if result.conditional and result.conditional_bias_coefficients is not None:
        b0, b1 = result.conditional_bias_coefficients
        s0, s1 = result.conditional_scale_coefficients
        x_line = np.linspace(np.min(result.x_values), np.max(result.x_values), 100)
        bias_line = b0 + b1 * x_line
        scale_line = s0 + s1 * x_line
        ax.plot(x_line, bias_line, "r-", linewidth=1.5, label="bias(c)")
        ax.plot(x_line, bias_line + 1.96 * scale_line, "r--", linewidth=1, label="±1.96·SD(c)")
        ax.plot(x_line, bias_line - 1.96 * scale_line, "r--", linewidth=1)
    else:
        ax.axhline(result.bias, color="r", linewidth=1.5, label="bias")
        ax.axhline(result.loa_low, color="r", linestyle="--", linewidth=1, label="limits of agreement")
        ax.axhline(result.loa_high, color="r", linestyle="--", linewidth=1)

    ax.set_xlabel(f"x-axis ({result.x_axis})")
    ax.set_ylabel("candidate - reference")
    ax.legend(loc="best", fontsize=8)
    return fig, ax
