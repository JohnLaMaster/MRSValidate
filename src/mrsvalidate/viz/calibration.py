"""Calibration curve plot.

Renders empirical vs. nominal coverage and the expected calibration
error (ECE) from a precomputed
:class:`~mrsvalidate.analysis.calibration.CalibrationResult`.

Notes
-----
This module must never import or call into ``mrsvalidate.analysis.*``
(enforced statically and dynamically by
``tests/plots/test_viz_boundary.py``). Empirical coverage and ECE are
read directly from the result; this module performs no coverage/ECE
calculation of its own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ..analysis.calibration import CalibrationResult

__all__ = ["plot_calibration_result"]


def plot_calibration_result(
    result: "CalibrationResult", *, ax: "Axes | None" = None
) -> tuple["Figure", "Axes"]:
    """Plot empirical vs. nominal coverage with the ECE annotated.

    Renders ``result.empirical_coverage`` against
    ``result.nominal_coverage``, the perfect-calibration diagonal, and
    an annotation of ``result.ece``. This function only renders a
    precomputed
    :class:`~mrsvalidate.analysis.calibration.CalibrationResult`; it
    never recomputes coverage or ECE.

    Parameters
    ----------
    result : CalibrationResult
        Precomputed calibration result from
        :func:`mrsvalidate.analysis.calibration.expected_calibration_error`.
        Nominal coverage, empirical coverage, and ECE are read directly
        from this object.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on. If ``None`` (default), a new figure and axes
        are created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing ``ax`` (the caller's figure, if ``ax``
        was supplied).
    ax : matplotlib.axes.Axes
        The axes the calibration curve and ECE annotation were drawn
        on.

    Notes
    -----
    This function never imports or calls into
    ``mrsvalidate.analysis.*`` — see the module docstring and
    :mod:`mrsvalidate.viz` for the architectural boundary this
    preserves.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import numpy as np
    >>> from mrsvalidate.analysis.calibration import expected_calibration_error
    >>> from mrsvalidate.viz import plot_calibration_result
    >>> rng = np.random.default_rng(0)
    >>> pit_values = rng.uniform(size=200)
    >>> confidence_levels = np.array([0.1, 0.5, 0.9])
    >>> sample_ids = np.array([f"s{i}" for i in range(200)])
    >>> result = expected_calibration_error(pit_values, confidence_levels, sample_ids)
    >>> fig, ax = plot_calibration_result(result)
    >>> type(fig).__name__
    'Figure'
    """

    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
    ax.plot(result.nominal_coverage, result.empirical_coverage, "o-", label="observed")
    ax.set_xlabel("nominal confidence level")
    ax.set_ylabel("empirical coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0.02, 0.98, f"ECE = {result.ece:.4g}", transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    ax.legend(loc="lower right", fontsize=8)
    return fig, ax
