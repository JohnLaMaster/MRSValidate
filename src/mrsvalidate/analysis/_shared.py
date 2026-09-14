"""Internal helpers shared across ``mrsvalidate.analysis`` submodules.

Not part of the public API. Every analysis function follows the same
shape: (1) find which rows are usable given the arrays it needs, (2)
decide a :class:`~mrsvalidate.results.ResultStatus` from that plus any
statistic-specific undefined condition, (3) build an
:class:`~mrsvalidate.results.AnalysisMeta` that fully accounts for
``n_total = n_used + n_omitted``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..missingness import MissingnessReport, analyze_missingness
from ..results import AnalysisMeta, ResultStatus, build_meta


def valid_subset(
    sample_ids: NDArray[np.str_], *columns: NDArray[np.float64]
) -> tuple[tuple[str, ...], MissingnessReport, tuple[NDArray[np.float64], ...]]:
    """Return ``(all_ids, missingness_report, valid_columns)``.

    ``valid_columns`` are ``columns`` filtered to the rows the report
    marks valid (finite across every supplied column), in the same
    order as ``columns``.
    """

    report = analyze_missingness(sample_ids, *columns)
    valid_arrays = tuple(
        np.asarray(c, dtype=np.float64)[report.valid_mask] for c in columns
    )
    all_ids = tuple(np.asarray(sample_ids, dtype=str).tolist())
    return all_ids, report, valid_arrays


def meta_for(
    *,
    all_ids: tuple[str, ...],
    report: MissingnessReport,
    method: str,
    status: ResultStatus | None = None,
    min_n: int = 1,
    configuration: Mapping[str, Any] | None = None,
    warnings: tuple[str, ...] = (),
) -> AnalysisMeta:
    """Build the :class:`AnalysisMeta` for a statistic computed on
    ``report.included_ids``.

    If ``status`` is not given, it defaults to
    ``ResultStatus.INSUFFICIENT_DATA`` when fewer than ``min_n`` valid
    rows remain, else ``ResultStatus.VALID``. Pass ``status`` explicitly
    to override with e.g. ``ResultStatus.UNDEFINED`` once the caller has
    detected a statistic-specific undefined condition (near-zero
    denominator, constant vector, ...).
    """

    if status is None:
        status = (
            ResultStatus.INSUFFICIENT_DATA
            if len(report.included_ids) < min_n
            else ResultStatus.VALID
        )
    return build_meta(
        status=status,
        all_ids=all_ids,
        included_ids=report.included_ids,
        exclusion_reasons=report.exclusion_reasons,
        warnings=warnings,
        method=method,
        configuration=configuration or {},
    )
