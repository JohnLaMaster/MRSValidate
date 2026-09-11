import pytest

from mrsvalidate.results import AnalysisMeta, ResultStatus, build_meta, is_usable, AnalysisResult


def test_build_meta_derives_counts_from_ids():
    meta = build_meta(
        status=ResultStatus.VALID,
        all_ids=("a", "b", "c"),
        included_ids=("a", "c"),
        method="unit_test",
    )
    assert meta.n_total == 3
    assert meta.n_used == 2
    assert meta.n_omitted == 1
    assert meta.excluded_ids == ("b",)


def test_analysis_meta_rejects_inconsistent_counts():
    with pytest.raises(ValueError):
        AnalysisMeta(status=ResultStatus.VALID, n_total=3, n_used=2, n_omitted=2)


def test_analysis_meta_rejects_negative_counts():
    with pytest.raises(ValueError):
        AnalysisMeta(status=ResultStatus.VALID, n_total=-1, n_used=0, n_omitted=0)


def test_is_usable():
    valid = AnalysisResult(
        meta=AnalysisMeta(status=ResultStatus.VALID, n_total=1, n_used=1, n_omitted=0)
    )
    undefined = AnalysisResult(
        meta=AnalysisMeta(
            status=ResultStatus.UNDEFINED, n_total=1, n_used=1, n_omitted=0
        )
    )
    assert is_usable(valid)
    assert not is_usable(undefined)
