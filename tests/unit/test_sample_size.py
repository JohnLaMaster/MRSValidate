from mrsvalidate.analysis.sample_size import (
    MINIMUM_INDEPENDENT_CLUSTERS,
    MINIMUM_INDEPENDENT_SAMPLES,
    check_minimum_clusters,
    check_minimum_sample_size,
)


def test_check_minimum_sample_size():
    assert check_minimum_sample_size(MINIMUM_INDEPENDENT_SAMPLES) is True
    assert check_minimum_sample_size(MINIMUM_INDEPENDENT_SAMPLES - 1) is False


def test_check_minimum_clusters():
    assert check_minimum_clusters(MINIMUM_INDEPENDENT_CLUSTERS) is True
    assert check_minimum_clusters(MINIMUM_INDEPENDENT_CLUSTERS - 1) is False
