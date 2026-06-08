import numpy as np
from numcompute_stream.stats import StreamingStats

def test_streaming_stats_mean():
    s = StreamingStats()
    s.update_stats(np.array([1.0, 2.0, 3.0]))
    result = s.result()
    assert np.isclose(result["mean"], 2.0)

def test_streaming_stats_reset():
    s = StreamingStats()
    s.update_stats(np.array([1.0, 2.0]))
    s.reset()
    result = s.result()
    assert result["count"] == 0

def test_streaming_stats_quantile():
    s = StreamingStats()
    s.update_stats(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert np.isclose(s.quantile(0.5), 3.0)

def test_streaming_stats_histogram():
    s = StreamingStats()
    s.update_stats(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    counts, edges = s.histogram(bins=5)
    assert counts.sum() == 5

def test_streaming_stats_nan_ignored():
    s = StreamingStats()
    s.update_stats(np.array([1.0, np.nan, 3.0]))
    result = s.result()
    assert result["count"] == 2

def test_streaming_stats_window():
    s = StreamingStats(window_size=2)
    s.update_stats(np.array([1.0, 2.0]))
    s.update_stats(np.array([3.0, 4.0]))
    s.update_stats(np.array([5.0, 6.0]))
    result = s.result()
    assert result["count"] > 0