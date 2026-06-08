import numpy as np
from numcompute_stream.metrics import StreamingAccuracy, StreamingMetrics, StreamingConfusionMatrix

def test_streaming_accuracy_update_result():
    sa = StreamingAccuracy()
    sa.update(np.array([0,1,1]), np.array([0,1,0]))
    assert np.isclose(sa.result(), 2/3)

def test_streaming_accuracy_reset():
    sa = StreamingAccuracy()
    sa.update(np.array([0,1]), np.array([0,1]))
    sa.reset()
    assert sa.result() == 0.0

def test_streaming_confusion_matrix_accumulates():
    scm = StreamingConfusionMatrix()
    scm.update(np.array([0,1]), np.array([0,1]))
    scm.update(np.array([0,1]), np.array([1,0]))
    cm = scm.result()
    assert cm.sum() == 4

def test_streaming_metrics_all_keys():
    sm = StreamingMetrics()
    sm.update(np.array([0,1,0,1]), np.array([0,1,1,0]))
    result = sm.result()
    assert all(k in result for k in ["accuracy","precision","recall","f1"])

def test_streaming_metrics_reset():
    sm = StreamingMetrics()
    sm.update(np.array([0,1]), np.array([0,1]))
    sm.reset()
    result = sm.result()
    assert result["accuracy"] == 0.0