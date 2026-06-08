import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
from numcompute_stream.visualise import (
    plot_metric_over_time,
    compare_models,
    plot_predictions_vs_ground_truth,
)

def test_plot_metric_over_time_runs():
    plot_metric_over_time([0.5, 0.6, 0.7], save_path="/tmp/test_metric.png")

def test_compare_models_runs():
    compare_models([0.5, 0.6], [0.4, 0.55], save_path="/tmp/test_compare.png")

def test_plot_predictions_runs():
    plot_predictions_vs_ground_truth(
        [0, 1, 0, 1], [0, 1, 1, 1],
        save_path="/tmp/test_preds.png"
    )

def test_plot_metric_empty_list():
    plot_metric_over_time([], save_path="/tmp/test_empty.png")

def test_compare_models_custom_labels():
    compare_models(
        [0.7, 0.8], [0.6, 0.75],
        labels=("Tree", "Forest"),
        save_path="/tmp/test_labels.png"
    )