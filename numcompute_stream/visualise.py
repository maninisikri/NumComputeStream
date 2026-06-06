import numpy as np
import matplotlib.pyplot as plt


def plot_metric_over_time(metric_values, title="Metric over Time",
                          ylabel="Metric", save_path=None):
    """
    Plot a single metric recorded after each streaming chunk.

    Useful for tracking how accuracy or loss changes as the model
    sees more data over time.

    Parameters
    ----------
    metric_values : list of float
        One value per chunk, in order.
    title : str
        Title shown at the top of the plot.
    ylabel : str
        Label for the y-axis (e.g. "Accuracy", "Loss").
    save_path : str or None
        If given, saves the figure to this path instead of showing it.
        Example: "plots/accuracy.png"
    """
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(metric_values) + 1), metric_values, marker='o')
    plt.title(title)
    plt.xlabel("Chunk")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def compare_models(metric1, metric2, labels=("Model 1", "Model 2"),
                   title="Model Comparison", ylabel="Metric", save_path=None):
    """
    Compare two models by plotting their metrics side by side over chunks.

    Helpful for seeing whether a single tree or an ensemble does better
    as more data arrives in a streaming setting.

    Parameters
    ----------
    metric1 : list of float
        Per-chunk metric values for the first model.
    metric2 : list of float
        Per-chunk metric values for the second model.
    labels : tuple of str
        Display names for the two models, e.g. ("Decision Tree", "Random Forest").
    title : str
        Title shown at the top of the plot.
    ylabel : str
        Label for the y-axis.
    save_path : str or None
        If given, saves the figure to this path instead of showing it.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(metric1) + 1), metric1,
             marker='o', label=labels[0])
    plt.plot(range(1, len(metric2) + 1), metric2,
             marker='s', label=labels[1])
    plt.title(title)
    plt.xlabel("Chunk")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_predictions_vs_ground_truth(y_true, y_pred,
                                     title="Predictions vs Ground Truth",
                                     save_path=None):
    """
    Compare predicted labels against actual labels for the latest chunk.

    Useful at the end of each chunk to get a quick visual sense of
    where the model is getting things right or wrong.

    Parameters
    ----------
    y_true : array-like, shape (n,)
        Actual labels for the chunk.
    y_pred : array-like, shape (n,)
        Predicted labels for the chunk.
    title : str
        Title shown at the top of the plot.
    save_path : str or None
        If given, saves the figure to this path instead of showing it.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    plt.figure(figsize=(8, 4))
    plt.plot(y_true, label="Ground Truth", alpha=0.7)
    plt.plot(y_pred, label="Predictions", alpha=0.7, linestyle='--')
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Label")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()