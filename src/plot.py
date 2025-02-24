import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    RocCurveDisplay
)


def plot_confusion_matrix(
        y_test: np.ndarray, y_pred: np.ndarray,
        path: str
        ) -> None:
    """Save confusion matrix based on prediction.

    Args:
        y_test (np.ndarray): true labels
        y_pred (np.ndarray): predicted labels
        path (str): path to save plot
    """
    cm = confusion_matrix(y_test, y_pred)
    cm_plot = ConfusionMatrixDisplay(cm)
    cm_plot.plot()
    plt.savefig(path)


def save_cls_report(
        y_test: np.ndarray, y_pred: np.ndarray,
        path: str, verbose: bool = True
        ) -> None:
    """Save classification report to csv based
    on prediction.

    Args:
        y_test (np.ndarray): true labels
        y_pred (np.ndarray): predicted labels
        path (str): path to save report
        verbose (bool, optional): 
            if True, print classification report in terminal.
            Default to True.
    """
    cls = classification_report(y_test, y_pred, output_dict=True)
    cls = pd.DataFrame(cls).transpose().round(2)
    cls.to_csv(path)

    if verbose:
        print(classification_report(y_test, y_pred))


def plot_roc(
        y_test: np.ndarray, y_pred: np.ndarray,
        path: str
        ) -> None:
    """Save ROC curve based on prediction

    Args:
        y_test (np.ndarray): true labels
        y_pred (np.ndarray): predicted labels
        path (str): path to save ROC curve
    """
    RocCurveDisplay.from_predictions(y_test, y_pred)
    plt.savefig(path)
