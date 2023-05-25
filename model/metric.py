import numpy as np
import sklearn.metrics
import torch


def accuracy(output: torch.Tensor, target: torch.Tensor) -> tuple:
    """Computes the accuracy of predictions.

    Args:
        output (torch.Tensor): The predicted output from the model.
        target (torch.Tensor): The true labels.

    Returns:
        tuple: A tuple containing the number of correct predictions, total predictions, and accuracy score.
    """
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct, len(target), correct / len(target)


def top_k_acc(output: torch.Tensor, target: torch.Tensor, k: int = 3) -> float:
    """Computes the top-k accuracy.

    Args:
        output (torch.Tensor): The predicted output from the model.
        target (torch.Tensor): The true labels.
        k (int, optional): The number of top predictions to consider. Default is 3.

    Returns:
        float: The top-k accuracy score.
    """
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def confusion_matrix(predictions: np.array, targets: np.array) -> np.array:
    """Computes the confusion matrix.

    Args:
        predictions (np.array): The predicted labels.
        targets (np.array): The true labels.

    Returns:
        np.array: The computed confusion matrix.
    """
    return sklearn.metrics.confusion_matrix(targets, predictions)




def sensitivity(cnf_matrix: np.array) -> np.array:
    """Computes the sensitivity from the confusion matrix.

    Args:
        cnf_matrix (np.array): The confusion matrix.

    Returns:
        np.array: The computed sensitivity values for each class.
    """

    eps = 1e-7
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    return TP / (TP + FN + eps)


def positive_predictive_value(cnf_matrix: np.array) -> np.array:
    """Computes the positive predictive value from the confusion matrix.

    Args:
        cnf_matrix (np.array): The confusion matrix.

    Returns:
        np.array: The computed positive predictive values for each class.
    """

    eps = 1e-7
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    return TP / (TP + FP + eps)
