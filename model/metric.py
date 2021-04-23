import numpy as np
import sklearn.metrics
import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct, len(target), correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def confusion_matrix(predictions, targets):
    return sklearn.metrics.confusion_matrix(targets, predictions)


# def sensitivity(predictions, targets):
#     cnf_matrix = sklearn.metrics.confusion_matrix(targets, predictions)
#
#     #print(cnf_matrix)
#
#     FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
#     FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
#     TP = np.diag(cnf_matrix)
#     TN = cnf_matrix.sum() - (FP + FN + TP)
#
#     return TP / (TP + FN)
#
#
#
# def  positive_predictive_value(predictions, targets):
#     cnf_matrix = sklearn.metrics.confusion_matrix(targets, predictions)
#
#     #print(cnf_matrix)
#
#     FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
#     FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
#     TP = np.diag(cnf_matrix)
#     TN = cnf_matrix.sum() - (FP + FN + TP)
#
#     return TP / (TP + FP)


def sensitivity(cnf_matrix):
    # print(cnf_matrix)
    eps = 1e-7
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    return TP / (TP + FN + eps)


def positive_predictive_value(cnf_matrix):
    # print(cnf_matrix)
    eps = 1e-7
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    # print(TP)
    # print(FP)
    # print(TP / (TP + FP))
    # print(TP.sum() / (TP + FP).sum())
    return TP / (TP + FP + eps)
