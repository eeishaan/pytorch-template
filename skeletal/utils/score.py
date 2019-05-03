'''
Module that contains scoring functions
'''

from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score
from sklearn.metrics.cluster import normalized_mutual_info_score


def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    return accuracy, f1, ari, nmi
