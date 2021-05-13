import numpy as np
from sklearn import metrics
from sklearn.metrics.cluster import adjusted_rand_score


def purity(predicted, real):
    contingency_matrix = metrics.cluster.contingency_matrix(real, predicted)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def rand_index(predicted, real):
    return adjusted_rand_score(real, predicted)
