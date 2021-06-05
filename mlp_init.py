import math
import numpy as np
from scipy.spatial.distance import euclidean


def mlp_set_labels(data_points, num_of_classes, incoming_seed_pints=None):
    no_of_points = len(data_points)
    # init
    labels = np.zeros(no_of_points)

    # select
    rng = np.random.default_rng()
    seed_points = incoming_seed_pints if incoming_seed_pints is not None else rng.choice(no_of_points,
                                                                                         size=num_of_classes,
                                                                                         replace=False)  # indexes
    # compute distances
    for point_index, point in enumerate(data_points):
        min_dist = math.inf
        nearest_seed = 0
        for index, seed in enumerate(seed_points):
            dist = euclidean(point, data_points[seed])
            if dist < min_dist:
                nearest_seed = index
                min_dist = dist
        labels[point_index] = float(nearest_seed)

    return labels


def get_new_seed_points(data_points: [], labels: [], features_length=784):
    clusters = []

    for i in range(10):
        current_cluster = np.argwhere(labels == i).flatten()
        clusters.append(current_cluster)

    def compute_cluster_mean(cluster):
        cluster_sum = np.zeros(features_length)
        for data_index in cluster:
            cluster_sum += data_points[data_index]
        return cluster_sum / len(cluster)

    def find_seed_point(cluster, cluster_avg):
        min_dist = math.inf
        center_index = 0
        for data_index in cluster:
            dist = euclidean(cluster_avg, data_points[data_index])
            if dist < min_dist:
                min_dist = dist
                center_index = data_index

        return center_index

    clusters_averages = []
    for cluster in clusters:
        clusters_averages.append(compute_cluster_mean(cluster))

    new_seed_pints = []
    for index, cluster_indexes in enumerate(clusters):
        new_seed_pints.append(find_seed_point(cluster_indexes, clusters_averages[index]))

    return np.array(new_seed_pints)
