import numpy as np
from reader import Reader
from sklearn.cluster import KMeans


def kmeans(train_data):
    data_reader = Reader('data')

    # Configs
    clusters_count = 10
    total_repeats = 30  # Count of total runs
    max_iterations = 300  # For single run
    tolerance = 0.01

    # Init
    k_means = KMeans(n_clusters=clusters_count, n_init=total_repeats, max_iter=max_iterations, tol=tolerance)

    # Fit Data
    fitted_data = k_means.fit(train_data)
    data_labels = np.array(fitted_data.labels_)

    # Show Clusters
    images_count_to_show = 25
    for i in range(clusters_count):
        cluster_images = [train_data[index] for index in np.where(data_labels == i)[0][:images_count_to_show]]
        data_reader.plot_images(cluster_images[:images_count_to_show], images_count_to_show)

    return data_labels
