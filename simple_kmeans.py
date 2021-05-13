import numpy as np
from reader import Reader
from sklearn.cluster import KMeans


def kmeans():
    # Read Data
    data_reader = Reader('data')
    train_date, train_labels = data_reader.get_data('train')

    # Clustering Using Kmeans
    clusters_count = 10
    total_repeats = 15  # Count of total runs
    max_iterations = 500  # For single run

    k_means = KMeans(n_clusters=clusters_count, n_init=total_repeats, max_iter=max_iterations)
    fitted_data = k_means.fit(train_date)
    # center_of_clusters = fitted_data.cluster_centers_
    data_labels = np.array(fitted_data.labels_)

    # Plot center of clusters
    # data_reader.plot_images(center_of_clusters, len(center_of_clusters))

    # Plot first 25 images of each cluster
    images_count_to_show = 25
    for i in range(clusters_count):
        cluster_images = [train_date[index] for index in np.where(data_labels == i)[0][:images_count_to_show]]
        data_reader.plot_images(cluster_images[:images_count_to_show], images_count_to_show)

    return data_labels, np.array(train_labels)
