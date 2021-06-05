from pprint import pprint

from mlp import mlp_main
from reader import Reader
from simple_kmeans import kmeans
from metr import purity, rand_index
from som_main import som_main

KMEANS = False
MLP = False
SOM = True

# Load Data
data_reader = Reader('data')
train_data, train_labels = data_reader.get_data('train')

# K-Means
if KMEANS:
    predicted_labels = kmeans(train_data)
    scores = {'purity': purity(predicted_labels, train_labels),
              'rand_index': rand_index(predicted_labels, train_labels)}
    print("K-Means Scores:")
    pprint(scores)

# SOM
if SOM:
    som_predicted_labels = som_main(train_data)
    som_scores = {'purity': purity(som_predicted_labels, train_labels),
                  'rand_index': rand_index(som_predicted_labels, train_labels)}

    print("SOM Scores:")
    pprint(som_scores)

# MLP-Customized
if MLP:
    SAMPLES = 60000
    mlp_predicted, real_labels = mlp_main(SAMPLES)

    scores = {'purity': purity(mlp_predicted, real_labels[:SAMPLES]),
              'rand_index': rand_index(mlp_predicted, real_labels[:SAMPLES])}
    print("MLP Scores:")
    pprint(scores)

    # print(mlp_predicted[:100])
    # print(train_labels[:100])

print("Done!")
