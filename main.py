from pprint import pprint

from simple_kmeans import kmeans
from metr import purity, rand_index

predicted_labels, real_labels = kmeans()
scores = {'purity': purity(predicted_labels, real_labels), 'rand_index': rand_index(predicted_labels, real_labels)}

print("Scores:")
pprint(scores)
