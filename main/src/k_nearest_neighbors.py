import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance
from operator import itemgetter
from sklearn import metrics

from pandas import read_csv
def k_nearest_neighbors(train_data, train_labels, test_data, k = 1):
    '''
    computes knn for each test point 

    :returns: predictions for test_data
    '''

    # find k nearest neighbors for each test point
    predictions = []
    for x in test_data:
        # distance
        cosine = np.dot(train_data, x)/(np.linalg.norm(train_data, axis=1)*np.linalg.norm(x))
        order = np.argsort(cosine)[::-1]
        ordered_labels = np.array(train_labels)[order]

        # get k nearest
        neighbors = []
        for n in range(k):
            neighbors.append(ordered_labels[n])

        # transform string to corresponding label number
        genres = {"blues":0, "classical":1, "country":2, 
                  "disco":3, "hiphop":4, "jazz":5, "metal":6, 
                  "pop":7, "reggae":8, "rock":9}
        neighbors = [genres[genre] for genre in neighbors]

        # find most present label
        count = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
        for label in neighbors:
            count[label] += 1
                
        # save prediction
        predictions.append(max(count, key=count.get))

    return predictions
