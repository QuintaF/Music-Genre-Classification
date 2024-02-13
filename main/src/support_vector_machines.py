import numpy as np
from sklearn import svm

def support_vector_machines(train_data, train_labels, test_data):    
    '''
    computes knn for each test point 

    :returns: predictions for test_data
    '''

    svm_model = svm.SVC(kernel="rbf")
    svm_model.fit(train_data, train_labels)

    predictions = svm_model.predict(test_data)
    
    # transform string to corresponding label number
    genres = {"blues":0, "classical":1, "country":2, 
                "disco":3, "hiphop":4, "jazz":5, "metal":6, 
                "pop":7, "reggae":8, "rock":9}
    
    predictions = [genres[genre] for genre in predictions]

    return predictions