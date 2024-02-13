import numpy as np


def support_vector_machines(features):
    data = np.load(f"../dataset/My_Data/pca_{features}.npy")
    labels = np.load(f"../dataset/My_Data/labels.npy")
    

    training_data = []
    train_labels = []
    for i in range(170, 570, 100):
        training_data.extend(data[i-170:i-100])
        train_labels.extend(labels[i-170:i-100])

    training_data.extend(data[i-170:i-1])
    train_labels.extend(labels[i-170:i-1])

    for i in range(669, 1069, 100):
        training_data.extend(data[i-70:i])
        train_labels.extend(labels[i-70:i])

    # test data
    test_data = np.vstack((data[70:100], data[170:200], data[270:300], data[370:400], data[470:500], data[569:599], data[669:699], data[769:799], data[869:899], data[969:999]))
    gold_labels = np.repeat(["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"],30)
