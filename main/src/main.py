'''
Main pipeline for the MGC.
May vary based on input args.
'''

#modules
import random
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics


#local methods
from feature_reduction import principal_component_analysis as pca
from feature_reduction import linear_discriminant_analysis as lda
from gaussian_naive_bayes import gaussian_naive_bayes as gnb
from discriminative import discriminative as disc
from k_nearest_neighbors import k_nearest_neighbors as knn
from arg_parser import parse_args


def split_data():
    '''
    divides the dataset into training and test set
    
    :returns: training_set, test_set and respective labels
    '''

    # load data
    data = np.load(f"../dataset/My_Data/pca_{args.features}.npy")
    data_labels = np.load(f"../dataset/My_Data/labels.npy")
    

    random.seed(0)
    ratio = int(args.ratio * 100)
    dataset = np.zeros(data.shape[0], dtype=np.short)
    for i in range(1, 11):
        min_ = (i-1)*100 - (i//6)
        max_ = i*100 - (i//6)
        train_choice = random.sample(range(min_, max_), ratio)
        dataset[train_choice] = 1

    # get training data
    train_idx = np.where(dataset == 1)
    train_data = data[train_idx]
    train_labels = data_labels[train_idx]

    # get test data 
    test_idx = np.where(dataset == 0)
    test_data = data[test_idx]
    true_labels = data_labels[test_idx]


    return train_data, train_labels, test_data, true_labels, dataset


def evaluation(predictions, true_labels):
    '''
    evaluates the model used for classification

    :returns: model accuracy
    '''

    # confusion matrix  
    genres = {"blues":0, "classical":1, "country":2, 
              "disco":3, "hiphop":4, "jazz":5, "metal":6, 
              "pop":7, "reggae":8, "rock":9}
        

    true_labels = np.array([genres[genre] for genre in true_labels])
    
    conf_matrix = metrics.confusion_matrix(true_labels, predictions)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"])

    cm_display.plot()
    plt.show()
    
    precision, recall, f_score, _ = metrics.precision_recall_fscore_support(true_labels, predictions)
    accuracy = np.sum(predictions == true_labels)/len(true_labels)

    print(f"Precisions: {precision}\nRecalls: {recall}\n F_scores: {f_score}\nAccuracy: {accuracy*100:.2f}%")

    return accuracy


def main():
    '''
    pipeline for music genre classification:
        - feature extraction(optional)
        - dataset splitting
        - classification + evaluation
    '''

    # feature extraction
    if args.reduction == "pca":
        reduced, original = pca()
        print(f"Dimensionality reduction through PCA: {original} -> {reduced}")
        print(f"File created: 'pca_{reduced}.npy' ")

    # split dataset
    train_data, train_labels, test_data, true_labels, dataset = split_data()

        
    # classification and model evaluation
    if args.classification == "bayes":
        predictions = gnb(train_data, train_labels, test_data)
        evaluation(predictions, true_labels)

    elif args.classification == "discriminative":
        predictions = disc(train_data, train_labels, test_data, args.features)
        evaluation(predictions, true_labels)

    elif args.classification == "knn":
        if not args.allknn:
            # input a value for k
            while True:
                try:
                    k = int(input(f"Value for the K in nearest neighbour approach({1} < k < {int(args.ratio*100)}) "))
                    break  # exit if input is converted to an integer
                except ValueError:
                    print("Invalid input. Please enter an integer.")

            k = max(min(k, args.ratio*100), 1)  #ensure  0 < k < test_data
            predictions = knn(train_data, train_labels, test_data, k)
            evaluation(predictions, true_labels)
            
        else:
            # discover best k value for classification
            accuracies = []
            for k in range(1,71):
                predictions = knn(train_data, train_labels, test_data, k)
                accuracies.append(evaluation(predictions, true_labels))
            
            best = np.argmax(k)
            print(f"Best K ={best + 1} with {accuracies[best]}% accuracy")


    return 0


if __name__ == '__main__':
    args = parse_args()
    main()