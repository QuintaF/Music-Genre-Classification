'''
Main pipeline for the MGC.
May vary based on input args.
'''

#modules
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics


#local methods
from feature_reduction import principal_component_analysis as pca
from gaussian_naive_bayes import gaussian_naive_bayes as gnb
from discriminative import discriminative as disc
from k_nearest_neighbors import k_nearest_neighbors as knn
from arg_parser import parse_args

# for 'random' dataset splitting
import random
SEED = 0


def split_data():
    '''
    divides the dataset into training and test set
    
    :returns: training_set, test_set and respective labels
    '''

    # load data
    data = np.load(f"../dataset/My_Data/pca_{args.features}.npy")
    data_labels = np.load(f"../dataset/My_Data/labels.npy")
    

    random.seed(SEED)
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


def evaluation(predictions, true_labels, name = None):
    '''
    evaluates the model used for classification

    :returns: model accuracy
    '''

    # labels from string to respective number
    genres = {"blues":0, "classical":1, "country":2, 
              "disco":3, "hiphop":4, "jazz":5, "metal":6, 
              "pop":7, "reggae":8, "rock":9}

    true_labels = np.array([genres[genre] for genre in true_labels])
    

    # plot confusion matrix
    conf_matrix = metrics.confusion_matrix(true_labels, predictions)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"])

    plt.rcParams['figure.figsize'] = [10, 7]
    cm_display.plot()

    if args.save and name is not None:
        plt.savefig(name[:-4] + ".jpg")

    plt.show()

    # compute evaluation metrics
    precisions, recalls, f_scores, _ = metrics.precision_recall_fscore_support(true_labels, predictions)
    accuracy = np.sum(predictions == true_labels)/len(true_labels)

    # terminal output
    genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    print(' '*15, end='')
    for genre in genres:
        print(f"{genre:^11}", end='')

    print(f"\n{'Precisions:':<15}", end='')
    for precision in precisions:
        print(f"{precision:^11.2f}", end='')

    print(f"\n{'Recalls:':<15}", end='')
    for recall in recalls:
        print(f"{recall:^11.2f}", end='')

    print(f"\n{'F-Scores:':<15}", end='')
    for fscore in f_scores:
        print(f"{fscore:^11.2f}", end='')

    print(f"\n{'Accuracy:':<15}{accuracy*100:.2f}%")

    return precisions, recalls, f_scores, accuracy


def save_output(dataset, name, eva):
    '''
    saves infos about dataset and evaluation in a text file
    '''
    
    try:
        # get filenames
        genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
        train_files = []
        test_files = []
        for i in range(1, 11):
            min_ = (i-1)*100 - (i//6)
            max_ = i*100 - (i//6)
            a = []
            b = []
            for j, value in enumerate(dataset[min_:max_]):
                if value == 0:
                    a.append(f"{genres[i-1]}.{j:0>5}.wav ")
                else:
                    b.append(f"{genres[i-1]}.{j:0>5}.wav ")

            train_files.append(a)
            test_files.append(b) 
            
        # write on file
        with open(name, "w") as file:
            file.write(f" - - - TRAINING SET - - -\n")
            for files in range(0, 10):
                file.write(genres[files].capitalize() + ":\n\t")
                file.writelines(train_files[files] + ["\n"])

            file.write(f"\n - - -   TEST SET   - - -\n")
            for files in range(0, 10):
                file.write(genres[files].capitalize() + ":\n\t")
                file.writelines(test_files[files] + ["\n"])
            
            if eva is not None:
                file.write("\n - - - EVALUATION - - - \n")
                file.write(" "*15)
                for genre in genres:
                    file.write(f"{genre:^11}")
                
                file.write(f"\n{'Precisions:':<15}")
                for precision in eva[0]:
                    file.write(f"{precision:^11.2f}")

                file.write(f"\n{'Recalls:':<15}")
                for recall in eva[1]:
                    file.write(f"{recall:^11.2f}")

                file.write(f"\n{'F-Scores:':<15}")
                for fscore in eva[2]:
                    file.write(f"{fscore:^11.2f}")

                file.writelines(f"\n{'Accuracy:':<15}{eva[3]*100:.2f}%")

    except (FileNotFoundError, PermissionError, OSError):
        print(f"Error:error while opening file at {name}")
        return 1

    return 0


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
        

    # output filse name
    if args.classification == 'knn':
        name = f"../output/out_rng{SEED}_KNN_{args.allknn}_pca{args.features}_ratio{args.ratio}.txt"
    else:
        name = f"../output/out_rng{SEED}_{args.classification}_pca{args.features}_ratio{args.ratio}.txt"
        

    # classification and model evaluation
    if args.classification == "bayes":
        predictions = gnb(train_data, train_labels, test_data)
        eva = evaluation(predictions, true_labels, name)

    elif args.classification == "discriminative":
        predictions = disc(train_data, train_labels, test_data, args.features)
        eva = evaluation(predictions, true_labels, name)

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
            eva = evaluation(predictions, true_labels, name)
            
        else:
            # discover best k value for classification
            accuracies = []
            for k in range(1,71):
                predictions = knn(train_data, train_labels, test_data, k)
                accuracies.append(evaluation(predictions, true_labels))
            
            best = np.argmax(k)
            print(f"Best K ={best + 1} with {accuracies[best]}% accuracy")
            eva = None

    if args.save:
        save_output(dataset, name, eva)
    
    return 0


if __name__ == '__main__':
    args = parse_args()
    main()