# Music Genre Classification

Music genre classification using various Machine Learning techniques. The project showcases various implementations of Machine Learning techniques to address the problem of Music Genre Classification (MGC). Among these implementations are: a Naive Bayes Classifier, a Discriminant Machine, a K-Nearest Neighbour and lastly a Support Vector Machine. This README is just a guide to the program execution. A more technical review is the pdf file.


## Usage
Start the algorithm:
```
usage: path/to/main.py [-h] [--reduction REDUCTION] [--classification {None,bayes,discriminative,knn,svm}] [--allknn] [--features FEATURES] [--ratio RATIO] [--seed SEED] [--save]
```

Usage options:
```
options:
  -h, --help            show this help message and exit

  --reduction REDUCTION, -rd REDUCTION
                        choose pca variance between 0.45 and 0.9 for feature extraction, or skip feature extraction

  --classification {None,bayes,discriminative,knn,svm}, -cl {None,bayes,discriminative,knn,svm}
                        choose classification model

  --allknn, -ak         if knn is the classification model it computes for all possible k values and returns the best

  --features FEATURES, -ft FEATURES
                        choose data file based on number of features; if it doesn't exist an error is shown

  --ratio RATIO, -rt RATIO
                        a ratio between 0.45 and 0.9 for deciding how large the training set will be

  --seed SEED, -sd SEED
                        choose the seed for the random dataset split generator

  --save, -sv           save information about execution in an output file
```


### Default Execution
The default execution, classifies using the pca_41.py file with the bayes classifier and returns some evaluation parameters(precision, recall, f1-score and accuracy) without saving them.<br>


## Repository Structure

main/\
├── dataset/\
│&emsp;&emsp;├── Data/genres_original &emsp;...contains GTZAN audio files divided by genre.\
│&emsp;&emsp;└── MyData/&emsp;...contains extracted feature files and pca reduced feature files.\
│&emsp;&emsp;&emsp;&emsp;&emsp;└── mel_spectrograms/&emsp;...contains mel spectrograms images, although image classification is not implemented yet.\
│\
├── output/&emsp;...contains output files for certain tests.\
│\
└── src/\
&emsp;&emsp;├── arg_parser.py&emsp;...python parser for command line arguments.\
&emsp;&emsp;├── discriminative.py&emsp;...algorithm for Discriminant Machine classifier.\
&emsp;&emsp;├── feature_extraction.py&emsp;...handcrafted feature extraction from the audio files (not necessary for classification).\
&emsp;&emsp;├── feature_extraction.py&emsp;...PCA algorithm for feature reduction.\
&emsp;&emsp;├── gaussian_naive_bayes.py&emsp;...algorithm for Naive Bayes classifier.\
&emsp;&emsp;├── k_nearest_neighbors.py&emsp;...algorithm for KNN classifier.\
&emsp;&emsp;├── main.py&emsp;...pipeline for the classification process.\
&emsp;&emsp;└── support_vector_machines.py&emsp;...algorithm for SVM classifier.\

## Some Results

### NB
file: pca_38.npy
|            |  Blues  | Classical | Country | Disco   | Hip-Hop | Jazz    | Metal   | Pop     | Reggae  | Rock    | 
|:----------:|:-------:|:---------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| Precision | 0.71 | 0.88 | 0.74 | 0.56 | 0.55  | 0.79 | 0.88 | 0.67 | 0.38 | 0.55 |  
|    Recall | 0.80 | 0.77 | 0.57 | 0.50 | 0.70 | 0.77 | 0.50 | 0.60 | 0.60 | 0.57 |   
|    F1-Score | 0.75 | 0.82 | 0.64 | 0.53 | 0.62 | 0.78 | 0.64 | 0.63 | 0.46 | 0.56 |   
|    Accuracy | 63.67\% 

### DM
file: pca_38.npy

|            |  Blues  | Classical | Country | Disco   | Hip-Hop | Jazz    | Metal   | Pop     | Reggae  | Rock    | 
|:----------:|:-------:|:---------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| Precision | 0.71 | 0.88 | 0.74 | 0.56 | 0.55  | 0.79 | 0.88 | 0.67 | 0.38 | 0.55 |  
|    Recall | 0.80 | 0.77 | 0.57 | 0.50 | 0.70 | 0.77 | 0.50 | 0.60 | 0.60 | 0.57 |   
|    F1-Score | 0.75 | 0.82 | 0.64 | 0.53 | 0.62 | 0.78 | 0.64 | 0.63 | 0.46 | 0.56 |   
|    Accuracy | 63.67\% 

### KNN
file: pca_49.npy
|            |  Blues  | Classical | Country | Disco   | Hip-Hop  | Jazz    | Metal   | Pop     | Reggae  | Rock    | 
|:----------:|:-------:|:---------:|:-------:|:-------:|:--------:|:-------:|:-------:|:-------:|:-------:|:-------:|
|  Precision | 0.88 | 0.88 | 0.58 | 0.54 | 0.67 | 0.93 | 0.83 | 0.71 | 0.68 | 0.61 |   
|   Recall   | 0.77 | 0.93 | 0.60 | 0.50 | 0.73 | 0.87 | 0.80 | 0.73 | 0.70 | 0.63 | 
|  F1-Score  | 0.82 | 0.90 | 0.59 | 0.52 | 0.70 | 0.90 | 0.81 | 0.72 | 0.69 | 0.62 |   
|  Accuracy  | 72.67\%

### SVM
file: pca_49.npy
|            |  Blues  | Classical | Country | Disco   | Hip-Hop  | Jazz    | Metal   |   Pop   | Reggae  | Rock    | 
|:----------:|:-------:|:---------:|:-------:|:-------:|:--------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| Precision  | 0.88    |   0.93    |   0.74  |  0.58   |   0.81   |  0.79   |  0.83   |  0.75   |  0.71   |  0.60   | 
| Recall     | 0.77    |   0.93    |   0.77  |   0.70  |   0.70   |  0.87   |  0.80   |  0.70   |  0.73   |  0.60   | 
| F1-Score   | 0.82    |   0.93    |   0.75  |  0.64   |   0.75   |  0.83   |  0.81   |  0.72   |  0.72   |  0.60   | 
| Accuracy   | 75.67%