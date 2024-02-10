import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from scipy.stats import multivariate_normal


def gaussian_naive_bayes(train_data, train_labels, test_data):
    '''
    computes log_likelihood between each class and test point

    :returns: predictions for test_data
    '''

    # class divided train data
    blues_train = train_data[np.where(train_labels == "blues")]
    classical_train = train_data[np.where(train_labels == "classical")]
    country_train = train_data[np.where(train_labels == "country")]
    disco_train = train_data[np.where(train_labels == "disco")]
    hiphop_train = train_data[np.where(train_labels == "hiphop")]
    jazz_train = train_data[np.where(train_labels == "jazz")]
    metal_train = train_data[np.where(train_labels == "metal")]
    pop_train = train_data[np.where(train_labels == "pop")]
    reggae_train = train_data[np.where(train_labels == "reggae")]
    rock_train = train_data[np.where(train_labels == "rock")]

    # model building
    mean_blues = np.mean(blues_train, axis=0)
    cov_blues = np.cov(blues_train, rowvar=False)
    mean_classical = np.mean(classical_train, axis=0)
    cov_classical = np.cov(classical_train, rowvar=False) 
    mean_country = np.mean(country_train, axis=0)
    cov_country = np.cov(country_train, rowvar=False) 
    mean_disco = np.mean(disco_train, axis=0)
    cov_disco = np.cov(disco_train, rowvar=False) 
    mean_hiphop = np.mean(hiphop_train, axis=0)
    cov_hiphop = np.cov(hiphop_train, rowvar=False) 
    mean_jazz = np.mean(jazz_train, axis=0)
    cov_jazz = np.cov(jazz_train, rowvar=False)
    mean_metal = np.mean(metal_train, axis=0)
    cov_metal = np.cov(metal_train, rowvar=False) 
    mean_pop = np.mean(pop_train, axis=0)
    cov_pop = np.cov(pop_train, rowvar=False) 
    mean_reggae = np.mean(reggae_train, axis=0)
    cov_reggae = np.cov(reggae_train, rowvar=False) 
    mean_rock = np.mean(rock_train, axis=0)
    cov_rock = np.cov(rock_train, rowvar=False) 

    # likelihoods(prior 1/10 for each class)
    lik1 = multivariate_normal.pdf(test_data, mean_blues, cov_blues) * 1/10
    lik2 = multivariate_normal.pdf(test_data, mean_classical, cov_classical) * 1/10
    lik3 = multivariate_normal.pdf(test_data, mean_country, cov_country) * 1/10
    lik4 = multivariate_normal.pdf(test_data, mean_disco, cov_disco) * 1/10
    lik5 = multivariate_normal.pdf(test_data, mean_hiphop, cov_hiphop) * 1/10
    lik6 = multivariate_normal.pdf(test_data, mean_jazz, cov_jazz) * 1/10
    lik7 = multivariate_normal.pdf(test_data, mean_metal, cov_metal) * 1/10
    lik8 = multivariate_normal.pdf(test_data, mean_pop, cov_pop) * 1/10
    lik9 = multivariate_normal.pdf(test_data, mean_reggae, cov_reggae) * 1/10
    lik10 = multivariate_normal.pdf(test_data, mean_rock, cov_rock) * 1/10
    
    # may encounter dvision by 0 warning, since it is not a problem the warning is silenced
    np.seterr(divide = 'ignore') 
    loglik = np.log(np.vstack((lik1, lik2, lik3, lik4, lik5, lik6, lik7, lik8, lik9, lik10)))
    
    # predictions
    predictions = np.argmax(loglik, axis=0)

    return predictions