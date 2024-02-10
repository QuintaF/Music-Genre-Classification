'''
In this file are implemented functions 
for the dimensionality reduction of  data.
'''

#directory change ->  ...\Music-Genre-Classification\main\src
import os, sys
file_path = os.path.dirname(__file__)
os.chdir(file_path)

#modules
import numpy as np
import matplotlib.pyplot as plt 
from pandas import read_csv


# Principal Component Analysis
def principal_component_analysis():
    '''
    performs data reduction through PCA:
        - centering/standardization
        - covariance matrix
        - compute eigenvalues and eigenvectors
        - dimensionality choice (maintain 80% or more variance)
        - data projection

    During execution graphs for 2D/3D reduction are shown.
    '''

    data = np.load("../dataset/My_Data/features.npy")
    lab_colors = ["blue"]*100 + ["dimgray"]*100 + ["forestgreen"]*100 + ["cyan"]*100 + ["darkred"]*100 + ["mediumpurple"]*99 + ["black"]*100 + ["magenta"]*100 + ["orange"]*100 + ["red"]*100 

    # centering/standardization data
    means = np.mean(data, axis= 0)
    means_c = data - means
    std = np.std(data, axis=0)
    means_c = np.divide(means_c, std)

    # covariance
    cov = np.cov(means_c, rowvar=False)

    # eigenvalues/ eigenvectors
    lambdas, vectors = np.linalg.eigh(cov)
    ordered_evals = np.argsort(lambdas)[::-1]
    best_evals = lambdas[ordered_evals]  # 61
    best_evecs = vectors[:, ordered_evals] # 61x61

    # optimal dimensionality reduction
    optimal_sigma = np.cumsum(best_evals)/np.sum(best_evals)
    dim = optimal_sigma.shape[0]
    plt.figure("Features Reduction Graph")
    plt.subplot(2,1,1)
    plt.scatter(np.arange(1,dim+1),best_evals) 

    plt.subplot(2,1,2)
    plt.scatter(np.arange(1,dim+1),optimal_sigma)
    plt.show()

    sigma = len(np.where(optimal_sigma >= .8)[0])  # at least 80% variance
    
    # data projection
    T = best_evecs[:,:sigma]
    projected_data = np.dot(means_c, T)  # reducing data from 61 to sigma

    if sigma > 2:
        fig = plt.figure("Projected Data - 3D")
        ax = fig.add_subplot(projection="3d")
        ax.scatter(projected_data[:,0], projected_data[:,1], projected_data[:,2], c=lab_colors)
        plt.show()

    if sigma > 1:
        plt.figure("Projected Data - 2D")
        plt.scatter(projected_data[:,0], projected_data[:,1], c=lab_colors)
        plt.show()

    np.save(f"../dataset/My_Data/pca_{sigma}.npy", projected_data)

    return sigma, data.shape[1]