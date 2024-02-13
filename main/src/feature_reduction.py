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


# Principal Component Analysis
def principal_component_analysis(variance = .8):
    '''
    performs data reduction through PCA:
        - centering/standardization
        - covariance matrix
        - compute eigenvalues and eigenvectors
        - dimensionality choice (maintain 80% or more variance)
        - data projection

    During execution graphs for 2D/3D reduction are shown.
    '''

    data = np.load(f"../dataset/My_Data/features.npy")

    # centering/standardization of training data
    means_c = centering(data)

    # covariance
    cov = np.cov(means_c, rowvar=False)

    # eigenvalues/ eigenvectors
    lambdas, vectors = np.linalg.eigh(cov)
    ordered_evals = np.argsort(lambdas)[::-1]
    best_evals = lambdas[ordered_evals]  # features: 61
    best_evecs = vectors[:, ordered_evals] # 61x61

    # optimal dimensionality reduction
    optimal_sigma = np.cumsum(best_evals)/np.sum(best_evals)
    dim = optimal_sigma.shape[0]
    plt.figure("Features Reduction Graph")
    plt.subplot(2,1,1)
    plt.scatter(np.arange(1,dim+1),best_evals) 

    plt.subplot(2,1,2)
    plt.scatter(np.arange(1,dim+1),optimal_sigma)
    plt.axhline(variance, color="red")
    plt.show()

    sigma = len(np.where(optimal_sigma >= variance)[0])  # keep at least k variance
    
    # data projection
    T = best_evecs[:,:sigma]
    projected_data = np.dot(means_c, T)  # reducing data from 61 to sigma

    plot(projected_data, sigma)

    np.save(f"../dataset/My_Data/pca_{sigma}.npy", projected_data)

    return sigma, data.shape[1]


def centering(data):
    '''
    normalizes and centers data

    :returns: centered data
    '''
    # centering/standardization data
    means = np.mean(data, axis= 0)
    means_c = data - means
    std = np.std(data, axis=0)
    means_c = np.divide(means_c, std)

    return means_c


def plot(data, sigma):
    '''
    scatter plot of data in 2 and 3 dimensions (if possible)
    '''

    ratios = [100, 100, 100, 100, 100, 99, 100, 100, 100, 100]
    colors = np.repeat(["blue", "dimgray", "forestgreen", "cyan", "darkred", "mediumpurple", "black", "magenta", "orange", "red"], ratios)

    if sigma > 2:
        fig = plt.figure(f"Projected Data - 3D")
        ax = fig.add_subplot(projection="3d")
        ax.scatter(data[:,0], data[:,1], data[:,2], c=colors)

    if sigma > 1:
        plt.figure(f"Projected Data - 2D")
        plt.scatter(data[:,0], data[:,1], c=colors)

        
    plt.show()