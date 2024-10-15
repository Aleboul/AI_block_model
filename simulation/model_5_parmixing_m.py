"""
Monte Carlo Simulation for Clustering Accuracy Evaluation Using the ECO Algorithm

This script implements a Monte Carlo simulation to assess the clustering accuracy 
of the ECO algorithm by generating synthetic samples based on copula-based methods 
using R functions. It utilizes multiprocessing for efficient computation across 
multiple iterations and block sizes.

Key Functions:
---------------
1. `make_sample(n_sample, d, k, m, p, seed)`: 
   - Generates synthetic samples using R functions for copula-based sampling.
   - Inputs:
     - n_sample (int): Number of samples to generate.
     - d (int): Number of dimensions (variables) in the data.
     - k (int): Related to the number of block maxima.
     - m (int): Length of block maxima.
     - p (float): Probability parameter for random generation.
     - seed (int): Random seed for reproducibility.
   - Returns: A NumPy array containing generated sample data.

2. `operation_model_3_ECO(dict, seed)`:
   - Executes a Monte Carlo simulation to evaluate clustering accuracy.
   - Inputs:
     - dict (dict): Contains parameters for clustering, including:
       - d1 (int): Dimension of the first sample group.
       - d2 (int): Dimension of the second sample group.
       - n_sample (int): Total number of samples.
       - k (int): Parameter for block maxima.
       - m (int): Block length parameter.
       - p (float): Probability parameter.
     - seed (int): Seed for random number generation.
   - Returns: Percentage of exact recovery of clusters.

Dependencies:
--------------
- rpy2: To interface with R and execute R scripts.
- numpy: For numerical operations and handling arrays.
- pandas: For data manipulation and analysis.
- matplotlib: For plotting (not utilized in this script, but imported).
- multiprocessing: For parallel processing.
- eco_alg: Custom module containing functions for extremal correlation and clustering.

Usage:
------
1. Ensure the R script 'model_3_parmixing_m.R' is in the same directory and 
   contains the necessary R functions for sampling and copula generation.
2. Run the script to perform simulations across various sample sizes and 
   save results to a CSV file.

Output:
-------
- The results of the simulations are stored in a pandas DataFrame and 
  saved to 'results_model_5_ECO_200_mixing_p0.9_m.csv'.
"""

import rpy2.robjects as robjects
r = robjects.r
r['source']('model_5_parmixing_m.R')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing as mp
import eco_alg

def init_pool_processes():
    np.random.seed()

def make_sample(n_sample, d, k, m, p, seed, K):
    """
    Generates a synthetic sample using R functions for copula-based sampling and 
    randomness generation, and computes ground-truth clusters based on the generated data.

    Inputs
    ------
    n_sample (int) : The number of samples (observations) to generate.
    d (int)        : The total number of dimensions or variables in the dataset.
    k (int)        : A parameter related to the number of block maxima (affecting block maxima).
    m (int)        : Block length parameter.
    p (float)      : Probability parameter influencing the random process.
    seed (int)     : Random seed for reproducibility.
    K (int)        : The number of cluster to generate.

    Outputs
    -------
    np.array       : A NumPy array containing the generated sample data.
    O_bar (dict)   : A dictionary representing the ground-truth partitioning of the dimensions.
                     - Keys are cluster labels (starting from 1).
                     - Values are ranges (arrays of indices) indicating which dimensions belong to each cluster.
    """

    # Retrieve necessary R functions from the global environment for copula-based sampling
    # Function to generate a copula model
    generate_copula = robjects.globalenv['generate_copula']
    # Function to generate random samples
    generate_randomness = robjects.globalenv['generate_randomness']
    # Function to process generated samples
    robservation = robjects.globalenv['robservation']

    # Step 1: Generate the copula model with specified dimensions 'd' and 'K' clusters
    copoc = generate_copula(d=d, K=K, seed=seed)

    # Extract the sizes of each cluster from the generated copula
    sizes = copoc[1]  # Cluster sizes returned by the copula model
    copoc = copoc[0]  # Actual copula model object used for generating samples

    # Step 2: Generate random samples using the copula model
    sample = generate_randomness(n_sample, p=p, d=d, seed=seed, copoc=copoc)

    # Step 3: Process the generated sample using the observation function to obtain structured data
    data = robservation(sample, m=m, d=d, k=k)

    # Step 4: Construct the ground-truth partitioning (O_bar) based on the sizes of the clusters
    O_bar = {}  # Initialize a dictionary to hold the clusters
    l = 0  # Cluster index
    _d = 0  # Dimension index tracking

    # Iterate over the sizes of the clusters to assign dimensions to each cluster
    for d_ in sizes:
        l += 1  # Increment the cluster index
        O_bar[l] = np.arange(_d, _d + d_)  # Assign dimensions to cluster 'l'
        _d += d_  # Update the dimension index for the next cluster

    # Step 5: Return the processed sample data and the ground-truth partitioning
    # Convert data to NumPy array before returning
    return np.array(data), O_bar


def operation_model_5_ECO(dict, seed):
    """
    Performs a Monte Carlo simulation to evaluate the accuracy of clustering using the ECO algorithm (Extremal Clustering).

    Input Parameters
    ----------------
    dict : dict
        A dictionary containing the following keys:
        - 'd1' (int)       : Dimension (number of variables) of the first group of the sample.
        - 'd2' (int)       : Dimension (number of variables) of the second group of the sample.
        - 'n_sample' (int) : Total number of samples (observations) to generate.
        - 'd' (int)        : Total number of dimensions (variables) in the data.
        - 'k' (int)        : Parameter related to the number of block maxima (size of clusters).
        - 'm' (int)        : Parameter related to block length (used in generating synthetic data).
        - 'p' (float)      : Probability parameter influencing the random process.
        - 'K' (int)        : Number of true clusters in the data (used to simulate ground truth).

    seed : int
        Seed for random number generation, ensuring reproducibility.

    Returns
    -------
    perc : float
        The percentage of exact recovery, which measures how well the clustering algorithm recovers the true partition
        (ground-truth clusters) of the data.
    """

    # Set the random seed for reproducibility, scaled by the provided seed
    np.random.seed(1 * seed)

    # Step 1: Generate a synthetic sample using the provided parameters and R-based functions
    # The function `make_sample` returns the generated sample and the ground truth clusters (O_bar)
    sample, O_bar = make_sample(n_sample=dict['n_sample'], d=dict['d'], k=dict['k'],
                                m=dict['m'], p=dict['p'], seed=seed, K=dict['K'])

    # Step 2: Initialize the number of dimensions based on the sample shape
    # 'd' is the total number of dimensions in the generated sample
    d = sample.shape[1]

    # Step 3: Initialize an empty matrix R to hold empirical cumulative distribution function (ECDF) values
    # R will have 'k' rows (number of clusters) and 'd' columns (dimensions)
    R = np.zeros([dict['k'], d])

    # Step 4: Calculate the ECDF for each dimension in the sample
    for j in range(0, d):  # Iterate over each dimension
        # Extract the j-th variable (column) from the sample
        X_vec = sample[:, j]
        # Compute the ECDF for the j-th variable and store it in R
        R[:, j] = eco_alg.ecdf(X_vec)

    # Step 5: Initialize the extremal correlation matrix Theta
    # Create a matrix to store extremal correlations (initialized to 1)
    Theta = np.ones([d, d])

    # Step 6: Calculate the extremal correlation between each pair of dimensions
    for j in range(0, d):  # Iterate over each dimension
        for i in range(0, j):  # Only iterate over the lower triangle to avoid redundant calculations
            # Compute the extremal correlation between dimensions i and j using the theta function
            Theta[i, j] = Theta[j, i] = 2 - eco_alg.theta(R[:, [i, j]])

    # Step 7: Perform clustering based on the extremal correlation matrix Theta
    # The `clust` function clusters the dimensions based on their extremal correlations, using 'k' as the number of clusters
    O_hat = eco_alg.clust(Theta, n=dict['k'], m=dict['m'])

    # Step 8: Calculate the percentage of exact recovery (how accurately the clustering matches the true partition)
    perc = eco_alg.perc_exact_recovery(O_hat, O_bar)

    # Step 9: Return the exact recovery percentage
    return perc  # This value indicates how well the clustering algorithm was able to recover the original clusters

d = 200
m_sample = [3,6,9,12,15,18,21,24,27,30]
n = 10000
n_iter = 10 #100
K = 10
p = 0.9
pool = mp.Pool(processes= 10, initializer=init_pool_processes)
mode = "ECO"

stockage = []

for m in m_sample:

    k = int(np.floor(n / m))

    input = {'n_sample' : k * m, 'd' : d, 'k' : k, 'm' : m, 'p' : p, 'K' : K}

    if mode == "ECO":

        result_objects = [pool.apply_async(operation_model_5_ECO, args = (input,i)) for i in range(n_iter)]

    results = [r.get() for r in result_objects]

    stockage.append(results)

    df = pd.DataFrame(stockage)

    print(df)

pool.close()
pool.join()


df.to_csv('results_model_5_ECO_200_mixing_p0.9_m.csv')