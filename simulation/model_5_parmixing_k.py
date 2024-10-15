"""
Monte Carlo Simulation for Clustering with the ECO Algorithm

This script performs a Monte Carlo simulation to evaluate the accuracy of clustering
using the ECO algorithm. The simulation is conducted across 
various sample sizes, with each iteration generating synthetic data, performing clustering, 
and measuring the exact recovery percentage of the clustering algorithm.

Dependencies:
-------------
- R environment with necessary functions (`model_3_parmixing_m.R`).
- Python packages: `numpy`, `pandas`, `matplotlib`, `multiprocessing`, `rpy2`, `eco_alg`.

Functions :
-----------
- `make_sample`: Generates synthetic data using R functions that create copula-based random samples.
- `operation_model_3_ECO`: Performs the core Monte Carlo simulation, including data generation, clustering, 
  and exact recovery calculation.
- `init_pool_processes`: Initializes the multiprocessing pool for parallel processing.
- Parallel Execution: The simulation runs multiple iterations in parallel, controlled by the variable `n_iter`.

Parameters:
-----------
- `d1`, `d2`: Dimensions of the two variable groups.
- `k_sample`: A list of sample sizes (number of block maxima) for the simulation.
- `m`: Block length parameter, used in sample generation and clustering.
- `n_iter`: Number of Monte Carlo iterations to run for each sample size.
- `p`: Probability parameter for random sampling.
- `mode`: Determines the model used for the simulation (set to "ECO" in this script).

Output:
-------
- The results of the simulation are saved as a CSV file (`results_model_3_ECO_1600_mixing_k_p0.9.csv`),
  containing the exact recovery percentages for each experiment.

Usage:
------
1. Set the desired parameters.
2. Run the script to execute the simulation.
3. The results will be printed and saved as CSV files.
"""

import eco_alg
import multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import rpy2.robjects as robjects
r = robjects.r
r['source']('model_5_parmixing_m.R')


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

    #sample = pd.DataFrame(sample)                      # FASTER but memory consuming
    #erank = np.array(sample.rank() / (dict['k'] + 1))
    #outer = (np.maximum(erank[:, :, None],
    #         erank[:, None, :])).sum(0) / dict['k']
    ## Calculate extreme coefficients
    #extcoeff = -np.divide(outer, outer - 1)
    #Theta = np.maximum(2 - extcoeff, 10e-5)

    # Step 7: Perform clustering based on the extremal correlation matrix Theta
    # The `clust` function clusters the dimensions based on their extremal correlations, using 'k' as the number of clusters
    O_hat = eco_alg.clust(Theta, n=dict['k'], m=dict['m'])

    # Step 8: Calculate the percentage of exact recovery (how accurately the clustering matches the true partition)
    perc = eco_alg.perc_exact_recovery(O_hat, O_bar)

    # Step 9: Return the exact recovery percentage
    return perc  # This value indicates how well the clustering algorithm was able to recover the original clusters


# Set key parameters for the Monte Carlo simulation
d = 200  # The total number of dimensions (features) in the generated sample
# Different sample sizes (k values) to evaluate
k_sample = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
# Block length parameter (used to compute the total number of samples n)
m = 20
n_iter = 10  # Number of Monte Carlo iterations for each value of k
K = 10  # Number of true clusters expected in the sample generation
p = 0.9  # Probability parameter used in the sample generation
# Simulation mode, controls which model to use (in this case, ECO)
mode = "ECO"

# Initialize a multiprocessing pool with 10 processes to run simulations in parallel
pool = mp.Pool(processes=10, initializer=init_pool_processes)

# Initialize an empty list to store the results for different values of 'k'
stockage = []

# Loop through each value of 'k' in 'k_sample', representing different sample sizes
for k in k_sample:

    # Step 1: Calculate the total number of samples (n) for this particular k
    # Total number of samples is k (sample size) multiplied by block length (m)
    n = k * m

    # Step 2: Define the input parameters for the simulation, bundled into a dictionary
    input = {'n_sample': n, 'd': d, 'k': k, 'm': m, 'p': p, 'K': K}

    # Step 3: Check if the mode is "ECO", which controls the behavior of the simulation
    if mode == "ECO":

        # Step 4: Run the 'operation_model_5_ECO' function in parallel using the multiprocessing pool
        # This will run the simulation 'n_iter' times for the current value of 'k', each time with a different seed (i)
        result_objects = [pool.apply_async(
            operation_model_5_ECO, args=(input, i)) for i in range(n_iter)]

    # Step 5: Gather the results from all the parallel processes using the 'get' method
    results = [r.get() for r in result_objects]

    # Step 6: Append the results for this value of 'k' to the 'stockage' list
    stockage.append(results)

    # Step 7: Convert the results (so far) into a pandas DataFrame for easier analysis and inspection
    df = pd.DataFrame(stockage)

    # Step 8: Print the current DataFrame to monitor progress
    print(df)

# Step 9: Once all the simulations are done, close the multiprocessing pool to prevent further task submissions
pool.close()

# Step 10: Wait for all processes in the pool to complete their execution
pool.join()

# Step 11: Save the final DataFrame, which contains the results for all k values, into a CSV file
df.to_csv('results_model_5_ECO_200_mixing_k_p0.9.csv')
