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
  saved to 'results_model_4_ECO_200_mixing_p0.9_m.csv'.
"""

import eco_alg
import multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import rpy2.robjects as robjects
r = robjects.r
r['source']('model_4_parmixing_m.R')


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
    K (int)        : The number of clusters components to generate.

    Outputs
    -------
    np.array       : A NumPy array containing the generated sample data.
    O_bar (dict)   : A dictionary representing the ground-truth partitioning of the dimensions.
                     - Keys are cluster labels (starting from 1).
                     - Values are ranges (arrays of indices) indicating which dimensions belong to each cluster.
    """

    # Retrieve necessary R functions for copula-based sampling and observation
    # Function to generate copula model
    generate_copula = robjects.globalenv['generate_copula']
    # Function to generate random sample
    generate_randomness = robjects.globalenv['generate_randomness']
    # Function to process sample
    robservation = robjects.globalenv['robservation']

    # Step 2: Generate the copula model with 'd' dimensions and 'K' clusters
    copoc = generate_copula(d=d, K=K, seed=seed)
    sizes = copoc[1]  # Cluster sizes returned by copula model
    copoc = copoc[0]  # Actual copula model object

    # Step 3: Generate random samples using the copula model
    sample = generate_randomness(n_sample, p=p, d=d, seed=seed, copoc=copoc)

    # Step 4: Process the generated sample using the observation function
    data = robservation(sample, m=m, d=d, k=k)

    # Step 5: Construct the ground-truth partition based on cluster sizes
    O_bar = {}
    l = 0  # Cluster index
    _d = 0  # Dimension index tracking
    for d_ in sizes:
        l += 1
        O_bar[l] = np.arange(_d, _d + d_)  # Assign dimensions to cluster l
        _d += d_  # Update dimension index for next cluster

    # Step 6: Return the processed sample and the ground-truth partition
    return np.array(data), O_bar


def operation_model_4_ECO(dict, seed):
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

    # Set the random seed to ensure reproducibility of results. The seed is scaled by 1 to avoid seed clashes.
    np.random.seed(1 * seed)

    # Step 1: Generate a synthetic sample and the ground-truth partition (O_bar) of clusters.
    # The `make_sample` function generates the sample data and ground truth cluster labels (O_bar).
    # Inputs: number of samples (`n_sample`), total dimensions (`d`), and other clustering parameters (`k`, `m`, `p`, `K`).
    sample, O_bar = make_sample(n_sample=dict['n_sample'], d=dict['d'],
                                k=dict['k'], m=dict['m'], p=dict['p'],
                                seed=seed, K=dict['K'])

    # Step 2: Initialization
    # `d` is the total number of variables (dimensions) in the generated sample.
    d = sample.shape[1]

    # Step 3: Create an ECDF matrix `R` to hold the empirical cumulative distribution function (ECDF) values.
    # The matrix `R` has `k` rows (number of clusters) and `d` columns (one for each dimension).
    R = np.zeros([dict['k'], d])

    # Step 4: Compute the ECDF for each variable (dimension) in the sample.
    # For each column (dimension) in the sample, compute the ECDF and store it in the corresponding column of `R`.
    for j in range(0, d):
        # Extract the j-th variable from the sample (column vector)
        X_vec = sample[:, j]
        # Compute the ECDF of the j-th variable using the `ecdf` function from `eco_alg`.
        R[:, j] = eco_alg.ecdf(X_vec)

    # Step 5: Create the extremal correlation matrix `Theta`.
    # Initialize `Theta` as a symmetric matrix with ones. It will store the extremal correlations between pairs of variables.
    Theta = np.ones([d, d])

    # Step 6: Compute the extremal correlations between each pair of variables.
    # Loop through the lower triangle of the matrix to compute the extremal correlation between variables `i` and `j`.
    for j in range(0, d):
        for i in range(0, j):
            # Calculate extremal correlation between the i-th and j-th variables using `theta` from `eco_alg`.
            # `Theta[i, j]` and `Theta[j, i]` are updated symmetrically.
            Theta[i, j] = Theta[j, i] = 2 - eco_alg.theta(R[:, [i, j]])

    # Step 7: Apply the clustering algorithm to the extremal correlation matrix `Theta`.
    # The `clust` function from `eco_alg` performs clustering based on the extremal correlations in `Theta`.
    # `O_hat` is the predicted cluster assignment from the algorithm.
    O_hat = eco_alg.clust(Theta, n=dict['k'], m=dict['m'])

    # Step 8: Evaluate the exact recovery by comparing the predicted clusters (`O_hat`) with the ground truth (`O_bar`).
    # `perc_exact_recovery` computes the percentage of exact recovery, measuring how well the predicted clusters match the true clusters.
    perc = eco_alg.perc_exact_recovery(O_hat, O_bar)

    # Step 9: Return the exact recovery percentage.
    return perc


# Define the number of dimensions for the sample
d = 200  # Total number of dimensions for the data

# List of sample sizes (m values) to evaluate during the simulation
m_sample = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]

# Set total number of observations for the simulation
n = 10000  # Total number of samples to be drawn in each simulation

# Define the number of clusters or groups to consider
K = 5  # Number of clusters for the ECO algorithm

# Set the number of iterations for each value of 'm' during the simulation
n_iter = 10  # Number of iterations to run for each sample size

# Probability parameter that influences the random process
p = 0.9  # Probability parameter for generating samples

# Initialize a multiprocessing pool with 10 processes to run simulations in parallel
pool = mp.Pool(processes=10, initializer=init_pool_processes)

# Set the mode for the simulation, controlling which model to use (in this case, ECO)
mode = "ECO"

# Initialize an empty list to store results for different m values
stockage = []

# Loop through each value of 'm' in 'm_sample', representing different block lengths
for m in m_sample:

    # Calculate the number of clusters (k) based on the current sample size (m)
    # Determine number of clusters by dividing total samples by current block length
    k = int(np.floor(n / m))

    # Define the input parameters for the simulation, bundled into a dictionary
    input = {
        # Total samples for this iteration (k multiplied by m)
        'n_sample': k * m,
        'd': d,              # Total number of dimensions
        'k': k,              # Calculated number of clusters
        'm': m,              # Current block length
        'p': p,              # Probability parameter
        'K': K               # Fixed number of clusters for the ECO algorithm
    }

    # Check if the mode is "ECO", which controls the behavior of the simulation
    if mode == "ECO":
        # Run the 'operation_model_4_ECO' function in parallel using the multiprocessing pool
        # This will run the simulation 'n_iter' times for the current value of 'm', each time with a different seed (i)
        result_objects = [pool.apply_async(
            operation_model_4_ECO, args=(input, i)) for i in range(n_iter)]

    # Gather the results from all the parallel processes using the 'get' method
    results = [r.get() for r in result_objects]

    # Append the results for this value of 'm' to the 'stockage' list
    stockage.append(results)

    # Convert the results (so far) into a pandas DataFrame for easier analysis and inspection
    df = pd.DataFrame(stockage)

    # Print the current DataFrame to monitor progress and check the results of the simulations
    print(df)

# Close the multiprocessing pool to prevent further task submissions
pool.close()

# Wait for all processes in the pool to complete their execution
pool.join()

# Save the final DataFrame to a CSV file for future reference and analysis
# Output results to a CSV file
df.to_csv('results_model_4_ECO_200_mixing_p_0.9_m.csv')
