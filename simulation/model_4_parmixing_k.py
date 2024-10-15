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


# Set the number of dimensions for the sample
d = 200  # Total number of dimensions (variables) in the generated sample

# Define a list of sample sizes (k values) to be tested in the simulation
k_sample = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

# Set parameters for the simulation
m = 20  # Block length parameter used in generating samples
K = 5   # Number of true clusters expected in the data
n_iter = 100  # Number of Monte Carlo iterations to run for each sample size
p = 0.9  # Probability parameter influencing the random process

# Initialize a multiprocessing pool with 10 parallel processes
pool = mp.Pool(processes=10, initializer=init_pool_processes)

# Set the mode of operation; "ECO" indicates using the Extremal Clustering algorithm
mode = "ECO"

# Initialize an empty list to store results from all iterations
stockage = []

# Loop through each value of 'k' in 'k_sample' to run simulations for different sample sizes
for k in k_sample:
    # Calculate the total number of samples (observations) for the current value of 'k'
    n = k * m  # Total samples is 'k' times the block length 'm'

    # Create a dictionary to hold input parameters for the simulation
    input = {'n_sample': n, 'd': d, 'k': k, 'm': m, 'p': p, 'K': K}

    # If the mode is set to "ECO", run the simulation using the ECO model
    if mode == "ECO":
        # Use the multiprocessing pool to apply the 'operation_model_4_ECO' function in parallel.
        # For each of the 'n_iter' iterations, pass the input parameters and a unique seed (i).
        result_objects = [pool.apply_async(
            operation_model_4_ECO, args=(input, i)) for i in range(n_iter)]

    # Gather results from all parallel processes once they complete
    results = [r.get() for r in result_objects]

    # Append the results for this value of 'k' to the 'stockage' list
    stockage.append(results)

    # Convert the list of results for the current 'k' into a pandas DataFrame for easy manipulation and inspection
    df = pd.DataFrame(stockage)

    # Print the DataFrame to check the results for each iteration
    print(df)

# Clean up the multiprocessing pool to prevent any further task submissions
pool.close()

# Wait for all processes in the pool to finish execution before moving on
pool.join()

# Save the results DataFrame to a CSV file for further analysis
df.to_csv('results_model_4_ECO_200_mixing_k_p0.9.csv')
