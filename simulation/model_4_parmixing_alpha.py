"""
Monte Carlo Simulation for Clustering Evaluation Using Extremal Correlation for Experiment 3.

This script performs a Monte Carlo simulation to evaluate clustering accuracy and the 
SECO criterion based on synthetic samples generated using copula-based methods. The 
samples are generated through R functions sourced from 'model_4_parmixing_m.R' using 
the rpy2 library. The simulation runs multiple iterations in parallel to analyze the 
performance of different clustering thresholds (_alpha_) and outputs the results to CSV files.

Dependencies:
--------------
- rpy2: Interface to R from Python, used for calling R functions.
- numpy: Library for numerical operations and handling arrays.
- matplotlib: Library for plotting (not used in the provided code).
- pandas: Library for data manipulation and saving results to CSV.
- multiprocessing: Module to enable parallel processing.
- eco_alg: Custom utility functions used for computing empirical CDFs and clustering.

Functions:
----------
- init_pool_processes(): Initializes random seed for each process in the pool.
- make_sample(n_sample, d, k, m, p, seed, K): 
    Generates synthetic samples and computes the ground-truth clusters.
    Inputs:
        - n_sample (int): Number of samples to generate.
        - d (int): Total number of dimensions in the dataset.
        - k (int): Parameter affecting block maxima.
        - m (int): Block length parameter.
        - p (float): Probability parameter for randomness.
        - seed (int): Seed for reproducibility.
        - K (int): Number of clusters.
    Outputs:
        - np.array: Generated sample data.
        - O_bar (dict): Ground-truth partitioning of dimensions.

- operation_model_4_ECO(dict, seed, _alpha_):
    Executes the clustering operation for a given set of parameters and evaluates the clustering results.
    Inputs:
        - dict (dict): Dictionary of model parameters.
        - seed (int): Random seed for reproducibility.
        - _alpha_ (list[float]): List of alpha thresholds for clustering.
    Outputs:
        - list: Contains recovery percentages and SECO values.

Parameters:
-----------
- d (int): Total number of dimensions (variables) in the sample.
- k (int): Parameter related to the number of observations (block maxima size).
- m (int): Block length parameter used in sampling and clustering.
- p (float): Probability parameter for sample generation.
- K (int): Number of clusters or copula components in the model.
- n_iter (int): Number of Monte Carlo iterations to run.

Output Files:
-------------
- perc_model_4_ECO_200_p0.9.csv: Contains the exact recovery percentages for each alpha.
- seco_model_4_ECO_200_p0.9.csv: Contains the SECO criterion values for each alpha.

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


def operation_model_4_ECO(dict, seed, _alpha_):
    """
    Performs a Monte Carlo simulation to evaluate clustering accuracy and SECO criterion based on generated samples.

    Inputs
    ------
    dict (dict) : A dictionary containing the following keys:
                  - 'd1' (int) : Dimension of the first sample group.
                  - 'd2' (int) : Dimension of the second sample group.
                  - 'n_sample' (int) : Length of the sample (number of observations).
                  - 'd' (int) : Total number of dimensions (d1 + d2).
                  - 'k' (int) : Parameter related to the block maxima.
                  - 'm' (int) : Block length parameter.
                  - 'p' (float) : Probability parameter for sample generation.
                  - 'K' (int) : Number of clusters or copula components in the model.
    seed (int)  : Seed for random number generation, ensuring reproducibility.
    _alpha_ (list[float]) : List of threshold values for clustering.

    Outputs
    -------
    list : A list of lists where each sublist contains:
           - perc (float) : The percentage of exact cluster recovery.
           - seco (float) : The SECO criterion value for the clustering result.
    """

    # Set random seed for reproducibility
    np.random.seed(1 * seed)

    # Step 2: Generate the sample and the ground-truth cluster structure (O_bar)
    sample, O_bar = make_sample(n_sample=dict['n_sample'], d=dict['d'], k=dict['k'],
                                m=dict['m'], p=dict['p'], seed=seed, K=dict['K'])

    # Initialization: Get total number of dimensions
    d = sample.shape[1]

    # Initialize the ECDF matrix R (observations x dimensions)
    R = np.zeros([sample.shape[0], d])

    # Step 3: Compute the empirical cumulative distribution function (ECDF) for each variable (dimension)
    for j in range(0, d):
        X_vec = sample[:, j]  # Extract the j-th variable (column)
        R[:, j] = eco_alg.ecdf(X_vec)  # Compute the ECDF for the j-th variable

    # Step 4: Initialize the extremal correlation matrix Theta
    Theta = np.ones([d, d])

    # Step 5: Calculate the extremal correlation between each pair of variables
    for j in range(0, d):
        for i in range(0, j):
            # Compute the extremal correlation using the 'theta' function from 'eco_alg'
            Theta[i, j] = Theta[j, i] = 2 - eco_alg.theta(R[:, [i, j]])

    # Step 6: Initialize an empty list to store results
    output = []

    # Step 7: Perform clustering and evaluate for each value of alpha in _alpha_
    for alpha in _alpha_:
        # Perform clustering using the extremal correlation matrix Theta and threshold alpha
        O_hat = eco_alg.clust(
            Theta, n=dict['n_sample'], m=dict['m'], alpha=alpha)

        # Step 8: Calculate the percentage of exact cluster recovery
        perc = eco_alg.perc_exact_recovery(O_hat, O_bar)

        # Step 9: Calculate the SECO criterion for the clustering result
        seco = eco_alg.SECO(R, O_hat)

        # Step 10: Append the results (perc, seco) to the output list
        output.append([perc, seco])

    # Step 11: Return the final output list containing the results for all alpha values
    return output


# Set model parameters
d = 200  # The total number of dimensions (variables) in the sample
# Parameter related to the number of observations (block maxima number)
k = 800
m = 20   # Block length parameter used in sampling
p = 0.9  # Probability parameter for process generation
K = 5    # Number of clusters or copula components for the model
n_iter = 10  # Number of Monte Carlo iterations to run

# Define the range of alpha values to use for clustering thresholds.
# The alpha values are scaled based on the model parameters (m, d, k)
_alpha_ = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75,
                    3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]) * (1/m + np.sqrt(np.log(d) / k))

# Set up a multiprocessing pool with 10 parallel processes for running simulations.
# The initializer sets up the environment for each process.
pool = mp.Pool(processes=10, initializer=init_pool_processes)

# Define the mode of operation as "ECO", which controls the simulation behavior.
mode = "ECO"

# Initialize an empty list to store results
stockage = []

# Calculate the total number of samples (observations) for each simulation as the product of k and m
n_sample = k * m

# Create a dictionary with input parameters to pass to the simulation function.
# This includes dimensions, sample size, block maxima parameters, and number of clusters (K).
input = {'d': d, 'n_sample': n_sample, 'k': k, 'm': m, 'p': p, 'K': K}

# Check if the mode is set to "ECO" and, if so, run the simulation.
if mode == "ECO":
    # Use the multiprocessing pool to run the `operation_model_4_ECO` function in parallel.
    # For each of the `n_iter` iterations, pass the input parameters, a unique seed (i), and alpha thresholds.
    result_objects = [pool.apply_async(operation_model_4_ECO, args=(
        input, i, _alpha_)) for i in range(n_iter)]

# Gather results from all parallel processes once they complete.
# Each `result_objects[i].get()` retrieves the output from the corresponding process.
results = np.array([r.get() for r in result_objects])

# Print the complete results for verification or debugging purposes.
print(results)

# Initialize empty lists to store SECO values and exact recovery percentages from the results.
seco = []
perc = []

# Loop through each result and extract the SECO and exact recovery values.
for r in results:
    # The second column (index 1) contains the SECO criterion values.
    seco.append(r[:, 1])
    # The first column (index 0) contains the exact recovery percentages.
    perc.append(r[:, 0])

# Convert the SECO and percentage lists into pandas DataFrames for easy manipulation.
seco = pd.DataFrame(seco)
perc = pd.DataFrame(perc)

# Print the SECO and exact recovery DataFrames for inspection.
print(seco)
print(perc)

# Clean up the multiprocessing pool to prevent any further task submissions.
pool.close()

# Wait for all processes in the pool to finish execution.
pool.join()

# Save the exact recovery percentage results to a CSV file.
perc.to_csv('perc_model_4_ECO_200_p0.9.csv')

# Save the SECO criterion results to a CSV file.
seco.to_csv('seco_model_4_ECO_200_p0.9.csv')
