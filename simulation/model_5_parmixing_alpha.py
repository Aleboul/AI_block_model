"""
Monte Carlo Simulation for Clustering Evaluation using Copula-Based Sampling

This script performs a Monte Carlo simulation to evaluate the clustering accuracy 
and SECO (Statistical Error Correction Operation) criterion based on synthetic 
samples generated using copula methods. The simulation leverages R functions 
imported through the rpy2 library, and it utilizes multiprocessing to efficiently 
run multiple iterations of the clustering model.

Libraries Used:
---------------
- rpy2.robjects: For interfacing with R and executing R functions for copula 
  generation and sampling.
- numpy: For numerical operations and handling arrays.
- matplotlib: For plotting results (imported but not used in the current version).
- pandas: For data manipulation and saving results to CSV files.
- multiprocessing: For parallel processing to speed up Monte Carlo iterations.
- eco_alg: A custom utility module containing functions for empirical CDF calculation, 
  extremal correlation computation, and clustering.

Functions:
----------
1. init_pool_processes(): Initializes random seed for each process in the pool.
2. make_sample(n_sample, d, k, m, p, seed, K): Generates a synthetic sample 
   and computes ground-truth clusters.
3. operation_model_5_ECO(dict, seed, _alpha_): Runs the Monte Carlo simulation 
   and evaluates clustering performance based on generated samples.

Main Parameters:
----------------
- d: Number of dimensions in the sample.
- k: Number of observations (block maxima number).
- m: Block length parameter.
- p: Probability parameter for the random process.
- K: Number of clusters in the model.
- n_iter: Number of Monte Carlo iterations.

Output:
-------
The script outputs the exact recovery percentage and SECO criterion results, 
which are saved to CSV files for further analysis.

Usage:
------
1. Set the desired parameters.
2. Run the script to execute the simulation.
3. The results will be printed and saved as CSV files.
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
    generate_copula = robjects.globalenv['generate_copula']  # Function to generate a copula model
    generate_randomness = robjects.globalenv['generate_randomness']  # Function to generate random samples
    robservation = robjects.globalenv['robservation']  # Function to process generated samples

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
    return np.array(data), O_bar  # Convert data to NumPy array before returning

def operation_model_5_ECO(dict, seed, _alpha_):
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
    
    # Step 1: Set the random seed for reproducibility, scaled by 1
    np.random.seed(1 * seed)

    # Step 2: Generate a synthetic sample and ground-truth partitioning using the make_sample function
    sample, O_bar = make_sample(n_sample=dict['n_sample'], d=dict['d'], 
                                k=dict['k'], m=dict['m'], p=dict['p'], 
                                seed=seed, K=dict['K'])
    
    # Step 3: Initialize variables
    d = sample.shape[1]  # Get the number of dimensions in the generated sample

    # Step 4: Create an empirical cumulative distribution function (ECDF) matrix R
    R = np.zeros([sample.shape[0], d])  # Initialize the matrix to hold ECDF values
    for j in range(0, d):  # Iterate over each dimension
        X_vec = sample[:, j]  # Extract the j-th variable (column)
        R[:, j] = eco_alg.ecdf(X_vec)  # Compute the ECDF for the j-th variable and store it in R
    
    # Step 5: Initialize the extremal correlation matrix Theta
    Theta = np.ones([d, d])  # Create a matrix to store extremal correlations

    # Step 6: Calculate the extremal correlation between each pair of variables
    for j in range(0, d):  # Iterate over each dimension
        for i in range(0, j):  # Only iterate over the lower triangle to avoid redundancy
            # Compute the extremal correlation using the 'theta' function from 'eco_alg'
            Theta[i, j] = Theta[j, i] = 2 - eco_alg.theta(R[:, [i, j]])

    # Step 7: Initialize an empty list to store results
    output = []

    # Step 8: Perform clustering and evaluate for each value of alpha in _alpha_
    for alpha in _alpha_:
        # Perform clustering using the extremal correlation matrix Theta and threshold alpha
        O_hat = eco_alg.clust(Theta, n=dict['n_sample'], m=dict['m'], alpha=alpha)

        # Step 9: Calculate the percentage of exact cluster recovery
        perc = eco_alg.perc_exact_recovery(O_hat, O_bar)

        # Step 10: Calculate the SECO criterion for the clustering result
        seco = eco_alg.SECO(R, O_hat)

        # Step 11: Append the results (perc, seco) to the output list
        output.append([perc, seco])

    # Step 12: Return the final output list containing the results for all alpha values
    return output

# Set model parameters for the simulation
d = 200  # The total number of dimensions (variables) in the sample
k = 800  # Parameter related to the number of observations (block maxima number)
m = 20   # Block length parameter used in sampling
p = 0.9  # Probability parameter for the random process
K = 10   # Number of clusters or copula components for the model
n_iter = 10  # Number of Monte Carlo iterations to run

# Define the range of alpha values to use for clustering thresholds.
# The alpha values are scaled based on the model parameters (m, d, k).
_alpha_ = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 
                    2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 
                    4.0, 4.25, 4.5, 4.75, 5.0]) * (1/m + np.sqrt(np.log(d) / k))

# Initialize a multiprocessing pool with 10 parallel processes for running simulations.
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
    # Use the multiprocessing pool to run the `operation_model_5_ECO` function in parallel.
    # For each of the `n_iter` iterations, pass the input parameters, a unique seed (i), and alpha thresholds.
    result_objects = [pool.apply_async(operation_model_5_ECO, args=(input, i, _alpha_)) for i in range(n_iter)]

# Gather results from all parallel processes once they complete.
# Each `result_objects[i].get()` retrieves the output from the corresponding process.
results = np.array([r.get() for r in result_objects])

# Print the complete results for verification or debugging purposes.
print(results)

# Initialize empty lists to store SECO values and exact recovery percentages from the results.
seco = []  # List to hold SECO criterion values
perc = []  # List to hold exact recovery percentages

# Loop through each result and extract the SECO and exact recovery values.
for r in results:
    seco.append(r[:, 1])  # The second column (index 1) contains the SECO criterion values.
    perc.append(r[:, 0])  # The first column (index 0) contains the exact recovery percentages.

# Convert the SECO and percentage lists into pandas DataFrames for easy manipulation.
seco = pd.DataFrame(seco)  # Create a DataFrame for SECO values
perc = pd.DataFrame(perc)  # Create a DataFrame for percentage values

# Print the SECO and exact recovery DataFrames for inspection.
print(seco)
print(perc)

# Clean up the multiprocessing pool to prevent any further task submissions.
pool.close()

# Wait for all processes in the pool to finish execution.
pool.join()

# Save the exact recovery percentage results to a CSV file for later analysis.
perc.to_csv('perc_model_5_ECO_200_p0.9.csv')

# Save the SECO criterion results to a CSV file for later analysis.
seco.to_csv('seco_model_5_ECO_200_p0.9.csv')