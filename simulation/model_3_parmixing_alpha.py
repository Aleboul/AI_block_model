"""
Monte Carlo Simulation for Clustering Evaluation Using Extremal Correlation.

This script uses R and Python to generate synthetic data based on copula models and 
performs clustering evaluation through Monte Carlo simulations. It computes the 
percentage of exact cluster recovery and SECO (Sum of Extremal COefficients) criterion.

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

- operation_model_3_ECO(dict, seed, _alpha_):
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
r['source']('model_3_parmixing_m.R')


def init_pool_processes():
    np.random.seed()


def make_sample(n_sample, d, k, m, p, seed):
    """
    Generates a sample using R functions for copula-based sampling and random number generation.

    Inputs
    ------
    n_sample (int) : The number of samples to generate.
    d (int)        : The number of dimensions or variables in the data.
    k (int)        : A parameter related to the number of block maxima.
    m (int)        : A parameter related to the length of block maxima.
    p (float)      : A probability parameter, related to the random process.
    seed (int)     : The random seed for reproducibility.

    Outputs
    -------
    np.array       : A NumPy array containing the generated sample data.
    """

    # Step 1: Retrieve the necessary R functions from the R environment
    # Function to generate random data
    generate_randomness = robjects.globalenv['generate_randomness']
    # Function to process the generated data
    robservation = robjects.globalenv['robservation']
    # Function to generate a copula model
    generate_copula = robjects.globalenv['generate_copula']

    # Step 2: Generate a copula model (possibly for dependency structure)
    copoc = generate_copula()  # Generates a copula object

    # Step 3: Generate random samples using the copula model
    # - n_sample: Number of samples
    # - p: Probability parameter
    # - d: Number of variables
    # - copoc: The copula model object
    # - seed: Random seed for reproducibility
    sample = generate_randomness(n_sample, p=p, d=d, copoc=copoc, seed=seed)

    # Step 4: Apply further processing or observation logic using the 'robservation' function
    # - sample: The generated random sample
    # - m: A parameter affecting the observation process
    # - d: Number of variables
    # - k: Another parameter affecting the sample generation process
    data = robservation(sample, m=m, d=d, k=k)

    # Step 5: Convert the resulting R data structure to a NumPy array and return it
    return np.array(data)


def operation_model_3_ECO(dict, seed, _alpha_):
    """
    Performs a Monte Carlo simulation to evaluate clustering accuracy and SECO criterion based on generated samples.

    Inputs
    ------
    dict (dict) : A dictionary containing the following keys:
                  - 'd1' (int) : Dimension of the first sample group.
                  - 'd2' (int) : Dimension of the second sample group.
                  - 'n_sample' (int) : Length of the sample (number of observations).
                  - 'k' (int) : Parameter used in sample generation (affecting structure).
                  - 'm' (int) : Parameter used in sample generation (affecting clustering).
                  - 'p' (float) : Probability parameter for sample generation.
    seed (int)  : Seed for random number generation, ensuring reproducibility.
    _alpha_ (list[float]) : List of threshold values for clustering.

    Outputs
    -------
    list : A list of lists where each sublist contains:
           - perc (float) : The percentage of exact cluster recovery.
           - seco (float) : The SECO criterion value for the clustering result.
    """

    # Set the random seed for reproducibility
    np.random.seed(1 * seed)

    # Step 1: Generate a sample using the 'make_sample' function
    # The sample is generated based on the combined dimensions d1 + d2 and other parameters from the 'dict'
    sample = make_sample(n_sample=dict['n_sample'], d=dict['d1'] +
                         dict['d2'], k=dict['k'], m=dict['m'], p=dict['p'], seed=seed)

    # Step 2: Initialize variables
    d = sample.shape[1]  # Total number of variables (columns) in the sample

    # Initialize an empty rank matrix 'R' to store ECDF values
    R = np.zeros([sample.shape[0], d])

    # Step 3: Calculate the empirical cumulative distribution function (ECDF) for each variable
    for j in range(0, d):
        # Extract the j-th variable (column) from the sample
        X_vec = sample[:, j]
        R[:, j] = eco_alg.ecdf(X_vec)  # Compute the ECDF for the j-th variable

    # Step 4: Initialize the Theta matrix for extremal correlation values
    Theta = np.ones([d, d])

    # Step 5: Calculate the extremal correlation between each pair of variables
    for j in range(0, d):
        for i in range(0, j):
            # Compute the extremal correlation using the 'theta' function from 'eco_alg'
            Theta[i, j] = Theta[j, i] = 2 - eco_alg.theta(R[:, [i, j]])

    # Step 6: Initialize an empty list to store the results
    output = []

    # Step 7: Perform clustering and evaluate for each value of alpha in _alpha_
    for alpha in _alpha_:
        # Perform clustering using the extremal correlation matrix Theta and threshold alpha
        O_hat = eco_alg.clust(
            Theta, n=dict['n_sample'], m=dict['m'], alpha=alpha)

        # Define the ground truth clusters (O_bar) as two clusters: [0, d1) and [d1, d)
        O_bar = {1: np.arange(0, dict['d1']), 2: np.arange(dict['d1'], d)}

        # Step 8: Calculate the percentage of exact recovery using the 'perc_exact_recovery' function
        perc = eco_alg.perc_exact_recovery(O_hat, O_bar)

        # Step 9: Calculate the SECO criterion using the 'SECO' function
        seco = eco_alg.SECO(R, O_hat)

        # Step 10: Append the results (perc, seco) to the output list
        output.append([perc, seco])

        # Print the output after each iteration (for debugging or tracking progress)
        print(output)

    # Step 11: Return the final output list containing the results for all alpha values
    return output


# Define model parameters
d1 = 100  # Dimension of the first cluster
d2 = 100  # Dimension of the second cluster
d = d1 + d2  # Total dimension, combining both sample groups
# Parameter related to the number of observations (sample size scaling factor)
k = 800
m = 20  # Block length parameter
p = 0.9  # Probability parameter for sample generation
n_iter = 100  # Number of Monte Carlo iterations
_alpha_ = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0,
                    3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]) * (1/m + np.sqrt(np.log(d)/k))
# _alpha_ is a range of thresholds used for clustering, adjusted based on the model parameters (m, d, k)

# Set up multiprocessing pool for parallel computation
# 10 processes for parallel simulation
pool = mp.Pool(processes=10, initializer=init_pool_processes)
# Operating mode for the simulation (likely controls the type of operation)
mode = "ECO"
stockage = []  # Initialize an empty list to store results

n_sample = k * m  # Calculate the total number of samples (sample length)

# Create a dictionary to pass input parameters to the function 'operation_model_3_ECO'
input = {'d1': d1, 'd2': d2, 'n_sample': n_sample, 'k': k, 'm': m, 'p': p}

# Check if the mode is set to "ECO" (which controls the type of operation being performed)
if mode == "ECO":

    # Step 1: Apply the 'operation_model_3_ECO' function in parallel for 'n_iter' iterations
    # Each worker in the pool runs a Monte Carlo iteration with the specified input and alpha thresholds
    result_objects = [pool.apply_async(operation_model_3_ECO, args=(
        input, i, _alpha_)) for i in range(n_iter)]

# Step 2: Gather the results from all parallel processes
# Collect results once each process completes
results = np.array([r.get() for r in result_objects])

# Print the results for verification or debugging purposes
print(results)

# Initialize empty lists to store SECO and exact recovery percentage values for each iteration
seco = []
perc = []

# Step 3: Extract SECO and exact recovery values from the results
for r in results:
    seco.append(r[:, 1])  # Extract SECO criterion (2nd column of each result)
    # Extract percentage of exact recovery (1st column of each result)
    perc.append(r[:, 0])

# Step 4: Convert the lists into pandas DataFrames for easy manipulation and saving
seco = pd.DataFrame(seco)
perc = pd.DataFrame(perc)

# Print SECO and exact recovery DataFrames
print(seco)
print(perc)

# Step 5: Clean up the multiprocessing pool (close and join)
pool.close()  # Close the pool to prevent new tasks from being submitted
pool.join()  # Wait for all the worker processes to finish

# Step 6: Save the SECO and exact recovery results to CSV files
# Save the exact recovery results to CSV
perc.to_csv('perc_model_3_ECO_200_p0.9.csv')
seco.to_csv('seco_model_3_ECO_200_p0.9.csv')  # Save the SECO results to CSV
