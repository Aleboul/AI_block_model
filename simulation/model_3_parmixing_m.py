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
  saved to 'results_model_3_ECO_200_mixing_p0.9_m.csv'.
"""


import rpy2.robjects as robjects
r = robjects.r
r['source']('model_3_parmixing_m.R')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing as mp
import eco_alg

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

def operation_model_3_ECO(dict, seed):
    """
    Performs a Monte Carlo simulation to evaluate clustering accuracy using the ECO algorithm.

    Parameters
    ----------
    dict : dict
        A dictionary containing the following keys:
        - 'd1' (int) : Dimension of the first sample group (number of variables in group 1).
        - 'd2' (int) : Dimension of the second sample group (number of variables in group 2).
        - 'n_sample' (int) : The total number of samples (observations).
        - 'k' (int) : Parameter related to the number of block maxima.
        - 'm' (int) : Block length parameter.
        - 'p' (float) : Probability parameter for generating samples.
    seed : int
        Seed for random number generation, ensuring reproducibility.

    Returns
    -------
    perc : float
        The percentage of exact recovery of clusters (how well the clustering algorithm recovers the true partition).
    """

    # Step 1: Set the random seed for reproducibility, scaled by 1
    np.random.seed(1 * seed)

    # Step 2: Generate a synthetic sample by calling `make_sample`.
    # The sample will have dimensions 'd1 + d2' (combined size of both groups) and length 'n_sample'.
    sample = make_sample(n_sample=dict['n_sample'], d=dict['d1'] + dict['d2'],
                         k=dict['k'], m=dict['m'], p=dict['p'], seed=seed)

    # Step 3: Initialize variables
    # Get the number of dimensions (variables) in the generated sample
    d = sample.shape[1]

    # Step 4: Create an empirical cumulative distribution function (ECDF) matrix `R`
    # Initialize a matrix to hold ECDF values, with 'k' rows and 'd' columns
    R = np.zeros([dict['k'], d])
    for j in range(0, d):  # Iterate over each dimension (variable)
        # Extract the j-th variable (column) from the sample
        X_vec = sample[:, j]
        # Compute the ECDF for the j-th variable and store it in `R`
        R[:, j] = eco_alg.ecdf(X_vec)

    # Step 5: Initialize the extremal correlation matrix `Theta`
    # Create a matrix to store extremal correlations (initialized to 1)
    Theta = np.ones([d, d])

    # Step 6: Calculate the extremal correlation between each pair of variables
    for j in range(0, d):  # Iterate over each dimension (variable)
        for i in range(0, j):  # Only iterate over the lower triangle to avoid redundant calculations
            # Compute the extremal correlation between variables i and j using the `theta` function
            Theta[i, j] = Theta[j, i] = 2 - eco_alg.theta(R[:, [i, j]])

    # Step 7: Perform clustering based on the extremal correlation matrix `Theta`
    # The `clust` function clusters the variables based on the extremal correlations, using parameters 'k' and 'm'
    O_hat = eco_alg.clust(Theta, n=dict['k'], m=dict['m'])

    # Step 8: Define the ground-truth partitioning of dimensions (`O_bar`).
    # Group 1 corresponds to the first 'd1' dimensions, and Group 2 corresponds to the remaining 'd2' dimensions.
    O_bar = {1: np.arange(0, dict['d1']), 2: np.arange(dict['d1'], d)}

    # Step 9: Calculate the percentage of exact recovery (how accurately the clustering matches the true partition)
    perc = eco_alg.perc_exact_recovery(O_hat, O_bar)

    # Step 10: Return the exact recovery percentage
    return perc

# Define the dimensions for the first and second samples
d1 = 100  # Number of dimensions for the first sample
d2 = 100  # Number of dimensions for the second sample

# Define a list of sample sizes (m values) to evaluate
m_sample = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]

# Set the total number of observations and number of iterations for the simulation
n = 10000  # Total number of observations in each simulation
n_iter = 10  # Number of iterations for each m value
p = 0.5  # Probability parameter influencing the random process

# Initialize a multiprocessing pool with 10 processes to run simulations in parallel
pool = mp.Pool(processes=10, initializer=init_pool_processes)

# Set the mode for the simulation, controlling which model to use (in this case, ECO)
mode = "ECO"

# Initialize an empty list to store results for different m values
stockage = []

# Loop through each value of 'm' in 'm_sample', representing different block lengths
for m in m_sample:

    # Calculate the number of clusters (k) based on the current sample size (m)
    k = int(np.floor(n / m))  # Number of clusters is the total number of observations divided by block length

    # Define the input parameters for the simulation, bundled into a dictionary
    input = {'n_sample': k * m, 'd1': d1, 'd2': d2, 'k': k, 'm': m, 'p': p}

    # Check if the mode is "ECO", which controls the behavior of the simulation
    if mode == "ECO":
        # Run the 'operation_model_3_ECO' function in parallel using the multiprocessing pool
        # This will run the simulation 'n_iter' times for the current value of 'm', each time with a different seed (i)
        result_objects = [pool.apply_async(operation_model_3_ECO, args=(input, i)) for i in range(n_iter)]

    # Gather the results from all the parallel processes using the 'get' method
    results = [r.get() for r in result_objects]

    # Append the results for this value of 'm' to the 'stockage' list
    stockage.append(results)

    # Convert the results (so far) into a pandas DataFrame for easier analysis and inspection
    df = pd.DataFrame(stockage)

    # Print the current DataFrame to monitor progress
    print(df)

# Step 9: Once all the simulations are done, close the multiprocessing pool to prevent further task submissions
pool.close()

# Step 10: Wait for all processes in the pool to complete their execution
pool.join()

# Print the final DataFrame containing results for all m values
print(df)

# Save the final DataFrame to a CSV file for future reference and analysis
df.to_csv('results_model_3_ECO_200_mixing_p0.9_m.csv')
