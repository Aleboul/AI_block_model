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


# Set the parameters for the simulation:
d1 = 100  # Dimension of the first sample group
d2 = 100  # Dimension of the second sample group
# Different sample sizes (block maxima)
k_sample = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
m = 20  # Block length parameter
n_iter = 100  # Number of Monte Carlo iterations to run
p = 0.9  # Probability parameter for the random process
mode = "ECO"  # Mode of operation to control the simulation behavior

# Initialize a multiprocessing pool with 10 parallel processes
pool = mp.Pool(processes=10, initializer=init_pool_processes)

# Initialize an empty list to store all the results
stockage = []

# Loop over each value of 'k' in 'k_sample' to run the simulation for different sample sizes
for k in k_sample:

    # Calculate the total number of samples (observations) for this value of 'k'
    n = k * m

    # Create a dictionary with the input parameters to pass to the simulation function
    input = {'n_sample': n, 'd1': d1, 'd2': d2, 'k': k, 'm': m, 'p': p}

    # If the mode is set to "ECO", run the simulation using the ECO model
    if mode == "ECO":

        # Use the multiprocessing pool to apply the 'operation_model_3_ECO' function in parallel.
        # For each of the 'n_iter' iterations, pass the input parameters and a unique seed (i).
        result_objects = [pool.apply_async(
            operation_model_3_ECO, args=(input, i)) for i in range(n_iter)]

    # Gather results from all parallel processes once they complete
    results = [r.get() for r in result_objects]

    # Append the results for this value of 'k' to the 'stockage' list
    stockage.append(results)

    # Convert the list of results into a pandas DataFrame for easy manipulation and inspection
    df = pd.DataFrame(stockage)

    # Print the DataFrame to check the results for each iteration
    print(df)

# Clean up the multiprocessing pool to prevent any further task submissions
pool.close()

# Wait for all processes in the pool to finish execution
pool.join()

# Print the final DataFrame containing results for all values of 'k'
print(df)

# Save the results DataFrame to a CSV file for further analysis
df.to_csv('results_model_3_ECO_1600_mixing_k_p0.9.csv')
