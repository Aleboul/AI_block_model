"""
Statistical Analysis and Clustering Functions

This module provides a collection of functions for statistical analysis, specifically focusing on 
extremal value theory and clustering based on extremal correlation matrices. The primary functions 
include the computation of the Empirical Cumulative Distribution Function (ECDF), the w-madogram, 
and a clustering algorithm that identifies clusters based on extremal correlations.

Functions:
----------
1. ecdf(X) : Computes the Empirical Cumulative Distribution Function for a given array of observations.
2. theta(R) : Calculates the w-madogram, a measure of dependence between variables in a rank matrix.
3. SECO(R, clst) : Evaluates the difference between the extremal coefficient of the entire rank matrix 
                   and the sum of extremal coefficients for each cluster of columns.
4. find_max(M, S) : Finds the maximum off-diagonal value in a submatrix defined by given indices.
5. clust(Theta, n, m, alpha=None) : Performs clustering based on an extremal correlation matrix and a threshold.
6. perc_exact_recovery(O_hat, O_bar) : Computes the percentage of exact recovery of true clusters by comparing 
                                         estimated clusters with true clusters.

Inputs and Outputs:
-------------------
- Each function takes specific inputs (e.g., NumPy arrays, dictionaries) and returns relevant outputs (e.g., floats, 
  dictionaries of clusters). Refer to each function's docstring for detailed information on inputs and outputs.

Dependencies:
-------------
- This module requires NumPy for numerical operations.

Usage Example:
--------------
Import the module and call the desired functions, passing the appropriate parameters as documented in the 
function definitions.

"""


import numpy as np


def ecdf(X):
    """
    Compute the Empirical Cumulative Distribution Function (ECDF) for a given array of observations.

    Inputs
    ------
    X (np.array[float]) : array of observations, a 1D NumPy array of floating point numbers.

    Output
    ------
    ecdf (np.array[float]) : The empirical cumulative distribution function values corresponding to 
                             each element in X, returned as a NumPy array of floats. The ECDF is 
                             uniform and represents the proportion of observations less than or 
                             equal to each observation.
    """

    # Sort the array to determine the order of observations
    index = np.argsort(X)  # Returns the indices that would sort the array X

    # Initialize an array of zeros to store the ECDF values
    ecdf = np.zeros(len(index))  # Same size as the input array X

    # Loop over each index of the sorted array
    for i in index:
        # Compute the proportion of observations that are less than or equal to the current observation
        # X <= X[i] creates a boolean array, np.sum counts the number of True values (i.e., counts observations <= X[i])
        # Divide by the total number of observations (X.shape[0]) to get the ECDF value for X[i]
        ecdf[i] = (1.0 / X.shape[0]) * np.sum(X <= X[i])

    return ecdf


def theta(R):
    """
    This function computes the extremal coefficient, a statistical tool used in extreme value theory to 
    measure dependence between variables.

    Inputs
    ------
    R (np.array[float] of shape (n_samples, d)) : Rank's matrix, where each row represents a sample 
                                                  and each column represents a variable (dimension d).

    Outputs
    -------
    madogram (float) : The computed madogram value, which is a measure of variability and dependence 
                         among the variables in the rank's matrix.
    """

    # Get the number of variables (Nnb) and the number of samples (Tnb) from the shape of R
    Nnb = R.shape[1]  # Number of variables (columns)
    Tnb = R.shape[0]  # Number of samples (rows)

    # Initialize an empty array V with the same number of rows as R and Nnb columns
    V = np.zeros([Tnb, Nnb])  # Shape: (n_samples, d), same as R

    # Loop over each variable (column) in the rank's matrix and copy it to V
    for j in range(0, Nnb):
        V[:, j] = R[:, j]  # Copy the j-th column of R to the j-th column of V

    # Calculate value_1, which is the maximum rank for each sample (across columns)
    value_1 = np.amax(V, axis=1)  # Max rank per row (across variables)

    # Calculate value_2, which is the mean rank for each sample (across columns)
    # Mean rank per row (across variables)
    value_2 = (1 / Nnb) * np.sum(V, axis=1)

    # Compute the madogram: average of the differences between the maximum and mean ranks
    mado = (1 / Tnb) * np.sum(value_1 - value_2)

    # Transform the madogram value into the extremal coefficient result using a specific formula
    value = (mado + 1/2) / (1/2 - mado)

    return value  # Return the extremal coefficient value


def SECO(R, clst):
    """
    This function evaluates a criterion based on the difference between the extremal coefficient value of the entire rank matrix 
    and the sum of the extremal coefficient values for each cluster of columns.

    Input
    -----
    R (np.array[float]) : n x d rank matrix, where rows represent samples and columns represent variables.
    clst (dict[int, list[int]]) : A dictionary representing the partition of columns. 
                                  The keys are cluster identifiers, and the values are lists of column indices that belong to each cluster.

    Output
    ------
    float : The difference between the extremal coefficient of the entire rank matrix (theta) and the sum of the extremal coefficients 
            for each cluster (theta_Σ).
    """

    # d represents the number of rows (samples) in the rank matrix R
    d = R.shape[0]

    # Step 1: Evaluate the w-madogram for the entire rank matrix R
    value = theta(R)  # Compute the w-madogram for the entire matrix R

    # Initialize a list to store w-madogram values for each cluster
    _value_ = []

    # Step 2: Loop through each cluster in the partition
    for key, c in clst.items():
        # Extract the submatrix _R_ corresponding to the columns in cluster 'c'
        # Subset of R that includes only the columns belonging to the cluster 'c'
        _R_ = R[:, c]

        # Compute the w-madogram for this submatrix and append it to the _value_ list
        _value_.append(theta(_R_))

    # Step 3: Return the difference between the sum of the w-madograms for clusters and the overall w-madogram
    # Sum of individual cluster extremal coefficients minus overall extremal coefficient
    return np.sum(_value_) - value


def find_max(M, S):
    """
    This function finds the maximum off-diagonal value in a submatrix of the given matrix M, defined by the indices in S.

    Inputs
    ------
    M (np.array[float]) : A square matrix from which the maximum value is to be found.
    S (list[int])       : A list of indices representing a subset of rows and columns in M, 
                          used to define the submatrix.

    Output
    ------
    (i, j) (tuple[int, int]) : The row and column indices (i, j) of the maximum value found in the submatrix of M, 
                               excluding the diagonal.
    """

    # Step 1: Create a boolean mask of the same shape as M, initialized with False
    mask = np.zeros(M.shape, dtype=bool)

    # Step 2: Create a submatrix of True values with dimensions len(S) x len(S)
    values = np.ones((len(S), len(S)), dtype=bool)

    # Step 3: Use the mask to select only the rows and columns of M defined by S
    # np.ix_(S,S) creates an index array for selecting the rows and columns in S
    mask[np.ix_(S, S)] = values

    # Step 4: Set the diagonal of the mask to False to exclude the diagonal elements
    np.fill_diagonal(mask, 0)  # We don't want to consider diagonal elements

    # Step 5: Extract the elements of M where the mask is True, then find the maximum value
    # Find the maximum value in the masked submatrix (off-diagonal)
    max_value = M[mask].max()

    # Step 6: Find the indices (i, j) where the value in the masked M equals the maximum value
    # np.multiply(M, mask * 1) zeroes out all elements of M not included in the mask
    # Sometimes there are duplicates for low values of n, hence only return the first occurrence
    i, j = np.where(np.multiply(M, mask * 1) == max_value)

    # Step 7: Return the first occurrence of the maximum value's indices (i, j)
    return i[0], j[0]


def clust(Theta, n, m=None, alpha=None):
    """
    Performs clustering based on an AI-block model using an extremal correlation matrix and a threshold alpha.

    Inputs
    ------
    Theta (np.array[float]) : Extremal correlation matrix of shape (d, d), where d is the number of variables.
    n (int)                 : The number of block maxima.
    m (int)                 : A parameter related to the block length.
    alpha (float, optional) : Threshold for clustering. If not provided, it is calculated as 2 * (1/m + sqrt(ln(d)/n)).

    Outputs
    -------
    dict[int, np.array[int]] : A dictionary where the keys are cluster identifiers (l), and the values are arrays of 
                               column indices representing the clusters. Each cluster is a partition of the set {1, ..., d}.
    """

    # d represents the number of columns (variables) in the extremal correlation matrix Theta
    d = Theta.shape[1]

    # Step 1: Initialization

    # S is the set of all column indices {0, 1, ..., d-1} to be clustered
    S = np.arange(d)

    # l is a counter for the cluster index
    l = 0

    # If alpha is not provided, calculate it using the formula 2 * (1/m + sqrt(ln(d) / n))
    if alpha is None:
        alpha = 2 * (1/m + np.sqrt(np.log(d) / n))

    # Initialize an empty dictionary to store the clusters
    cluster = {}

    # Step 2: Iterative clustering process
    # Continue clustering until all elements in S are assigned to a cluster
    while len(S) > 0:
        l = l + 1  # Increment the cluster index

        # If only one element remains in S, assign it to a new cluster
        if len(S) == 1:
            # Assign the last remaining element to cluster l
            cluster[l] = np.array(S)

        else:
            # Find the pair (a_l, b_l) with the maximum extremal correlation in the submatrix of Theta corresponding to S
            a_l, b_l = find_max(Theta, S)

            # If the maximum correlation is less than the threshold alpha, assign a_l to a singleton cluster
            if Theta[a_l, b_l] < alpha:
                # Create a singleton cluster with a_l
                cluster[l] = np.array([a_l])

            else:
                # Find all indices in S that have extremal correlation >= alpha with both a_l and b_l
                # Indices where correlation with a_l is >= alpha
                index_a = np.where(Theta[a_l, :] >= alpha)
                # Indices where correlation with b_l is >= alpha
                index_b = np.where(Theta[b_l, :] >= alpha)

                # Create the cluster as the intersection of indices with high correlation to both a_l and b_l
                cluster[l] = np.intersect1d(S, index_a, index_b)

        # Remove the newly formed cluster from the set S to avoid re-clustering those elements
        S = np.setdiff1d(S, cluster[l])

    # Step 3: Return the partition of clusters
    return cluster


def perc_exact_recovery(O_hat, O_bar):
    """
    Computes the percentage of exact recovery of true clusters by comparing the estimated clusters with the true clusters.

    Inputs
    ------
    O_hat (dict[int, np.array[int]]) : A dictionary representing the estimated clusters. 
                                       The keys are cluster labels, and the values are arrays of indices in each estimated cluster.
    O_bar (dict[int, np.array[int]]) : A dictionary representing the true clusters. 
                                       The keys are cluster labels, and the values are arrays of indices in each true cluster.

    Outputs
    -------
    float : The proportion of true clusters that are exactly recovered by the estimated clusters. 
            A value of 1.0 means all true clusters are perfectly recovered, while 0.0 means none are.
    """

    # Initialize a counter for the number of exactly recovered clusters
    value = 0

    # Step 1: Loop through each true cluster in O_bar
    for key1, true_clust in O_bar.items():

        # Step 2: For each true cluster, loop through the estimated clusters in O_hat
        for key2, est_clust in O_hat.items():

            # Step 3: Check if the true cluster and the estimated cluster have the same size
            if len(true_clust) == len(est_clust):

                # Find the common elements (intersection) between the true and estimated cluster
                test = np.intersect1d(true_clust, est_clust)

                # Step 4: Check if the intersection is non-empty, matches in size with the true cluster,
                # and that all elements in both clusters are the same (when sorted)
                if len(test) > 0 and len(test) == len(true_clust) and np.sum(np.sort(test) - np.sort(true_clust)) == 0:

                    # Increment the exact recovery counter if the clusters are perfectly matched
                    value += 1

    # Step 5: Return the proportion of exactly recovered clusters (value / total number of true clusters)
    return value / len(O_bar)
