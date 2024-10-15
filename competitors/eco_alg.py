"""
Clustering and Recovery Analysis

This module contains functions to perform clustering based on an extremal correlation matrix using
the AI-block model. It also includes functionality to evaluate the percentage of exact recovery of
clusters between estimated and true cluster assignments.

Functions:
----------
1. find_max(M, S)
    - Find the maximum value in a matrix M, excluding specific rows and columns indicated by S.

    Inputs:
        M : np.ndarray
            A 2D numpy array (matrix) from which to find the maximum value.
        S : list or np.ndarray
            A list or array of indices representing the rows and columns to exclude from consideration.

    Outputs:
        tuple
            A tuple containing the indices (i, j) of the maximum value found in M,
            where both i and j are indices of the original matrix M.

2. clust(Theta, alpha)
    - Performs clustering in the AI-block model based on an extremal correlation matrix.

    Inputs:
        Theta : np.ndarray
            An extremal correlation matrix of shape (d, d) where d is the number of variables.
        alpha : float
            A threshold value, typically of order sqrt(ln(d)/n), used to determine cluster membership.

    Outputs:
        dict
            A dictionary where each key represents a cluster label, and the associated value is a 
            numpy array containing the indices of the elements in that cluster.

3. perc_exact_recovery(O_hat, O_bar)
    - Calculate the percentage of exact recovery of clusters.

    Inputs:
        O_hat : dict
            A dictionary where keys are estimated cluster labels and values are arrays of indices 
            representing the estimated clusters.
        O_bar : dict
            A dictionary where keys are true cluster labels and values are arrays of indices 
            representing the true clusters.

    Outputs:
        float
            The percentage of true clusters that were exactly recovered in the estimated clusters.
"""


import numpy as np


def find_max(M, S):
    '''
    Find the maximum value in a matrix M, excluding specific rows and columns indicated by S.

    Inputs
    ------
    M : np.ndarray
        A 2D numpy array (matrix) from which to find the maximum value.
    S : list or np.ndarray
        A list or array of indices representing the rows and columns to exclude from consideration.

    Outputs
    -------
    tuple
        A tuple containing the indices (i, j) of the maximum value found in M,
        where both i and j are indices of the original matrix M.
    '''

    # Create a boolean mask with the same shape as M, initialized to False
    mask = np.zeros(M.shape, dtype=bool)

    # Create a boolean values matrix of the same size as S, initialized to True
    values = np.ones((len(S), len(S)), dtype=bool)

    # Update the mask to include True values for the indices specified in S
    mask[np.ix_(S, S)] = values

    # Set the diagonal of the mask to False to exclude self-connections (i.e., same row and column)
    np.fill_diagonal(mask, 0)

    # Find the maximum value in the matrix M, using the mask to ignore the excluded rows and columns
    max_value = M[mask].max()

    # Find the indices of the maximum value in the original matrix M, applying the mask to avoid excluded areas
    # Multiply by 1 to convert boolean to integer
    i, j = np.where(np.multiply(M, mask * 1) == max_value)

    # Return the first pair of indices (i[0], j[0]) where the maximum value is located
    return i[0], j[0]


def clust(Theta, alpha):
    """ 
    Performs clustering in the AI-block model using an extremal correlation matrix.

    Inputs
    ------
        Theta : np.ndarray
            An extremal correlation matrix of shape (d, d), where d is the number of variables.
        alpha : float
            A threshold value of order sqrt(ln(d)/n), used to determine cluster membership.

    Outputs
    -------
        dict
            A dictionary representing a partition of the set {1, ..., d}, 
            where each key is a cluster label, and the corresponding value 
            is a numpy array containing the indices of the elements in that cluster.
    """
    # Get the number of variables (dimensions) from the shape of Theta
    d = Theta.shape[1]

    # Initialize the list of indices to include all variables
    S = np.arange(d)
    l = 0  # Initialize a counter for cluster labels

    # Dictionary to store the resulting clusters
    cluster = {}

    # Loop until all elements have been assigned to clusters
    while len(S) > 0:
        l = l + 1  # Increment the cluster label

        if len(S) == 1:  # If only one element remains in S
            # Assign the single element to a new cluster
            cluster[l] = np.array(S)
        else:
            # Find the pair of indices (a_l, b_l) with the maximum correlation
            # in the submatrix defined by the remaining indices S
            a_l, b_l = find_max(Theta, S)

            # Check if the maximum correlation value is below the threshold alpha
            if Theta[a_l, b_l] < alpha:
                # If the correlation is low, create a singleton cluster with a_l
                cluster[l] = np.array([a_l])
            else:
                # Identify indices of elements in S that have correlations
                # greater than or equal to alpha with a_l and b_l
                index_a = np.where(Theta[a_l, :] >= alpha)
                index_b = np.where(Theta[b_l, :] >= alpha)

                # Find the intersection of indices in S with those indices
                # to form a cluster
                cluster[l] = np.intersect1d(S, index_a, index_b)

        # Remove the current cluster elements from the list S
        S = np.setdiff1d(S, cluster[l])

    return cluster  # Return the final clusters as a dictionary


def perc_exact_recovery(O_hat, O_bar):
    """Calculate the percentage of exact recovery of clusters.

    Inputs
    ------
        O_hat : dict
            A dictionary where keys are estimated cluster labels and values are arrays of indices representing 
            the estimated clusters.
        O_bar : dict
            A dictionary where keys are true cluster labels and values are arrays of indices representing 
            the true clusters.

    Outputs
    -------
        float
            The percentage of true clusters that were exactly recovered in the estimated clusters.
    """
    value = 0  # Initialize a counter for exact recoveries

    # Iterate through each true cluster in O_bar
    for key1, true_clust in O_bar.items():
        # Iterate through each estimated cluster in O_hat
        for key2, est_clust in O_hat.items():
            # Check if the true cluster and estimated cluster have the same size
            if len(true_clust) == len(est_clust):
                # Find the intersection of the true cluster and estimated cluster
                test = np.intersect1d(true_clust, est_clust)

                # Check if the intersection is non-empty and matches the true cluster
                if (len(test) > 0 and
                    len(test) == len(true_clust) and
                        np.sum(np.sort(test) - np.sort(true_clust)) == 0):
                    value += 1  # Increment the counter for exact recovery

    # Calculate and return the percentage of exact recoveries
    return value / len(O_bar)
