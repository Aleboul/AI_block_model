import numpy as np

def theta(R):
    """
    Computes the extremal coefficient from a rank's matrix.
    
    Inputs
    ------
    R : np.ndarray
        A 2D numpy array of shape (n_sample, d) representing the rank's matrix,
        where n_sample is the number of samples and d is the dimensionality.
    
    Outputs
    -------
    float
        The computed extremal coefficient value.
    """

    # Get the number of dimensions (Nnb) and number of samples (Tnb)
    Nnb = R.shape[1]  # Number of dimensions
    Tnb = R.shape[0]  # Number of samples

    # Initialize a matrix V with zeros, having the same shape as R
    V = np.zeros([Tnb, Nnb])

    # Copy the values from the rank matrix R into V
    for j in range(0, Nnb):  # Iterate over each dimension
        V[:, j] = R[:, j]  # Assign values from R to V

    # Compute the maximum value for each sample across all dimensions
    value_1 = np.amax(V, 1)  # Max across rows for each sample

    # Compute the average value for each sample across all dimensions
    value_2 = (1/Nnb) * np.sum(V, 1)  # Average across rows for each sample

    # Compute the w-madogram using the formula
    mado = (1/Tnb) * np.sum(value_1 - value_2)

    # Normalize the computed w-madogram
    value = (mado + 1/2) / (1/2 - mado)

    return value  # Return the final computed w-madogram

def SECO(R, clst):
    """
    Evaluation of the SECO criterion.

    Inputs
    ------
    R : np.ndarray
        A rank matrix of shape (n, d) where n is the number of samples and d is the number of variables/dimensions.
    clst : dict
        A dictionary representing a partition of the columns (or variables). 
        Keys are cluster labels, and values are lists/arrays of column indices that belong to each cluster.

    Output
    ------
    float
        The evaluation of the SECO criterion, computed as (theta - theta_Σ), 
        where theta is the w-madogram of the entire rank matrix and theta_Σ is the sum of w-madograms for each cluster.
    """

    # d is the number of rows (samples) in the rank matrix R
    d = R.shape[0]

    # Compute the w-madogram for the entire rank matrix R
    value = theta(R)

    # Initialize an empty list to store the w-madogram values for each cluster
    _value_ = []

    # Iterate over each cluster (key is cluster label, c is the array of column indices in that cluster)
    for key, c in clst.items():
        # Extract the submatrix corresponding to the current cluster by selecting the columns in c
        _R_2 = R[:, c]

        # Compute the w-madogram for the submatrix of the current cluster and append it to _value_
        _value_.append(theta(_R_2))

    # Return the sum of the w-madograms of the clusters minus the w-madogram of the full matrix
    return np.sum(_value_) - value