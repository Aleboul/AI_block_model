import numpy as np

def theta(R):
    """
        This function computes the w-madogram

        Inputs
        ------
        R (array([float]) of n_sample \times d) : rank's matrix
                                              w : element of the simplex
                            miss (array([int])) : list of observed data
                           corr (True or False) : If true, return corrected version of w-madogram

        Outputs
        -------
        w-madogram
    """

    Nnb = R.shape[1]
    Tnb = R.shape[0]
    V = np.zeros([Tnb, Nnb])
    cross = np.ones(Tnb)
    for j in range(0, Nnb):
        V[:, j] = R[:, j]
    V *= cross.reshape(Tnb, 1)
    value_1 = np.amax(V, 1)
    value_2 = (1/Nnb) * np.sum(V, 1)
    mado = (1/Tnb) * np.sum(value_1 - value_2)

    value = (mado + 1/2) / (1/2-mado)
    return value


def SECO(R, clst):
    """ evaluation of the criteria

    Input
    -----
        R (np.array(float)) : n x d rank matrix
                  w (float) : element of the simplex
           cols (list(int)) : partition of the columns

    Output
    ------
        Evaluate (theta - theta_\Sigma)

    """

    d = R.shape[0]

    # Evaluate the cluster as a whole

    value = theta(R)

    _value_ = []
    for key, c in clst.items():
        _R_2 = R[:, c]
        _value_.append(theta(_R_2))

    return np.sum(_value_) - value
