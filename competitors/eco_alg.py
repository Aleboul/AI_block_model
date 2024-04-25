import numpy as np

def find_max(M, S):
    mask = np.zeros(M.shape, dtype = bool)
    values = np.ones((len(S),len(S)),dtype = bool)
    mask[np.ix_(S,S)] = values
    np.fill_diagonal(mask,0)
    max_value = M[mask].max()
    i , j = np.where(np.multiply(M, mask * 1) == max_value) # Sometimes doublon happens for excluded clusters, if n is low
    return i[0], j[0]

def clust(Theta,alpha):
    """ Performs clustering in AI-block model
    
    Inputs
    ------
        Theta : extremal correlation matrix
        alpha : threshold, of order sqrt(ln(d)/n)
    
    Outputs
    -------
        Partition of the set \{1,\dots, d\}
    """
    d = Theta.shape[1]

    # Initialisation

    S = np.arange(d)
    l = 0
    
    cluster = {}
    while len(S) > 0:
        l = l + 1
        if len(S) == 1:
            cluster[l] = np.array(S)
        else :
            a_l, b_l = find_max(Theta, S)
            if Theta[a_l,b_l] < alpha :
                cluster[l] = np.array([a_l])
            else :
                index_a = np.where(Theta[a_l,:] >= alpha)
                index_b = np.where(Theta[b_l,:] >= alpha)
                cluster[l] = np.intersect1d(S,index_a,index_b)
        S = np.setdiff1d(S, cluster[l])
    
    return cluster

def perc_exact_recovery(O_hat, O_bar):
    value = 0
    for key1, true_clust in O_bar.items():
        for key2, est_clust in O_hat.items():
            if len(true_clust) == len(est_clust):
                test = np.intersect1d(true_clust,est_clust)
                if len(test) > 0 and len(test) == len(true_clust) and np.sum(np.sort(test) - np.sort(true_clust)) == 0 :
                    value +=1
    return value / len(O_bar)