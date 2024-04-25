"""
    Numerical results for Experiment E1 and Framework F2
"""

from clayton.rng.archimedean import Clayton
import numpy as np
from scipy.stats import pareto
import pandas as pd
import matplotlib.pyplot as plt
import eco_alg
from sklearn.metrics.cluster import adjusted_rand_score

import damex as dmx
import clef as clf
import hill as hill
import utilities as ut
import ut_eco as ut_eco

from rpy2 import robjects as ro
import os

np.random.seed(42)


def runif_in_simplex(n):
    ''' Return uniformly random vector in the n-simplex '''

    k = np.random.exponential(scale=1.0, size=n)
    return k / sum(k)

# Setting Parameters


# Define parameters for data generation
k = 250  # Number of block maxima
p = 2  # Dimension of the additional variables
m = 20  # Number of observations per cluster
n = k * m  # Total number of observations

# Define the multiplier for cluster dimensions
dim_fact = [1, 2, 4, 8, 16]


for sparsity_index in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:

    for etape, dim in enumerate(dim_fact):
        # Data Generation

        # Print message indicating the current dimension
        print('dim:', dim)

        # Define the lengths of clusters based on the multiplier
        D = [6, 5, 4, 3, 2] * dim

        # Calculate cumulative sum of cluster lengths
        cusum = np.cumsum(D)

        # Calculate total number of dimensions for data generation
        K = 100 * dim

        # Calculate total number of data points
        d = np.sum(D)

        # Set the number of iterations for clustering algorithms
        niter = 50

        # Calculate step size for cluster assignment
        step = int(K / len(D))

        # Initialize dictionary to store true cluster assignments
        true_cluster = {}

        # Assign data points to true clusters
        for i in range(0, len(D)):
            if i == 0:
                true_cluster[str(i + 1)] = np.arange(0, cusum[i])
            else:
                true_cluster[str(i + 1)] = np.arange(cusum[i - 1], cusum[i])

        true_labels = np.zeros(d)

        for key, values in true_cluster.items():
            true_labels[values] = key

        # Initialize array to store Adjusted Rand Index for each clustering algorithm
        ari_dmx = []
        ari_clf = []
        ari_muscle = []
        ari_eco_seco = []

        for i in range(niter):
            # Data Generation and Processing

            # Print the current value of the iteration index
            print(i)

            # Sample from Clayton copula
            clayton = Clayton(dim=K, n_samples=n+p, theta=1.0)
            sample_unimargin = clayton.sample_unimargin()
            sample_Cau = 1 / (1 - sample_unimargin)  # Set to Pareto margins

            # Define parameters for multivariate normal distribution
            mean = np.zeros(d)
            cov = np.eye(d)

            # Set correlation coefficient for the copula
            rho = 0.8

            # Initialize matrix A to store data points
            A = np.zeros((d, K))

            # Initialize variables for loop iteration
            l = 0
            first = 0

            # Generate data points for each dimension
            for dim in D:
                second = first + dim
                for j in range(first, second):
                    index = np.arange(l * step, step * (l + 1))
                    # Randomly select support points for each dimension
                    support = np.random.choice(
                        index, size=sparsity_index, replace=False)
                    A[j, support] = runif_in_simplex(len(support))
                l += 1
                first += dim

            # Update the total number of dimensions
            d = A.shape[0]

            # Generate weights for the copula
            W = np.array([np.max(np.c_[np.power(rho, 0) * sample_Cau[i, :], np.power(rho, 1) *
                         sample_Cau[i - 1, :], np.power(rho, 2) * sample_Cau[i - 2, :]], axis=1) for i in range(2, n + p)])

            # Compute final data points by matrix multiplication and adding multivariate normal noise
            X = np.array([np.matmul(A, W[i, :]) for i in range(n)]) + \
                np.random.multivariate_normal(mean, cov, size=1)[0]

            # Saving Data for R Source File and Post-Processing

            # Print message indicating saving process
            print('... Saving for R source file ...')

            # Perform rank transformation on the data
            V = ut.rank_transformation(X)

            # Convert transformed data to DataFrame
            export = pd.DataFrame(V)

            # Save DataFrame to CSV file
            export.to_csv("results_model_3/data/data/" +
                          "data_" + str(1) + ".csv")

            # Print message indicating source file execution
            print('... Source R file Muscle ...')

            # Execute R source file
            r = ro.r
            r.source('model_3.R')

            # Remove the saved CSV file
            os.remove("results_model_3/data/data/data_" + str(1) + ".csv")

            # Print message indicating file removal
            print('... File Removed ...')

            print('... SKmean Estimation ...')

            # Muscle Estimation

            print('... Muscle Estimation ...')

            # Read the muscle data from the CSV file
            Memp = pd.read_csv(
                'results_model_3/results_muscle/muscle/Memp_'+str(1)+'.csv')
            Memp = np.matrix(Memp)
            ndirections = int(Memp.shape[1])
            faces_muscle = []

            # Identify clusters based on muscle data
            if ndirections == 1:
                faces_muscle.append(np.where(Memp[:, 0] == 1)[0])
            if ndirections >= 2:
                for j in range(1, ndirections):
                    faces_muscle.append(np.where(Memp[:, j] == 1)[0])

            clusters = {}
            l = 1
            S = np.arange(len(faces_muscle))

            # Merge clusters based on shared features
            while len(S) > 0:
                i = S[0]
                clusters[l] = faces_muscle[i]
                iterand = S
                while len(iterand) > 0:
                    j = iterand[0]
                    if set(clusters[l]) & set(faces_muscle[j]):
                        clusters[l] = np.union1d(clusters[l], faces_muscle[j])
                        S = np.setdiff1d(S, j)
                        iterand = S
                    else:
                        iterand = np.setdiff1d(iterand, j)
                l += 1

            # Assign cluster labels to data points
            pred_labels_muscle = np.zeros(d)
            for key, values in clusters.items():
                pred_labels_muscle[values] = key

            # Compute Adjusted Rand Index to assess clustering quality
            ari_muscle.append(adjusted_rand_score(
                pred_labels_muscle, true_labels))

            # ECO Estimation

            print('... ECO Estimation ...')

            # Calculate block maxima
            block_maxima = np.zeros((k, d))
            for j in range(d):
                sample = X[0:(k*m), j]
                sample = sample.reshape((k, m))
                block_maxima[:, j] = np.max(sample, axis=1)
            block_maxima = pd.DataFrame(block_maxima)
            erank = np.array(block_maxima.rank() / (k+1))

            # Compute outer maximum
            outer = (np.maximum(erank[:, :, None],
                     erank[:, None, :])).sum(0) / k

            # Compute extremal coefficient
            extcoeff = -np.divide(outer, outer - 1)
            Theta = np.maximum(2 - extcoeff, 10e-5)

            ### Tuning Parameter Using SECO ###
            _tau_ = np.array(np.arange(1.25, 1.75, step=0.005))
            value_crit = 10e5
            tuned_delta = _tau_[0] * (1/m + np.sqrt(np.log(d) / k))
            for tau in _tau_:
                delta = tau * (1/m + np.sqrt(np.log(d) / k))
                clusters = eco_alg.clust(Theta, delta)
                value = ut_eco.SECO(erank, clusters)
                if value < value_crit:
                    tuned_delta = delta
                    value_crit = value

            # Cluster using ECO algorithm with the tuned delta
            O_hat = eco_alg.clust(Theta, tuned_delta)
            pred_labels_eco = np.zeros(d)
            for key, values in O_hat.items():
                pred_labels_eco[values] = key

            # Compute Adjusted Rand Index to assess clustering quality
            ari_eco_seco.append(adjusted_rand_score(
                pred_labels_eco, true_labels))
            del (tuned_delta)

            # DAMEX Estimation

            print('... DAMEX Estimation ...')

            # Set parameters
            R = int(np.sqrt(n))
            eps = 0.3
            nb_min = 8

            # Apply DAMEX algorithm
            faces_dmx = dmx.damex(V, R, eps, nb_min)

            # Cluster faces using DAMEX algorithm
            clusters = {}
            l = 1
            S = np.arange(len(faces_dmx))
            while len(S) > 0:
                i = S[0]
                clusters[l] = faces_dmx[i]
                iterand = S
                while len(iterand) > 0:
                    j = iterand[0]
                    if set(clusters[l]) & set(faces_dmx[j]):
                        clusters[l] = np.union1d(clusters[l], faces_dmx[j])
                        S = np.setdiff1d(S, j)
                        iterand = S
                    else:
                        iterand = np.setdiff1d(iterand, j)
                l += 1

            # Assign labels based on clusters
            pred_labels_dmx = np.zeros(d)
            for key, values in clusters.items():
                pred_labels_dmx[values] = key

            # Compute Adjusted Rand Index to assess clustering quality
            ari_dmx.append(adjusted_rand_score(pred_labels_dmx, true_labels))

            # CLEF Estimation

            print('... CLEF Estimation ...')

            # Check if the dimension is less than or equal to 640
            if d <= 640:
                # Perform rank transformation
                V = ut.rank_transformation(X)
                R = int(np.sqrt(n))

                # Apply CLEF algorithm
                faces_clf = clf.clef(V, R, kappa_min=0.2)

                # Cluster faces using CLEF algorithm
                clusters = {}
                l = 1
                S = np.arange(len(faces_clf))
                while len(S) > 0:
                    i = S[0]
                    clusters[l] = faces_clf[i]
                    iterand = S
                    while len(iterand) > 0:
                        j = iterand[0]
                        if set(clusters[l]) & set(faces_clf[j]):
                            clusters[l] = np.union1d(clusters[l], faces_clf[j])
                            S = np.setdiff1d(S, j)
                            iterand = S
                        else:
                            iterand = np.setdiff1d(iterand, j)
                    l += 1

                # Assign labels based on clusters
                pred_labels_clf = np.zeros(d)
                for key, values in clusters.items():
                    pred_labels_clf[values] = key

                # Compute Adjusted Rand Index to assess clustering quality
                ari_clf.append(adjusted_rand_score(
                    pred_labels_clf, true_labels))

        ari_dmx = pd.DataFrame(ari_dmx)
        ari_clf = pd.DataFrame(ari_clf)
        ari_muscle = pd.DataFrame(ari_muscle)
        ari_eco_seco = pd.DataFrame(ari_eco_seco)

        print(ari_dmx)
        print(ari_muscle)
        print(ari_clf)
        print(ari_eco_seco)

        pd.DataFrame.to_csv(ari_dmx, "results_model_3/results/model_3_" +
                            str(int(d)) + "_" + str(int(sparsity_index)) + "/ari_dmx" + ".csv")
        pd.DataFrame.to_csv(ari_clf, "results_model_3/results/model_3_" +
                            str(int(d)) + "_" + str(int(sparsity_index)) + "/ari_clf" + ".csv")
        pd.DataFrame.to_csv(ari_muscle, "results_model_3/results/model_3_" +
                            str(int(d)) + "_" + str(int(sparsity_index)) + "/ari_muscle" + ".csv")
        pd.DataFrame.to_csv(ari_eco_seco, "results_model_3/results/model_3_" + str(
            int(d)) + "_" + str(int(sparsity_index)) + "/ari_eco_seco" + ".csv")
