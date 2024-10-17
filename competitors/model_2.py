"""
# Numerical Experimentation in Experiment E2 and Framework F1

## Purpose
This script conducts numerical experiments within Experiment E2 and Framework F1. It generates artificial data and evaluates various clustering algorithms using the Adjusted Rand Index (ARI) as a performance metric.

## Description
The code initializes parameters for a comparative analysis of clustering algorithms applied to data generated via the Clayton copula. It employs several algorithms including:
- DAMEX (Detecting Anomaly among Multivariate Extremes)
- CLEF (CLustering Extreme Features)
- MUSCLE (MUltivariate Sparse CLustering for Extremes)
- ECO (Extremal COrrelation)
- SKmeans (Spherical Kmeans)

The experiments vary based on the sparsity index and dimensionality factors. The code generates datasets, performs clustering, and evaluates performance metrics. 

## Parameters
- `k` : Number of block maxima
- `p` : Order of the moving maxima process
- `m` : Length of each block
- `dim_fact` : List of dimensionality factors for generating artificial data
- `sparsity_index` : Index determining the sparsity of the generated data
- `niter` : Number of iterations for each clustering algorithm

## Functions
- `runif_in_simplex(n)`: Returns a uniformly random vector in the n-simplex.

## Dependencies
- clayton
- numpy
- pandas
- matplotlib
- sklearn
- rpy2
"""


from clayton.rng.archimedean import Clayton
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import eco_alg
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import adjusted_rand_score

import damex as dmx
import clef as clf
import utilities as ut
import ut_eco as ut_eco

from rpy2 import robjects as ro
import os

np.random.seed(42)


def runif_in_simplex(n):
    '''
    Generate a uniformly random vector in the n-simplex.

    The n-simplex is the set of points in n-dimensional space that sum to 1 
    and have non-negative coordinates. This function samples from a 
    probability distribution that produces a point uniformly within the n-simplex.

    Parameters:
    n (int): The dimension of the simplex. It specifies the number of elements
             in the output vector.

    Returns:
    numpy.ndarray: A 1D array of shape (n,) containing a uniformly random 
                   vector in the n-simplex, where each element is non-negative 
                   and the sum of all elements equals 1.
    '''

    # Generate n samples from an exponential distribution
    k = np.random.exponential(scale=1.0, size=n)

    # Normalize the samples by dividing by their sum to ensure they sum to 1
    return k / sum(k)


# Dimension multiplier
dim_fact = [1, 2, 4, 8, 16, 32]

# Number of block maxima
_k_ = [100, 150, 200, 250, 300, 350, 400, 450, 500]

# Threshold for SKmeans clustering
seuil_skmeans = [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]

for k in _k_:
    for etape, dim in enumerate(dim_fact):
        # Print dimensions and number of clusters
        print('dim:', dim, 'k:', k)

        # Define lengths of clusters
        D = [6, 5, 4, 3, 2] * dim

        # Calculate cumulative sum of cluster lengths
        cusum = np.cumsum(D)

        # Set other parameters
        p = 2
        m = 20
        n = k * m
        K = 100 * dim
        d = np.sum(D)
        niter = 50
        step = int(K / len(D))

        # Initialize dictionary to store true clusters
        true_cluster = {}

        # Assign data points to true clusters
        for i in range(0, len(D)):
            if i == 0:
                true_cluster[str(i+1)] = np.arange(0, cusum[i])
            else:
                true_cluster[str(i+1)] = np.arange(cusum[i-1], cusum[i])

        # Initialize array to store true labels
        true_labels = np.zeros(d)

        # Assign true labels to data points based on true clusters
        for key, values in true_cluster.items():
            true_labels[values] = key

        # Initialize arrays to store ARI values for different clustering methods
        ari_dmx = []
        ari_clf = []
        ari_muscle = []
        ari_skmeans = []
        ari_eco_seco = []

        for i in range(niter):
            # Print current iteration index
            print(i)

            # Generate Data

            # Print message indicating data generation process
            print('... Generate Data ...')

            # Sample Clayton copula
            clayton = Clayton(dim=K, n_samples=n+p, theta=1.0)
            sample_unimargin = clayton.sample_unimargin()
            sample_Cau = 1/(1-sample_unimargin)  # Set to Pareto margins

            # Initialize mean and covariance for the multivariate normal distribution
            mean = np.zeros(d)
            cov = np.eye(d)

            # Set coefficient of the moving maxima process
            rho = 0.8
            l = 0
            A = np.zeros((d, K))
            first = 0
            # Generate random data matrix A
            for dim in D:
                second = first + dim
                for j in range(first, second):
                    index = np.arange(l*step, step*(l+1))
                    A[j, index] = runif_in_simplex(step)
                l += 1
                first += dim

            # Calculate W matrix based on sample_Cau (Clayton copula + Pareto margins) and coefficient of the moving maxima process rho
            W = np.array([np.max(np.c_[np.power(rho, 0)*sample_Cau[i, :], np.power(rho, 1) *
                         sample_Cau[i-1, :], np.power(rho, 2)*sample_Cau[i-2, :]], axis=1) for i in range(2, n+p)])

            # Generate final data matrix X by multiplying A with W and adding multivariate normal noise
            X = np.array([np.matmul(A, W[i, :]) for i in range(n)]) + \
                np.random.multivariate_normal(mean, cov, size=1)[0]

            # Saving data for R source file
            print('... Saving for R source file ...')

            # Perform rank transformation on data matrix X
            V = ut.rank_transformation(X)

            # Convert transformed data to DataFrame
            export = pd.DataFrame(V)

            # Export DataFrame to CSV file
            export.to_csv("results_model_2/data/data/" +
                          "data_" + str(1) + ".csv")

            print('... Source R file ...')

            # Execute R script and remove temporary CSV file

            # Import R interface

            r = ro.r

            # Source the R script 'model_1.R'

            r.source('model_2.R')

            # Remove temporary CSV file

            os.remove("results_model_2/data/data/data_" + str(1) + ".csv")

            print('... File Removed ...')
            # Skmean extreme

            print('... SKmean Estimation ...')
#
            # Check if the number of true clusters is less than the square root of n, i.e., the number of extremes

            if len(true_cluster) < int(np.sqrt(n)):
                # Read cluster centers from CSV file
                centers = np.matrix(pd.read_csv(
                    'results_model_2/results_skmeans/skmeans/centers_'+str(1)+'.csv'))
                nclusters = centers.shape[0]
                faces_skmeans = []
                for j in range(nclusters):
                    # Extract indices where cluster centers exceed the threshold
                    indices = np.where(centers[j, :] > seuil_skmeans[etape])[1]
                    faces_skmeans.append(indices)

                clusters = {}  # Initialize dictionary to store clusters
                l = 1  # Initialize cluster label
                S = np.arange(len(faces_skmeans))  # Initialize set of indices

                while len(S) > 0:
                    i = S[0]
                    # Assign face indices to cluster
                    clusters[l] = faces_skmeans[i]
                    iterand = S
                    while len(iterand) > 0:
                        j = iterand[0]
                        if set(clusters[l]) & set(faces_skmeans[j]):
                            clusters[l] = np.union1d(
                                clusters[l], faces_skmeans[j])  # Merge clusters with intersecting faces
                            # Remove merged cluster index from set
                            S = np.setdiff1d(S, j)
                            iterand = S
                        else:
                            # Continue with next index
                            iterand = np.setdiff1d(iterand, j)
                    l += 1

                # Initialize predicted labels array
                pred_labels_skmeans = np.zeros(d)
                # Assign cluster labels to data points based on SKmeans clustering
                for key, values in clusters.items():
                    pred_labels_skmeans[values] = key

                # Calculate Adjusted Rand Index for SKmeans clustering
                ari_skmeans.append(adjusted_rand_score(
                    pred_labels_skmeans, true_labels))

            # Modèle Muscle

            # Muscle Estimation

            # Print message indicating Muscle estimation process
            print('... Muscle Estimation ...')

            # Read Muscle estimation results from CSV file
            Memp = pd.read_csv(
                'results_model_2/results_muscle/muscle/Memp_'+str(1)+'.csv')
            Memp = np.matrix(Memp)
            ndirections = int(Memp.shape[1])
            faces_muscle = []

            # Process Muscle estimation results
            if ndirections == 1:
                faces_muscle.append(np.where(Memp[:, 0] == 1)[0])
            if ndirections >= 2:
                for j in range(1, int(ndirections)):
                    faces_muscle.append(np.where(Memp[:, j] == 1)[0])

            clusters = {}  # Initialize dictionary to store clusters
            l = 1  # Initialize cluster label
            S = np.arange(len(faces_muscle))  # Initialize set of indices

            while len(S) > 0:
                i = S[0]
                clusters[l] = faces_muscle[i]  # Assign face indices to cluster
                iterand = S
                while len(iterand) > 0:
                    j = iterand[0]
                    if set(clusters[l]) & set(faces_muscle[j]):
                        clusters[l] = np.union1d(
                            clusters[l], faces_muscle[j])  # Merge clusters with intersecting faces
                        # Remove merged cluster index from set
                        S = np.setdiff1d(S, j)
                        iterand = S
                    else:
                        # Continue with next index
                        iterand = np.setdiff1d(iterand, j)
                l += 1

            # Initialize predicted labels array
            pred_labels_muscle = np.zeros(d)
            # Assign cluster labels to data points based on Muscle estimation clustering
            for key, values in clusters.items():
                pred_labels_muscle[values] = key

            # Calculate Adjusted Rand Index for Muscle estimation clustering
            ari_muscle.append(adjusted_rand_score(
                pred_labels_muscle, true_labels))

            # ECO Estimation

            # Print message indicating ECO estimation process
            print('... ECO Estimation ...')

            block_maxima = np.zeros((k, d))

            # Compute block maxima
            for j in range(d):
                sample = X[0:(k*m), j]
                sample = sample.reshape((k, m))
                block_maxima[:, j] = np.max(sample, axis=1)

            block_maxima = pd.DataFrame(block_maxima)

            # Compute empirical rank
            erank = np.array(block_maxima.rank() / (k + 1))

            outer = (np.maximum(erank[:, :, None],
                     erank[:, None, :])).sum(0) / k

            # Calculate extreme coefficients
            extcoeff = -np.divide(outer, outer - 1)
            Theta = np.maximum(2 - extcoeff, 10e-5)

            ### Tuning Parameter Using SECO ###

            _tau_ = np.array(np.arange(1.25, 1.75, step=0.005))
            value_crit = 10e5
            tuned_delta = _tau_[0] * (1/m + np.sqrt(np.log(d) / k))

            # Tune delta parameter using SECO
            for tau in _tau_:
                delta = tau * (1/m + np.sqrt(np.log(d) / k))
                clusters = eco_alg.clust(Theta, delta)
                value = ut_eco.SECO(erank, clusters)
                if value < value_crit:
                    tuned_delta = delta
                    value_crit = value

            O_hat = eco_alg.clust(Theta, tuned_delta)
            pred_labels_eco = np.zeros(d)

            # Assign cluster labels to data points based on ECO estimation clustering
            for key, values in O_hat.items():
                pred_labels_eco[values] = key

            # Calculate Adjusted Rand Index for ECO estimation clustering
            ari_eco_seco.append(adjusted_rand_score(
                pred_labels_eco, true_labels))

            del (tuned_delta)  # Clean up tuned delta parameter

            # DAMEX Estimation

            # Print message indicating DAMEX estimation process
            print('... DAMEX Estimation ...')

            # Set parameters for DAMEX algorithm
            R = int(np.sqrt(n))
            eps = 0.3  # 0.3
            nb_min = 8  # 8

            # Run DAMEX algorithm to obtain faces
            faces_dmx = dmx.damex(V, R, eps, nb_min)

            clusters = {}  # Initialize dictionary to store clusters
            l = 1  # Initialize cluster label
            S = np.arange(len(faces_dmx))  # Initialize set of indices

            # Process faces to assign data points to clusters
            while len(S) > 0:
                i = S[0]
                clusters[l] = faces_dmx[i]  # Assign face indices to cluster
                iterand = S
                while len(iterand) > 0:
                    j = iterand[0]
                    if set(clusters[l]) & set(faces_dmx[j]):
                        clusters[l] = np.union1d(
                            clusters[l], faces_dmx[j])  # Merge clusters with intersecting faces
                        # Remove merged cluster index from set
                        S = np.setdiff1d(S, j)
                        iterand = S
                    else:
                        # Continue with next index
                        iterand = np.setdiff1d(iterand, j)
                l += 1

            pred_labels_dmx = np.zeros(d)  # Initialize predicted labels array
            # Assign cluster labels to data points based on DAMEX estimation clustering
            for key, values in clusters.items():
                pred_labels_dmx[values] = key

            # Calculate Adjusted Rand Index for DAMEX estimation clustering
            ari_dmx.append(adjusted_rand_score(pred_labels_dmx, true_labels))

            # Modèle CLEF
            # CLEF Estimation

            # Print message indicating CLEF estimation process
            print('... CLEF Estimation ...')

            # Check if the dimension is less than or equal to 640
            if d <= 640:
                # Perform rank transformation on data
                V = ut.rank_transformation(X)
                R = int(np.sqrt(n))

                # Run CLEF algorithm to obtain faces
                faces_clf = clf.clef(V, R, kappa_min=0.2)

                clusters = {}  # Initialize dictionary to store clusters
                l = 1  # Initialize cluster label
                S = np.arange(len(faces_clf))  # Initialize set of indices

                # Process faces to assign data points to clusters
                while len(S) > 0:
                    i = S[0]
                    # Assign face indices to cluster
                    clusters[l] = faces_clf[i]
                    iterand = S
                    while len(iterand) > 0:
                        j = iterand[0]
                        if set(clusters[l]) & set(faces_clf[j]):
                            clusters[l] = np.union1d(
                                clusters[l], faces_clf[j])  # Merge clusters with intersecting faces
                            # Remove merged cluster index from set
                            S = np.setdiff1d(S, j)
                            iterand = S
                        else:
                            # Continue with next index
                            iterand = np.setdiff1d(iterand, j)
                    l += 1

                # Initialize predicted labels array
                pred_labels_clf = np.zeros(d)
                # Assign cluster labels to data points based on CLEF estimation clustering
                for key, values in clusters.items():
                    pred_labels_clf[values] = key

                # Calculate Adjusted Rand Index for CLEF estimation clustering
                ari_clf.append(adjusted_rand_score(
                    pred_labels_clf, true_labels))

        # Convert ARI results to DataFrames and save to CSV files

        # Convert ARI results to DataFrames

        ari_dmx = pd.DataFrame(ari_dmx)
        ari_clf = pd.DataFrame(ari_clf)
        ari_muscle = pd.DataFrame(ari_muscle)
        ari_skmeans = pd.DataFrame(ari_skmeans)
        ari_eco_seco = pd.DataFrame(ari_eco_seco)

        # Print ARI results
        print(ari_dmx)
        print(ari_muscle)
        print(ari_clf)
        print(ari_skmeans)
        print(ari_eco_seco)

        # Save ARI results to CSV files

        pd.DataFrame.to_csv(ari_dmx, "results_model_2/results/model_2_" +
                            str(int(d)) + "_" + str(int(k)) + "/ari_dmx" + ".csv")
        pd.DataFrame.to_csv(ari_clf, "results_model_2/results/model_2_" +
                            str(int(d)) + "_" + str(int(k)) + "/ari_clf" + ".csv")
        pd.DataFrame.to_csv(ari_muscle, "results_model_2/results/model_2_" +
                            str(int(d)) + "_" + str(int(k)) + "/ari_muscle" + ".csv")
        pd.DataFrame.to_csv(ari_skmeans, "results_model_2/results/model_2_" +
                            str(int(d)) + "_" + str(int(k)) + "/ari_skmeans" + ".csv")
        pd.DataFrame.to_csv(ari_eco_seco, "results_model_2/results/model_2_" +
                            str(int(d)) + "_" + str(int(k)) + "/ari_eco_seco" + ".csv")
