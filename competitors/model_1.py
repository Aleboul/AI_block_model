"""
    Produce numerical results in Experiment E2 and Framework F2
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

# Initialization of parameters for the comparative analysis


k = 250  # Dumber of block maxima
p = 2  # Order of the moving maxima process
m = 20  # Length of block
n = k * m  # Total number of data points

# Dimensionality factors for creating artificial data
dim_fact = [1, 2, 4, 8, 16]

# Loop for generating artificial data with specified sparsity index and dimension factors

for sparsity_index in [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:

    for etape, dim in enumerate(dim_fact):
        print('dim:', dim)
        # Length of clusters multiplied by dimension factor
        D = np.array([6, 5, 4, 3, 2])*dim
        cusum = np.cumsum(D)
        K = 100  # Number of latent factors
        d = np.sum(D)  # Number of the dimension
        niter = 10  # Number of iterations
        step_clust = int(K / len(D))  # step

        # Generating true cluster assignments and labels

        true_cluster = {}  # Dictionary to store true cluster assignments

        # Generate true cluster assignments based on cumulative sums of cluster lengths
        for i in range(0, len(D)):

            if i == 0:
                true_cluster[str(i+1)] = np.arange(0, cusum[i])
            else:
                true_cluster[str(i+1)] = np.arange(cusum[i-1], cusum[i])

        true_labels = np.zeros(d)
        # Assign true labels based on true cluster assignments
        for key, values in true_cluster.items():
            true_labels[values] = key

        # Initializing lists to store Adjusted Rand Index scores for different algorithms

        ari_dmx = []  # DAMEX Algorithm
        ari_clf = []  # CLEF Algorithm
        ari_muscle = []  # MUSCLE Algorithm
        ari_eco_seco = []  # ECO + SECO Algorithm

        for i in range(niter):
            print(i)

            # Generating a sample using the Clayton copula

            # Initialize Clayton copula with specified dimension and sample size
            clayton = Clayton(dim=K, n_samples=n+p, theta=1.0)

            # Generate unimarginal samples from the Clayton copula
            sample_unimargin = clayton.sample_unimargin()

            # Transform samples to have Pareto margins
            sample_Cau = 1/(1-sample_unimargin)

            # Initialize mean and covariance for the multivariate normal distribution
            mean = np.zeros(d)
            cov = np.eye(d)

            # Set coefficient of the moving maxima process
            rho = 0.8
            l = 0
            A = np.zeros((d, K))
            first = 0

            # Generate random data matrix A with specified sparsity index
            for dim in D:
                second = first + dim
                for j in range(first, second):
                    union_support = np.array(
                        [j % step_clust + l*step_clust, (j+1) % step_clust + l*step_clust])
                    index = np.arange(l*step_clust, step_clust*(l+1))
                    index = np.setdiff1d(index, union_support)
                    support = np.random.choice(
                        index, size=sparsity_index-2, replace=False)
                    support = np.union1d(support, union_support)
                    A[j, support] = runif_in_simplex(len(support))
                l += 1
                first += dim

            # Update dimensionality after generating matrix A to be sure
            d = A.shape[0]

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
            export.to_csv("results_model_1/data/data/" +
                          "data_" + str(1) + ".csv")

            print('... Source R file ...')

            # Execute R script and remove temporary CSV file

            # Import R interface
            r = ro.r

            # Source the R script 'model_1.R'
            r.source('model_1.R')

            # Remove temporary CSV file
            os.remove("results_model_1/data/data/data_" + str(1) + ".csv")

            print('... File Removed ...')

            # Modèle Muscle

            print('... Muscle Estimation ...')
            # Load membership matrix from CSV file

            Memp = pd.read_csv(
                'results_model_1/results_muscle/muscle/Memp_'+str(1)+'.csv')
            Memp = np.matrix(Memp)
            ndirections = int(Memp.shape[1])
            faces_muscle = []

            # Extract faces (indices) based on membership matrix

            if ndirections == 1:
                faces_muscle.append(np.where(Memp[:, 0] == 1)[0])
            if ndirections >= 2:
                for j in range(1, ndirections):
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
            # Assign cluster labels to data points based on cluster assignments
            for key, values in clusters.items():
                pred_labels_muscle[values] = key

            # Calculate Adjusted Rand Index for Muscle clustering
            ari_muscle.append(adjusted_rand_score(
                pred_labels_muscle, true_labels))

            # ECO Model
            print('... ECO Estimation ...')

            # Initialize array to store block maxima
            block_maxima = np.zeros((k, d))

            # Calculate componentwise block maxima
            for j in range(d):
                sample = X[0:(k*m), j]
                sample = sample.reshape((k, m))
                block_maxima[:, j] = np.max(sample, axis=1)

            # Convert block maxima to DataFrame
            block_maxima = pd.DataFrame(block_maxima)

            # Compute empirical ranks
            erank = np.array(block_maxima.rank() / (k+1))

            # Calculate outer maxmum of empirical ranks and sum
            outer = (np.maximum(erank[:, :, None],
                     erank[:, None, :])).sum(0) / k

            # Calculate extreme coefficients and extremal correlation
            extcoeff = -np.divide(outer, outer-1)
            Theta = np.maximum(2-extcoeff, 10e-5)

            ### Tuning Parameter Using SECO ###

            # Initialize array for tuning parameter tau
            _tau_ = np.array(np.arange(1.25, 1.75, step=0.005))
            value_crit = 10e5

            # Initialize tuned delta
            tuned_delta = _tau_[0]*(1/m+np.sqrt(np.log(d)/k))

            # Iterate over tuning parameter tau
            for tau in _tau_:
                delta = tau*(1/m+np.sqrt(np.log(d)/k))
                clusters = eco_alg.clust(Theta, delta)
                value = ut_eco.SECO(erank, clusters)
                if value < value_crit:
                    tuned_delta = delta
                    value_crit = value

            # Perform ECO clustering with tuned delta
            O_hat = eco_alg.clust(Theta, tuned_delta)

            # Assign cluster labels to data points based on ECO clustering with tuned delta
            pred_labels_eco = np.zeros(d)
            for key, values in O_hat.items():
                pred_labels_eco[values] = key

            # Calculate Adjusted Rand Index for ECO clustering with tuned delta
            ari_eco_seco.append(adjusted_rand_score(
                pred_labels_eco, true_labels))

            # Delete tuned_delta variable
            del (tuned_delta)

            # Modèle DAMEX
            # print('... DAMEX Estimation ...')

            # Define parameter R, number of extremes, as square root of the total number of data points
            R = int(np.sqrt(n))

            # Set parameters
            eps = 0.3  # 0.3
            nb_min = 8  # 8

            # Perform DAMEX algorithm
            faces_dmx = dmx.damex(V, R, eps, nb_min)

            clusters = {}  # Initialize dictionary to store clusters
            l = 1  # Initialize cluster label
            S = np.arange(len(faces_dmx))  # Initialize set of indices
            while len(S) > 0:
                i = S[0]
                clusters[l] = faces_dmx[i]  # Assign face indices to cluster
                iterand = S
                while len(iterand) > 0:
                    j = iterand[0]
                    if set(clusters[l]) & set(faces_dmx[j]):
                        # Merge clusters with intersecting faces
                        clusters[l] = np.union1d(clusters[l], faces_dmx[j])
                        # Remove merged cluster index from set
                        S = np.setdiff1d(S, j)
                        iterand = S
                    else:
                        iterand = np.setdiff1d(iterand, j)
                l += 1  # Continue with next index

            pred_labels_dmx = np.zeros(d)  # Initialize predicted labels array
            # Assign cluster labels to data points based on DAMEX Algorithm
            for key, values in clusters.items():
                pred_labels_dmx[values] = key

            # Calculate Adjusted Rand Index for DAMEX Algorithm
            ari_dmx.append(adjusted_rand_score(pred_labels_dmx, true_labels))

            # Modèle CLEF

            print('... CLEF Estimation ...')

            # marche si d <=80 ou sparsity_index <= 12 pour d = 320. Si d = 160 alors prendre d <=40 et sparsity_index <= 16
            # Check conditions for executing CLEF clustering due to memory limitations
            if (d <= 40) or (sparsity_index <= 16):

                # Define parameter R, number of extremes, as square root of the total number of data points
                R = int(np.sqrt(n))
                try:
                    # Perform CLEF clustering
                    faces_clf = clf.clef(V, R, kappa_min=0.2)
                    clusters = {}  # Initialize dictionary to store clusters
                    l = 1  # Initialize cluster label
                    S = np.arange(len(faces_clf))  # Initialize set of indices
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
                                iterand = np.setdiff1d(iterand, j)
                        l += 1  # Continue with next index

                    pred_labels_clf = np.zeros(d)

                    # Assign cluster labels to data points based on CLEF Algorithm
                    for key, values in clusters.items():
                        pred_labels_clf[values] = key

                    # Calculate Adjusted Rand Index for CLEF clustering
                    ari_clf.append(adjusted_rand_score(
                        pred_labels_clf, true_labels))
                except MemoryError:
                    pass

        # Convert ARI results to DataFrames and save to CSV files

        # Convert ARI results to DataFrames
        ari_dmx = pd.DataFrame(ari_dmx)
        ari_clf = pd.DataFrame(ari_clf)
        ari_muscle = pd.DataFrame(ari_muscle)
        ari_eco_seco = pd.DataFrame(ari_eco_seco)

        # Print ARI results
        print(ari_dmx)
        print(ari_muscle)
        print(ari_clf)
        print(ari_eco_seco)

        # Save ARI results to CSV files
        pd.DataFrame.to_csv(ari_dmx, "results_model_1/results/model_1_" +
                           str(int(d)) + "_" + str(int(sparsity_index)) + "/ari_dmx" + ".csv")
        pd.DataFrame.to_csv(ari_clf, "results_model_1/results/model_1_" +
                           str(int(d)) + "_" + str(int(sparsity_index)) + "/ari_clf" + ".csv")
        pd.DataFrame.to_csv(ari_muscle, "results_model_1/results/model_1_" +
                           str(int(d)) + "_" + str(int(sparsity_index)) + "/ari_muscle" + ".csv")
        pd.DataFrame.to_csv(ari_eco_seco, "results_model_1/results/model_1_" + str(
           int(d)) + "_" + str(int(sparsity_index)) + "/ari_eco_seco" + ".csv")
