"""
    Numerical results for experiment E1 and framework F1
"""

from clayton.rng.archimedean import Clayton
import numpy as np
import pandas as pd
import eco_alg
from sklearn.metrics.cluster import adjusted_rand_score
import matplotlib.pyplot as plt

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


dim_fact = [1, 2, 4, 8, 16, 32]  # Dimension multiplier
_k_ = [100, 150, 200, 250, 300, 350, 400, 450, 500]  # Number of block maxima
seuil_skmeans = [0.15, 0.1, 0.05, 0.05, 0.04,
                 0.025, 0.02]  # Threshold for skmeans


for k in _k_:
    for etape, dim in enumerate(dim_fact):
        print('dim:', dim, 'k:', k)

        # Define the length of clusters based on the dimension factor
        D = np.array([6, 5, 4, 3, 2]) * dim

        # Define parameters
        p = 2  # Number of parameters
        m = 20  # Number of samples
        n = k * m  # Total number of samples
        K = 100  # Number of clusters
        d = np.sum(D)  # Total dimension
        niter = 50  # Number of iterations
        step = int(K / len(D))  # Step size for iteration

        # Define true cluster labels
        true_cluster = {'1': np.arange(0, D[0]), '2': np.arange(D[0], D[1] + D[0]),
                        '3': np.arange(D[1] + D[0], D[2] + D[1] + D[0]),
                        '4': np.arange(D[2] + D[1] + D[0], D[3] + D[2] + D[1] + D[0]),
                        '5': np.arange(D[3] + D[2] + D[1] + D[0], D[4] + D[3] + D[2] + D[1] + D[0])}

        # Initialize true labels
        true_labels = np.zeros(d)

        # Assign true labels to true clusters
        for key, values in true_cluster.items():
            true_labels[values] = key

        # Initialize lists to store adjusted rand index values for different models
        ari_dmx = []
        ari_clf = []
        ari_muscle = []
        ari_skmeans = []
        ari_eco_seco = []

        for i in range(niter):
            print(i)

            # Sample Clayton copula
            print('... Generate Data ...')

            # Generate data using Clayton copula
            clayton = Clayton(dim=K, n_samples=n+p, theta=1.0)
            sample_unimargin = clayton.sample_unimargin()
            sample_Cau = 1 / (1 - sample_unimargin)  # Set to Pareto margins

            # Define parameters for data generation
            mean = np.zeros(d)
            cov = np.eye(d)
            rho = 0.8
            l = 0
            A = np.zeros((d, K))
            first = 0

            # Generate random matrices A and W to create data X
            for dim in D:
                second = first + dim
                for j in range(first, second):
                    index = np.arange(l * step, step * (l + 1))
                    # Sample uniformly on simplex
                    A[j, index] = runif_in_simplex(step)
                l += 1
                first += dim

            # Calculate data points X using A and W matrices
            W = np.array([np.max(np.c_[np.power(rho, 0) * sample_Cau[i, :], np.power(rho, 1) * sample_Cau[i - 1, :],
                                       np.power(rho, 2) * sample_Cau[i - 2, :]], axis=1) for i in range(2, n + p)])
            X = np.array([np.matmul(A, W[i, :]) for i in range(n)]) + \
                np.random.multivariate_normal(mean, cov, size=1)[0]

            print('... Saving for R source file ...')

            # Perform rank transformation on data X
            V = ut.rank_transformation(X)

            # Export transformed data to CSV file
            export = pd.DataFrame(V)
            export.to_csv("results_model_4/data/data/" +
                          "data_" + str(1) + ".csv")

            # Print message indicating source R file
            print('... Source R file Muscle ...')

            # Source R file using R interface
            r = ro.r
            r.source('model_4.R')

            # Remove CSV file after processing
            os.remove("results_model_4/data/data/data_" + str(1) + ".csv")

            # Print message indicating file removal
            print('... File Removed ...')

            # Skmean Algorithm

            print('... Skmeans Estimation ...')

            # Check if the number of true clusters is less than the square root of n
            if len(true_cluster) < int(np.sqrt(n)):
                # Read cluster centers from CSV file
                centers = np.matrix(pd.read_csv(
                    'results_model_4/results_skmeans/skmeans/centers_'+str(1)+'.csv'))
                nclusters = centers.shape[0]
                faces_skmeans = []

                # Iterate over cluster centers to identify extreme clusters
                for j in range(nclusters):
                    indices = np.where(centers[j, :] > seuil_skmeans[etape])[1]
                    faces_skmeans.append(indices)

                # Initialize clusters dictionary
                clusters = {}
                l = 1
                S = np.arange(len(faces_skmeans))

                # Perform clustering based on extreme clusters
                while len(S) > 0:
                    i = S[0]
                    clusters[l] = faces_skmeans[i]
                    iterand = S
                    while len(iterand) > 0:
                        j = iterand[0]
                        if set(clusters[l]) & set(faces_skmeans[j]):
                            clusters[l] = np.union1d(
                                clusters[l], faces_skmeans[j])
                            S = np.setdiff1d(S, j)
                            iterand = S
                        else:
                            iterand = np.setdiff1d(iterand, j)
                    l += 1

                # Assign predicted labels based on clustered data
                pred_labels_skmeans = np.zeros(d)
                for key, values in clusters.items():
                    pred_labels_skmeans[values] = key

                # Compute adjusted Rand score and append to list
                ari_skmeans.append(adjusted_rand_score(
                    pred_labels_skmeans, true_labels))

            # Muscle Model

            print('... Muscle Estimation ...')

            # Read muscle data from CSV file
            Memp = pd.read_csv(
                'results_model_4/results_muscle/muscle/Memp_'+str(1)+'.csv')
            Memp = np.matrix(Memp)
            ndirections = int(Memp.shape[1])
            faces_muscle = []

            # Extract faces for muscle estimation
            if ndirections == 1:
                faces_muscle.append(np.where(Memp[:, 0] == 1)[0])
            if ndirections >= 2:
                for j in range(1, int(ndirections)):
                    faces_muscle.append(np.where(Memp[:, j] == 1)[0])

            # Initialize clusters dictionary
            clusters = {}
            l = 1
            S = np.arange(len(faces_muscle))

            # Perform clustering based on muscle data
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

            # Assign predicted labels based on clustered data
            pred_labels_muscle = np.zeros(d)
            for key, values in clusters.items():
                pred_labels_muscle[values] = key

            # Compute adjusted Rand score and append to list
            ari_muscle.append(adjusted_rand_score(
                pred_labels_muscle, true_labels))

#
            # Modèle eco.
            print('... ECO Estimation ...')
            block_maxima = np.zeros((k, d))
            for j in range(d):
                sample = X[0:(k*m), j]
                sample = sample.reshape((k, m))
                block_maxima[:, j] = np.max(sample, axis=1)
            block_maxima = pd.DataFrame(block_maxima)
            erank = np.array(block_maxima.rank() / (k+1))

            outer = (np.maximum(erank[:, :, None],
                     erank[:, None, :])).sum(0) / k

            extcoeff = -np.divide(outer, outer-1)
            Theta = np.maximum(2-extcoeff, 10e-5)

            ## Tuning Parameter Using SECO ###
            _tau_ = np.array(np.arange(1.25, 1.75, step=0.005))
            value_crit = 10e5
            tuned_delta = _tau_[0]*(1/m+np.sqrt(np.log(d)/k))
            for tau in _tau_:
                delta = tau*(1/m+np.sqrt(np.log(d)/k))
                clusters = eco_alg.clust(Theta, delta)
                value = ut_eco.SECO(erank, clusters)
                if value < value_crit:
                    tuned_delta = delta
                    value_crit = value

            O_hat = eco_alg.clust(Theta, tuned_delta)
            pred_labels_eco = np.zeros(d)
            for key, values in O_hat.items():
                pred_labels_eco[values] = key
            ari_eco_seco.append(adjusted_rand_score(
                pred_labels_eco, true_labels))
            del (tuned_delta)

            # Modèle DAMEX
            print('... DAMEX Estimation ...')

            # Calculate the number of extremes for DAMEX
            R = int(np.sqrt(n))
            eps = 0.3  # Threshold value
            nb_min = 8

            # Perform DAMEX clustering
            faces_dmx = dmx.damex(V, R, eps, nb_min)

            # Initialize clusters dictionary
            clusters = {}
            l = 1
            S = np.arange(len(faces_dmx))

            # Cluster merging loop
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

            # Assign predicted labels based on clustered data
            pred_labels_dmx = np.zeros(d)
            for key, values in clusters.items():
                pred_labels_dmx[values] = key

            # Compute adjusted Rand score and append to list
            ari_dmx.append(adjusted_rand_score(pred_labels_dmx, true_labels))

            # Modèle CLEF
            print('... CLEF Estimation ...')

            # Check if the dimension is within the limit for CLEF estimation
            if d <= 40:

                # Calculate the number of rows for CLEF
                R = int(np.sqrt(n))

                # Obtain faces using CLEF algorithm
                faces_clf = clf.clef(V, R, kappa_min=0.2)

                # Initialize clusters dictionary
                clusters = {}
                l = 1
                S = np.arange(len(faces_clf))

                # Cluster merging loop
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

                # Assign predicted labels based on clustered data
                pred_labels_clf = np.zeros(d)
                for key, values in clusters.items():
                    pred_labels_clf[values] = key

                # Compute adjusted Rand score and append to list
                ari_clf.append(adjusted_rand_score(
                    pred_labels_clf, true_labels))

        ari_dmx = pd.DataFrame(ari_dmx)
        ari_clf = pd.DataFrame(ari_clf)
        ari_muscle = pd.DataFrame(ari_muscle)
        ari_skmeans = pd.DataFrame(ari_skmeans)
        ari_eco_seco = pd.DataFrame(ari_eco_seco)

        print(ari_dmx)
        print(ari_muscle)
        print(ari_clf)
        print(ari_skmeans)
        print(ari_eco_seco)

        # pd.DataFrame.to_csv(ari_dmx, "results_model_4/results/model_4_" + str(int(d)) + "_" + str(int(k)) + "/ari_dmx" + ".csv")
        # pd.DataFrame.to_csv(ari_clf, "results_model_4/results/model_4_" + str(int(d)) + "_" + str(int(k)) + "/ari_clf" + ".csv")
        # pd.DataFrame.to_csv(ari_muscle, "results_model_4/results/model_4_" + str(int(d)) + "_" + str(int(k)) + "/ari_muscle" + ".csv")
        # pd.DataFrame.to_csv(ari_skmeans, "results_model_4/results/model_4_" + str(int(d)) + "_" + str(int(k)) + "/ari_skmeans" + ".csv")
        # pd.DataFrame.to_csv(ari_eco_seco, "results_model_4/results/model_4_" + str(int(d)) + "_" + str(int(k)) + "/ari_eco_seco" + ".csv")
