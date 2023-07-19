"""
Performs clustering according to an AI block model of the CHB-MIT Scalp EEG database for patient 05.

The analysis is performed where no seizures are observed.

We calibrate the threshold according to the SECO value. We find two drops for this score metric. We
kept the largest threshold for which the drop is observed (0.4) due to better results.

We hence plot the matrix of extremal correlation obtained. Three clusters are noticed with strong
extremal dependence inside clusters while weak extremal correlation in the off-diagonal of the
groups.
"""

import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patheffects as mpe
import mne

plt.style.use('qb-light.mplstyle')


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


def find_max(M, S):
    mask = np.zeros(M.shape, dtype=bool)
    values = np.ones((len(S), len(S)), dtype=bool)
    mask[np.ix_(S, S)] = values
    np.fill_diagonal(mask, 0)
    max_value = M[mask].max()
    # Sometimes doublon happens for excluded clusters, if n is low
    i, j = np.where(np.multiply(M, mask * 1) == max_value)
    return i[0], j[0]


def clust(Theta, n, alpha=None):
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

    if alpha is None:
        alpha = 2 * np.sqrt(np.log(d)/n)

    cluster = {}
    max_index = {}
    while len(S) > 0:
        l = l + 1
        if len(S) == 1:
            cluster[l] = np.array(S)
            max_index[l] = int(S)
        else:
            a_l, b_l = find_max(Theta, S)
            if Theta[a_l, b_l] < alpha:
                cluster[l] = np.array([a_l])
                max_index[l] = int(a_l)
            else:
                index_a = np.where(Theta[a_l, :] >= alpha)
                index_b = np.where(Theta[b_l, :] >= alpha)
                cluster[l] = np.intersect1d(S, index_a, index_b)
                max_index[l] = int(a_l)
        S = np.setdiff1d(S, cluster[l])

    return cluster  # , max_index


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

# Load data


data_eeg = pd.read_csv('data/eeg_wo_seizures.csv', sep=',')
fig, ax = plt.subplots()
ax.plot(data_eeg)
plt.show()
fig.savefig('results/wo_seizures/ts.pdf')
data_eeg.drop(columns=['P7-T7', 'T8-P8-1'], inplace=True)

d = data_eeg.shape[1]
n = data_eeg.shape[0]

# Compute sampling version of the extremal correlation matrix

erank = np.array(data_eeg.rank() / (n+1))
print(erank.shape)
outer = (np.maximum(erank[:, :, None], erank[:, None, :])).sum(0) / n

extcoeff = -np.divide(outer, outer-1)
chi = 2-extcoeff

# Calibrate the threshold with the SECO
# Then plot

_tau_ = np.array(np.arange(0.1, 0.5, step=0.0025))
value_SECO = []
for tau in _tau_:
    print(tau)
    clusters = clust(chi-10e-10, n=1000, alpha=tau)
    value = SECO(erank, clusters)
    value_SECO.append(value)
value_SECO = np.array(value_SECO)
value_SECO = np.log(1+value_SECO - np.min(value_SECO))
ind = np.argmin(value_SECO)
fig, ax = plt.subplots()
ax.plot(_tau_, value_SECO, marker='o', linestyle='solid',
        markerfacecolor='white', lw=1, markersize=2)
ax.set_ylabel('SECO')
ax.set_xlabel(r'Treshold $\tau$')
plt.show()
fig.savefig('results/wo_seizures/seco_tau.pdf')

# Use calibrated threshold

O_hat = clust(chi-10e-10, n=1000, alpha=0.4)

a = {'hello': 'world'}

with open('O_hat.pickle', 'wb') as o_hat:
    pickle.dump(O_hat, o_hat, protocol=pickle.HIGHEST_PROTOCOL)
index = []
for key, item in O_hat.items():
    shuffled = sorted(item, key=lambda k: random.random())
    index.append(shuffled)  # or item

index = np.hstack(index)

# Emphasized block extremal correlation matrix

new_Theta = chi[index, :][:, index]
sizes = np.zeros(len(O_hat)+1)

for key, item in O_hat.items():
    sizes[key] = len(item)

cusizes = np.cumsum(sizes) - 0.5


fig, ax = plt.subplots()
ax.grid(False)
im = plt.imshow(new_Theta, cmap="Blues_r", vmin=0.0)
for i in range(0, len(O_hat)):
    ax.add_patch(Rectangle((cusizes[i], cusizes[i]), sizes[i+1],
                 sizes[i+1], edgecolor='#323232', fill=False, lw=2))
plt.colorbar(im)
plt.show()
fig.savefig('results/wo_seizures/eco_mat_emphase.pdf')

# Spatial representation of clusters

data_eeg = data_eeg.rename(columns={'# FP1-F7': 'Fp1-F7', 'FP1-F3': 'Fp1-F3',
                           'FP2-F4': 'Fp2-F4', 'FP2-F8': 'Fp2-F8', 'T8-P8-0': 'T8-P8', 'FZ-CZ': 'Fz-Cz', 'CZ-PZ': 'Cz-Pz'})


layout = mne.channels.read_layout("EEG1005")
selection = [
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T7",
    "T8",
    "P7",
    "P8",
    "Fz",
    "Cz",
    "Pz",
    "FT9",
    "FT10"
]
picks = []
for channel in selection:
    picks.append(layout.names.index(channel))
# display = layout.plot(picks=picks)

names = np.array(layout.names)[picks]
coord = layout.pos[picks][:, [0, 1]]
fig, ax = plt.subplots()
for i in range(len(coord)):
    plt.text(coord[i, 0], coord[i, 1], names[i])
colors = ['steelblue', 'salmon', 'limegreen']
pe1 = [mpe.Stroke(linewidth=7, foreground='black'),
       mpe.Stroke(foreground='white', alpha=1),
       mpe.Normal()]
for i in range(len(O_hat)):
    clust = data_eeg.columns[O_hat[i+1]]
    for j in range(len(clust)):
        channel = clust[j]
        print(channel)
        channel = channel.rsplit('-', 1)
        print(channel)
        index_0 = np.where(names == channel[0])
        index_1 = np.where(names == channel[1])
        x = np.c_[coord[index_0, 0], coord[index_1, 0]][0]
        y = np.c_[coord[index_0, 1], coord[index_1, 1]][0]
        ax.plot(x, y, '-', color=colors[i], linewidth=5, path_effects=pe1)
ax.set_aspect(1.25)
ax.axis('off')
plt.show()
fig.savefig('results/wo_seizures/spatial_clust.pdf')
