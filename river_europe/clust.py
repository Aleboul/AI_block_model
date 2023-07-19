"""
Clustering is performed based on an AI block model of the EFAS database in the European domain.

The threshold is calibrated according to the SECO value, resulting in the identification of two drops in this score metric.

Subsequently, we visualize the obtained matrix of extremal correlation. It reveals the presence of three clusters with strong
extremal dependence within each cluster, while exhibiting weak extremal correlation in the off-diagonal regions of the groups.
"""

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import sample

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import seaborn as sns
import geopandas as gpd
from shapely.geometry import Polygon
from matplotlib.patches import Rectangle

from itertools import islice
from shapely.ops import unary_union

plt.style.use('qb-light.mplstyle')


def ecdf(X):
    """ Compute uniform ECDF.

    Inputs
    ------
        X (np.array[float]) : array of observations

    Output
    ------
        Empirical uniform margin
    """

    index = np.argsort(X)
    ecdf = np.zeros(len(index))
    for i in index:
        ecdf[i] = (1.0 / (len(index))) * np.sum((X <= X[i]))
    return ecdf


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


def SECO(R, miss, clst):
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

    value = theta(R, miss)

    _value_ = []
    for key, c in clst.items():
        _R_2 = R[:, c]
        _miss = miss[:, c]
        _value_.append(theta(_R_2, miss))

    return np.sum(_value_) - value


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
    while len(S) > 0:
        l = l + 1
        if len(S) == 1:
            cluster[l] = np.array(S)
        else:
            a_l, b_l = find_max(Theta, S)
            if Theta[a_l, b_l] < alpha:
                cluster[l] = np.array([a_l])
            else:
                index_a = np.where(Theta[a_l, :] >= alpha)
                index_b = np.where(Theta[b_l, :] >= alpha)
                cluster[l] = np.intersect1d(S, index_a, index_b)
        S = np.setdiff1d(S, cluster[l])

    return cluster


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


january = pd.read_csv('data/january.csv')
february = pd.read_csv('data/february.csv')
mars = pd.read_csv('data/mars.csv')
april = pd.read_csv('data/april.csv')
may = pd.read_csv('data/may.csv')
july = pd.read_csv('data/july.csv')
june = pd.read_csv('data/june.csv')
august = pd.read_csv('data/aout.csv')
september = pd.read_csv('data/september.csv')
october = pd.read_csv('data/october.csv')
november = pd.read_csv('data/novembre.csv')
december = pd.read_csv('data/december.csv')
stations = pd.read_csv('data/stations_w.csv')

data = pd.concat([january, february, mars, april, may, june,
                 july, august, september, october, november, december])
data['Historical_times'] = pd.to_datetime(data['Historical_times'])
data = data.groupby(pd.Grouper(key="Historical_times",
                    freq="14D")).max()
data = data.dropna()
print(data.shape)
remove = ['4001', '4003', '4004', '4006', '4007', '4008',
          '4009', '4011', '4013', '4014', '4015', '4016',
          '1680', '1681', '1683', '1685', '1688', '1690',
          '2274', '2190', '2193', '2194', '2195']  # Remove Albania due to processing errors, all obtained time series are the same.
# And Israel, Island because too far apart
data = data.drop(remove, axis=1)
d = data.shape[1]
n = data.shape[0]
matrix = data.to_numpy()
print(n, d)

# Compute sampling version of the extremal correlation matrix

erank = np.array(data.rank() / (n+1))
outer = (np.maximum(erank[:, :, None], erank[:, None, :])).sum(0) / n

extcoeff = -np.divide(outer, outer-1)
Theta = np.maximum(2-extcoeff, 10e-5)

# Calibrate the threshold with the SECO
# Then plot

_tau_ = np.array(np.arange(0.1, 0.3, step=0.0025))
value_SECO = []
for tau in _tau_:
    print(tau)
    clusters = clust(Theta-10e-10, n=1000, alpha=tau)
    value = SECO(Theta, clusters)
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
fig.savefig('results/seco_tau.pdf')

# Use calibrated threshold

tau = 0.25
O_hat = clust(Theta, alpha=tau, n=n)
seco = SECO(Theta, O_hat)

index = []
for key, item in O_hat.items():
    shuffled = sorted(item, key=lambda k: random.random())
    index.append(shuffled)  # or item

index = np.hstack(index)

new_Theta = Theta[index, :][:, index]
sizes = np.zeros(len(O_hat)+1)

for key, item in O_hat.items():
    sizes[key] = len(item)

cusizes = np.cumsum(sizes) - 0.5

# Emphasized block extremal correlation matrix

fig, ax = plt.subplots()
im = plt.imshow(new_Theta, cmap="Blues_r")
for i in range(0, len(O_hat)):
    ax.add_patch(Rectangle((cusizes[i], cusizes[i]), sizes[i+1],
                 sizes[i+1], edgecolor='#323232', fill=False, lw=2))
plt.colorbar(im)
ax.grid(False)
fig.savefig('results/clust_matrix.pdf')

# Spatial representation of clusters

polys1 = gpd.GeoSeries(Polygon([(-16, 29), (43, 29), (43, 76), (-16, 76)]))
df1 = gpd.GeoDataFrame({'geometry': polys1}).set_crs('epsg:4326')
colors = ["steelblue", "lightblue", "darkorange", "chocolate", "darkseagreen", "limegreen", "darkslateblue", "royalblue", "yellow",
          "gold", "steelblue", "lightblue", "darkorange", "chocolate", "darkseagreen", "limegreen", "darkslateblue", "royalblue", "yellow", "gold"]
world = gpd.read_file("data/world-administrative-boundaries.geojson")
frankreich = gpd.overlay(world, df1, how='intersection')

O_hat = dict(sorted(O_hat.items(), key=lambda i: -len(i[1])))
print(O_hat)
print(np.array(O_hat))
qualitative_colors = sns.color_palette("Paired", 12)
plt.style.use('seaborn-whitegrid')
fig, ax = plt.subplots()
i = 0
for clst in [1, 2, 3]:
    cluster = stations.iloc[O_hat[clst], :]
    y = cluster['StationLat']
    x = cluster['StationLon']
    coordinate_ = np.c_[x, y]
    polygon = []
    for coord in coordinate_:
        polygon_geom = Polygon([((coord[0]-0.125), (coord[1]-0.125)), ((coord[0]+0.125), (coord[1]-0.125)),
                               ((coord[0]+0.125), (coord[1]+0.125)), ((coord[0]-0.125), (coord[1]+0.125))])
        polygon.append(polygon_geom)
        cu = unary_union(polygon)

    for geom in cu.geoms:
        xs, ys = geom.exterior.xy
        ax.fill(xs, ys, fc=qualitative_colors[i], ec='none')
        ax.plot(xs, ys, color=qualitative_colors[i], linewidth=0.75)
    i += 1
frankreich.boundary.plot(ax=ax, linewidth=0.5, color='black')
frankreich.plot(ax=ax, color='white', alpha=0.2)
ax.set_aspect(1.0)
plt.show()
fig.savefig('results/clust_1_3_together.pdf')
