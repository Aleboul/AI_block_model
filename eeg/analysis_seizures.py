"""
Clustering and Visualization of EEG Data Using Extremal Correlation Analysis

This script performs clustering and visualization on EEG data from the CHB-MIT Scalp EEG database 
for patient 05 during seizure events, leveraging an AI-based extremal correlation block model. The model 
identifies clusters of dependent EEG channels by calibrating the clustering threshold using SECO 
(Structural Extremal Coefficient Objective) to avoid trivial solutions.

Key steps include:
1. Loading and preprocessing EEG data.
2. Computing extremal correlation and the chi matrix.
3. Calibrating clustering thresholds based on SECO values.
4. Visualizing the extremal correlation matrix and its block structure.
5. Providing spatial representation of EEG clusters on 2D scalp maps.

The outputs include time-series plots, SECO calibration results, extremal correlation block structures, 
and spatial representations, saved as PDF files.

Dependencies:
- numpy, pandas, matplotlib, matplotlib.patheffects, mne, eco_alg
"""


import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patheffects as mpe
import mne  # For EEG and MEG data handling
import eco_alg  # For algorithmic analysis

# Setting a specific style for the matplotlib plots
plt.style.use('qb-light.mplstyle')

# Load EEG data from a CSV file
data_eeg = pd.read_csv('data/eeg_seizures.csv', sep=',')

# Create a figure and axis object for plotting
fig, ax = plt.subplots()

# Plot the entire EEG data on the created axis
ax.plot(data_eeg)

# Display the plot
plt.show()

# Save the plot to a PDF file in the results/seizures directory
fig.savefig('results/seizures/ts.pdf')

# Remove the columns 'P7-T7' and 'T8-P8-1' from the EEG dataset as they might be irrelevant or noisy
data_eeg.drop(columns=['P7-T7', 'T8-P8-1'], inplace=True)

# Store the number of features (columns) in 'd' and the number of observations (rows) in 'n'
d = data_eeg.shape[1]
n = data_eeg.shape[0]

# Compute sampling version of the extremal correlation matrix

# Step 1: Rank the EEG data
# erank is an array that stores the rank of each value in the EEG dataset, normalized by dividing by (n+1).
erank = np.array(data_eeg.rank() / (n+1))

# Step 2: Compute pairwise outer ranks for the extremal correlation matrix
# The outer product is calculated to find the maximum rank between pairs of observations, which is summed across samples.
outer = (np.maximum(erank[:, :, None], erank[:, None, :])).sum(0) / n

# Step 3: Compute the extremal coefficient matrix
# The extremal coefficient is derived from the outer rank matrix using the formula -outer / (outer - 1).
extcoeff = -np.divide(outer, outer-1)

# Step 4: Compute the chi matrix, which is related to the extremal dependence between variables.
# A small value (10e-5) is added to avoid very small values.
chi = np.maximum(2-extcoeff, 10e-5)

# Calibrate the threshold with the SECO (Structural Extremal Coefficient Objective)
# Step 5: Create a range of threshold values (_tau_) between 0.1 and 0.5, with a step size of 0.0025.
_tau_ = np.array(np.arange(0.1, 0.5, step=0.0025))

# Initialize an empty list to store the SECO values for each threshold.
value_SECO = []

# Step 6: Iterate over each threshold tau to evaluate clustering and SECO
for tau in _tau_:
    print(tau)  # Print the current threshold value for monitoring.

    # Perform clustering using the eco_alg.clust function on the chi matrix.
    # chi-10e-10 ensures numerical stability when clustering.
    clusters = eco_alg.clust(chi, n=n, alpha=tau)

    # Compute the SECO value for the current clustering solution.
    value = eco_alg.SECO(erank, clusters)

    # Append the computed SECO value to the value_SECO list.
    value_SECO.append(value)

# Step 7: Convert the list of SECO values to a numpy array.
value_SECO = np.array(value_SECO)

# Step 8: Apply a log transformation to the SECO values for better visualization.
# The transformation ensures all values are positive and enhances differences between values.
value_SECO = np.log(1 + value_SECO - np.min(value_SECO))

# Step 9: Find the index of the minimum SECO value (optimal threshold).
ind = np.argmin(value_SECO)

# Step 10: Plot the SECO values against the corresponding tau thresholds.
fig, ax = plt.subplots()

# Plot the SECO values as a function of the threshold tau.
ax.plot(_tau_, value_SECO, marker='o', linestyle='solid',
        markerfacecolor='white', lw=1, markersize=2)

# Set the y-axis label to 'SECO'.
ax.set_ylabel('SECO')

# Set the x-axis label to the threshold value (τ).
ax.set_xlabel(r'Treshold $\tau$')

# Display the plot.
plt.show()

# Save the plot as a PDF file in the 'results/seizures/' directory.
fig.savefig('results/seizures/seco_tau.pdf')

# Use the calibrated threshold

# Step 1: Perform clustering using the optimal threshold from the SECO minimization.
# 'alpha' parameter is set to the optimal value found previously (_tau_[ind]).
O_hat = eco_alg.clust(chi, n=n, alpha=_tau_[ind])

# Step 2: Reorganize the indices for plotting
# Initialize an empty list to store the reordered indices of clusters.
index = []

# Step 3: Randomly shuffle the items within each cluster and append to the index list.
# This is done to ensure that the visualization isn't biased by the ordering of elements within clusters.
for key, item in O_hat.items():
    # Shuffle the items in each cluster randomly.
    shuffled = sorted(item, key=lambda k: random.random())
    index.append(shuffled)  # Append the shuffled indices to the index list.

# Step 4: Flatten the index list into a single array.
index = np.hstack(index)

# Emphasized block extremal correlation matrix

# Step 5: Reorder the chi matrix based on the new index to emphasize block structure.
new_Theta = chi[index, :][:, index]

# Step 6: Compute the sizes of each cluster.
# 'sizes' array holds the size of each cluster. len(O_hat) + 1 accounts for all the clusters.
sizes = np.zeros(len(O_hat)+1)

# Step 7: Populate the 'sizes' array with the sizes of each cluster.
for key, item in O_hat.items():
    sizes[key] = len(item)

# Step 8: Compute the cumulative sum of sizes, which will be used for block positioning in the visualization.
# Subtracting 0.5 for proper rectangle alignment.
cusizes = np.cumsum(sizes) - 0.5

# Step 9: Create the plot for the block extremal correlation matrix.
fig, ax = plt.subplots()
ax.grid(False)  # Disable the grid for a cleaner plot.

# Step 10: Display the reordered extremal correlation matrix using the 'Blues_r' colormap.
im = plt.imshow(new_Theta, cmap="Blues_r", vmin=0.0)

# Step 11: Add rectangles around each cluster to emphasize the block structure.
for i in range(0, len(O_hat)):
    ax.add_patch(Rectangle((cusizes[i], cusizes[i]), sizes[i+1],
                 sizes[i+1], edgecolor='#323232', fill=False, lw=2))  # Dark borders around blocks.

# Step 12: Add a color bar to the plot.
plt.colorbar(im)

# Step 13: Show the plot.
plt.show()

# Step 14: Save the plot as a PDF file in the 'results/seizures/' directory.
fig.savefig('results/seizures/eco_mat_emphase.pdf')

# Spatial representation of clusters

# Step 1: Rename columns in the EEG dataset for consistency in channel naming.
data_eeg = data_eeg.rename(columns={'# FP1-F7': 'Fp1-F7', 'FP1-F3': 'Fp1-F3',
                           'FP2-F4': 'Fp2-F4', 'FP2-F8': 'Fp2-F8', 'T8-P8-0': 'T8-P8',
                                    'FZ-CZ': 'Fz-Cz', 'CZ-PZ': 'Cz-Pz'})

# Step 2: Print the clustering results
print(O_hat)

# Step 3: Load a predefined layout for the EEG channel positions (e.g., "EEG1005" system).
layout = mne.channels.read_layout("EEG1005")

# Step 4: Define a selection of the channels to be plotted. These channels represent specific regions of the scalp.
selection = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz", "FT9", "FT10"
]

# Step 5: Extract the indices of the selected channels from the layout.
picks = []
for channel in selection:
    # Find the index of each channel in the layout.
    picks.append(layout.names.index(channel))

# Step 6: Extract the names and coordinates of the selected channels.
names = np.array(layout.names)[picks]  # Channel names
coord = layout.pos[picks][:, [0, 1]]   # Coordinates (X, Y) for each channel

# Step 7: Create a plot for visualizing the clusters on a 2D scalp map.
fig, ax = plt.subplots()

# Step 8: Plot each channel's name at its corresponding position on the scalp map.
for i in range(len(coord)):
    # Display the channel name at the (x, y) coordinates.
    plt.text(coord[i, 0], coord[i, 1], names[i])

# Step 9: Define the colors for different clusters and a path effect (to emphasize the cluster edges).
# Define colors for each cluster.
colors = ['steelblue', 'salmon', 'limegreen']
pe1 = [mpe.Stroke(linewidth=7, foreground='black'),  # Stroke effect with a black border.
       # White foreground stroke for clarity.
       mpe.Stroke(foreground='white', alpha=1),
       mpe.Normal()]                                # Normal path effect for the text.

# Step 10: Iterate over each cluster and plot the connections between electrodes within the same cluster.
for i in range(len(O_hat)):
    # Get the EEG channels belonging to the current cluster.
    clusters = data_eeg.columns[O_hat[i+1]]
    for j in range(len(clusters)):
        channel = clusters[j]
        print(channel)  # Print the channel name for debugging purposes.
        # Split the channel name into two parts (e.g., 'Fp1-F7' becomes ['Fp1', 'F7']).
        channel = channel.rsplit('-', 1)
        print(channel)  # Print the split channel for verification.

        # Step 11: Locate the indices of both electrodes that form the current channel pair.
        # Find the index of the first electrode in the 'names' array.
        index_0 = np.where(names == channel[0])
        # Find the index of the second electrode in the 'names' array.
        index_1 = np.where(names == channel[1])

        # Step 12: Extract the coordinates (x, y) for both electrodes to draw a line connecting them.
        # X coordinates for the line between the two electrodes.
        x = np.c_[coord[index_0, 0], coord[index_1, 0]][0]
        # Y coordinates for the line between the two electrodes.
        y = np.c_[coord[index_0, 1], coord[index_1, 1]][0]

        # Step 13: Plot the line representing the connection between the two electrodes within the cluster.
        # Use a thick colored line with path effects.
        ax.plot(x, y, '-', color=colors[i], linewidth=5, path_effects=pe1)

# Step 14: Set the aspect ratio for the plot and remove the axis (for cleaner visualization).
ax.set_aspect(1.25)
ax.axis('off')  # Turn off the axis for a cleaner plot.

# Step 15: Display the plot.
plt.show()

# Step 16: Save the plot as a PDF file in the 'results/seizures/' directory.
fig.savefig('results/seizures/spatial_clust.pdf')
