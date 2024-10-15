# Header:
# This script analyzes the clustering performance of different algorithms (ECO, CLEF, DAMEX, MUSCLE)
# across different dimensions (d) and sparsity levels (s), calculating the Adjusted Rand Index (ARI).
# The ARI scores are read from CSV files, and their means are calculated for comparison.
# The results are then visualized using Seaborn's lineplot.

# Import necessary libraries
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For plot creation
import seaborn as sns  # For enhanced visualization
from plot_helpers import *  # Custom plotting functions (not provided)
plt.style.use('qb-light.mplstyle')  # Set the plot style

# Define dimensions (d) and sparsity indices to analyze
d = [20, 160]  # List of different dimensions to analyze
k = [250]  # Sample size list (only 250 in this case)
sparsity_index = [20, 19, 18, 17, 16, 15,  # Different sparsity levels
                  14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3]

# Define markers and line styles for plotting
markers = ['o', '^', '*', 's', 'p']  # Markers for different algorithms
linestyles = ['solid', 'dashdot']  # Line styles for different dimensions

# Initialize the results DataFrame
results = pd.DataFrame(columns=['ARI', 'Algorithm', 'd', 's'])  # DataFrame to store results
print(results)  # Print the initial empty DataFrame

# Main loop over dimensions (d)
for d_ in d:
    # Initialize lists to store mean ARI values for each algorithm
    ari_eco_mean = []
    ari_eco_bunea_mean = []
    ari_eco_seco_mean = []
    ari_dmx_mean = []
    ari_clf_mean = []
    ari_muscle_mean = []
    ari_skmeans_mean = []

    # Loop over sparsity levels (s)
    for s_ in sparsity_index:
        # Read ARI values from CSV files for each algorithm
        ari_eco = np.array(pd.read_csv(f"model_3_{int(d_)}_{int(s_)}/ari_eco.csv", index_col=0))
        ari_dmx = np.array(pd.read_csv(f"model_3_{int(d_)}_{int(s_)}/ari_dmx.csv", index_col=0))
        ari_skmeans = np.array(pd.read_csv(f"model_3_{int(d_)}_{int(s_)}/ari_skmeans.csv", index_col=0))
        ari_eco_bunea = np.array(pd.read_csv(f"model_3_{int(d_)}_{int(s_)}/ari_eco_bunea.csv", index_col=0))
        ari_eco_seco = np.array(pd.read_csv(f"model_3_{int(d_)}_{int(s_)}/ari_eco_seco.csv", index_col=0))

        # Calculate the mean ARI values for each algorithm and append to the lists
        ari_eco_mean.append(np.mean(ari_eco))
        ari_dmx_mean.append(np.mean(ari_dmx))
        ari_skmeans_mean.append(np.mean(ari_skmeans))
        ari_eco_bunea_mean.append(np.mean(ari_eco_bunea))
        ari_eco_seco_mean.append(np.mean(ari_eco_seco))

        # Read ARI values for CLEF and MUSCLE algorithms
        ari_clf = np.array(pd.read_csv(f"model_3_{int(d_)}_{int(s_)}/ari_clf.csv", index_col=0))
        ari_clf_mean.append(np.mean(ari_clf))
        ari_muscle = np.array(pd.read_csv(f"model_3_{int(d_)}_{int(s_)}/ari_muscle.csv", index_col=0))
        ari_muscle_mean.append(np.mean(ari_muscle))

        # Create a dictionary to store the mean ARI values, algorithm names, dimensions (d), and sparsity (s)
        d = {
            'ARI': [np.mean(ari_eco_seco), np.mean(ari_clf), np.mean(ari_dmx), np.mean(ari_muscle)],
            'Algorithm': ['ECO', 'CLEF', 'DAMEX', 'MUSCLE'],  # Algorithm names
            'd': [d_, d_, d_, d_],  # Dimension repeated for each algorithm
            's': [s_, s_, s_, s_]  # Sparsity level repeated for each algorithm
        }

        # Convert the dictionary into a DataFrame and append it to the main results DataFrame
        inappend = pd.DataFrame(data=d)
        print(inappend)  # Print the intermediate DataFrame for checking
        results = pd.concat([results, inappend])  # Append to the results DataFrame
        print(results)  # Print updated results DataFrame

# Reset the index of the results DataFrame after appending all results
results = results.reset_index(drop=True)
print(results)  # Print the final results DataFrame

# Plotting the results
fig, ax = plt.subplots()  # Create a new figure and axis

# Line plot using Seaborn, with ARI on the y-axis, sparsity index (s) on the x-axis, and different
# algorithms distinguished by hue and dimensions by line style
sns.lineplot(
    data=results, x='s', y='ARI',
    hue='Algorithm',  # Different colors for each algorithm
    style='d',  # Different line styles for each dimension
    markers=['o', 's'],  # Marker shapes for each algorithm
    palette=['#EF476F', '#06D6A0', '#118AB2', '#FFD166'],  # Custom color palette
    lw=1, markersize=5, ax=ax  # Line width and marker size
)

# Show the plot
plt.show()

# Save the plot as a PDF file
fig.savefig('ARI_CLEF_DAMEX_ECO_MUSCLE_MSC.pdf')
