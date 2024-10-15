# Header:
# This script evaluates the performance of several clustering algorithms (ECO, CLEF, DAMEX, MUSCLE, sKmeans)
# by computing the Adjusted Rand Index (ARI) for different dimensions (d) and sample sizes (k).
# The ARI values are read from CSV files, and the means are calculated for comparison.
# The results are visualized using Seaborn's lineplot, with different styles for dimensions and sample sizes.

# Import necessary libraries
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For enhanced visualizations
from plot_helpers import *  # Custom plotting helpers (not provided)
plt.style.use('qb-light.mplstyle')  # Set custom plot style

# Define dimensions (d) and sample sizes (k)
d = [40, 320]  # Two different dimensions to analyze
k = [100, 150, 200, 250, 300, 350, 400, 450, 500]  # Various sample sizes

# Define markers and line styles for different algorithms and dimensions
markers = ['o', '^', '*', 's', 'p']  # Different marker styles
linestyles = ['solid', 'dashdot']  # Different line styles

# Initialize a DataFrame to store the results
results = pd.DataFrame(columns=['ARI', 'Algorithm', 'd', 'n'])  # Columns for ARI, algorithm, dimension (d), and sample size (n)
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

    # Loop over sample sizes (k)
    for k_ in k:
        # Read ARI values from CSV files for each algorithm
        ari_eco = np.array(pd.read_csv(f"model_4_{int(d_)}_{int(k_)}/ari_eco.csv", index_col=0))
        ari_dmx = np.array(pd.read_csv(f"model_4_{int(d_)}_{int(k_)}/ari_dmx.csv", index_col=0))
        ari_skmeans = np.array(pd.read_csv(f"model_4_{int(d_)}_{int(k_)}/ari_skmeans.csv", index_col=0))
        ari_eco_bunea = np.array(pd.read_csv(f"model_4_{int(d_)}_{int(k_)}/ari_eco_bunea.csv", index_col=0))
        ari_eco_seco = np.array(pd.read_csv(f"model_4_{int(d_)}_{int(k_)}/ari_eco_seco.csv", index_col=0))

        # Calculate the mean ARI values for each algorithm and append to the respective lists
        ari_eco_mean.append(np.mean(ari_eco))
        ari_dmx_mean.append(np.mean(ari_dmx))
        ari_skmeans_mean.append(np.mean(ari_skmeans))
        ari_eco_bunea_mean.append(np.mean(ari_eco_bunea))
        ari_eco_seco_mean.append(np.mean(ari_eco_seco))

        # Read ARI values for CLEF and MUSCLE algorithms
        ari_clf = np.array(pd.read_csv(f"model_4_{int(d_)}_{int(k_)}/ari_clf.csv", index_col=0))
        ari_clf_mean.append(np.mean(ari_clf))
        ari_muscle = np.array(pd.read_csv(f"model_4_{int(d_)}_{int(k_)}/ari_muscle.csv", index_col=0))
        ari_muscle_mean.append(np.mean(ari_muscle))

        # Prepare the data to be appended, using mean ARI values for each algorithm
        d = {
            'ARI': [np.mean(ari_eco_seco), np.mean(ari_clf), np.mean(ari_dmx), np.mean(ari_muscle), np.mean(ari_skmeans)],
            'Algorithm': ['ECO', 'CLEF', 'DAMEX', 'MUSCLE', 'sKmeans'],  # List of algorithms
            'd': [d_, d_, d_, d_, d_],  # Dimension value repeated for each algorithm
            'n': [k_ * 20, k_ * 20, k_ * 20, k_ * 20, k_ * 20]  # Sample size (k) multiplied by 20 for each algorithm
        }

        # Convert the data dictionary into a DataFrame
        inappend = pd.DataFrame(data=d)
        print(inappend)  # Print the intermediate DataFrame

        # Append the current data to the results DataFrame
        results = pd.concat([results, inappend])
        print(results)  # Print updated results DataFrame

# Reset the index of the final results DataFrame
results = results.reset_index(drop=True)
print(results)  # Print the final results DataFrame

# Plotting the results
fig, ax = plt.subplots()  # Create a new figure and axis for the plot

# Line plot using Seaborn, with ARI on the y-axis and sample size (n) on the x-axis
sns.lineplot(
    data=results, x='n', y='ARI',
    hue='Algorithm',  # Different colors for each algorithm
    style='d',  # Different line styles for each dimension
    markers=['o', 's'],  # Marker shapes for each algorithm
    palette=['#EF476F', '#06D6A0', '#118AB2', '#FFD166', "#FFACBB"],  # Custom color palette
    lw=1, markersize=5, ax=ax  # Line width and marker size
)

# Display the plot
plt.show()

# Save the plot as a PDF file
fig.savefig('ARI_CLEF_DAMEX_ECO_MUSCLE_E5_F5.pdf')
