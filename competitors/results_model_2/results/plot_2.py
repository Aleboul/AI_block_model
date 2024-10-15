# Header:
# This script performs clustering performance analysis by calculating the Adjusted Rand Index (ARI) 
# for various algorithms (ECO, CLEF, DAMEX, MUSCLE, sKmeans) across different dimensions (d) 
# and sample sizes (k). It reads ARI values from CSV files, computes the mean ARI for each algorithm, 
# and then visualizes the results using Seaborn's lineplot. 

# Import necessary libraries
import numpy as np  # For numerical operations
from scipy.stats import pareto  # Pareto distribution (unused in this script but imported)
import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For plot creation
import seaborn as sns  # For enhanced visualization
from plot_helpers import *  # Custom plotting functions (not provided in the script)
plt.style.use('qb-light.mplstyle')  # Use custom plot style

# Define dimensions (d) and sample sizes (k) to analyze
d = [40, 320]  # List of different dimensions
k = [100, 150, 200, 250, 300, 350, 400, 450, 500]  # List of different sample sizes

# Define marker styles and line styles for plotting
markers = ['o', '^', '*', 's', 'p']  # Different markers for different algorithms
linestyles = ['solid', 'dashdot']  # Different line styles for dimensions

# Initialize variables
lt = 0  # Line type index (used for alternating styles)
results = pd.DataFrame(columns=['ARI', 'Algorithm', 'd', 'k'])  # Empty DataFrame to store results
print(results)  # Print initial empty DataFrame

# Main loop over different dimensions (d)
for d_ in d:
    l = 0  # Line style index
    # Lists to store mean ARI values for each algorithm across different sample sizes
    ari_eco_mean = []
    ari_eco_bunea_mean = []
    ari_eco_seco_mean = []
    ari_dmx_mean = []
    ari_clf_mean = []
    ari_muscle_mean = []
    ari_skmeans_mean = []

    # Loop over different sample sizes (k)
    for k_ in k:
        # Read ARI values from CSV files for each algorithm
        ari_eco = np.array(pd.read_csv(f"model_2_{int(d_)}_{int(k_)}/ari_eco.csv", index_col=0))
        ari_dmx = np.array(pd.read_csv(f"model_2_{int(d_)}_{int(k_)}/ari_dmx.csv", index_col=0))
        ari_skmeans = np.array(pd.read_csv(f"model_2_{int(d_)}_{int(k_)}/ari_skmeans.csv", index_col=0))
        ari_eco_bunea = np.array(pd.read_csv(f"model_2_{int(d_)}_{int(k_)}/ari_eco_bunea.csv", index_col=0))
        ari_eco_seco = np.array(pd.read_csv(f"model_2_{int(d_)}_{int(k_)}/ari_eco_seco.csv", index_col=0))

        # Append the mean ARI values for each algorithm to their respective lists
        ari_eco_mean.append(np.mean(ari_eco))
        ari_dmx_mean.append(np.mean(ari_dmx))
        ari_skmeans_mean.append(np.mean(ari_skmeans))
        ari_eco_bunea_mean.append(np.mean(ari_eco_bunea))
        ari_eco_seco_mean.append(np.mean(ari_eco_seco))

        # Read ARI values for CLEF and MUSCLE algorithms
        ari_clf = np.array(pd.read_csv(f"model_2_{int(d_)}_{int(k_)}/ari_clf.csv", index_col=0))
        ari_clf_mean.append(np.mean(ari_clf))
        ari_muscle = np.array(pd.read_csv(f"model_2_{int(d_)}_{int(k_)}/ari_muscle.csv", index_col=0))
        ari_muscle_mean.append(np.mean(ari_muscle))

        # Create a dictionary with mean ARI values, algorithm names, dimensions (d), and sample sizes (n)
        d = {'ARI': [np.mean(ari_eco_seco), np.mean(ari_clf), np.mean(ari_dmx), np.mean(ari_muscle), np.mean(ari_skmeans)], 
             'Algorithm': ['ECO', 'CLEF', 'DAMEX', 'MUSCLE', 'sKmeans'],  
             'd': [d_, d_, d_, d_, d_], 
             'n': [20 * k_, 20 * k_, 20 * k_, 20 * k_, 20 * k_]}  # 'n' is sample size multiplied by 20

        inappend = pd.DataFrame(data=d)  # Convert dictionary to DataFrame
        print(inappend)  # Print DataFrame to check values
        results = pd.concat([results, inappend])  # Append results to the main DataFrame
        print(results)  # Print updated results DataFrame

# Reset the index of the final results DataFrame
results = results.reset_index(drop=True)
print(results)  # Print final DataFrame

# Plotting the results
fig, ax = plt.subplots()  # Create a new figure and axis
# Line plot with Seaborn, using ARI as the y-axis, sample size (n) as x-axis, and distinguishing 
# algorithms by hue and dimensions by style
sns.lineplot(
    data=results, x='n', y='ARI',
    hue='Algorithm',  # Different lines for each algorithm
    style='d',  # Different line styles for each dimension
    markers=['o', 's'],  # Different markers for different algorithms
    palette=['#EF476F', '#06D6A0', '#118AB2', '#FFD166', "#FFACBB"],  # Custom color palette
    lw=1, markersize=5, ax=ax  # Line width and marker size
)

# Show the plot
plt.show()

# Save the figure to a PDF file
fig.savefig('ARI_CLEF_DAMEX_ECO_MUSCLE_MSC.pdf')
