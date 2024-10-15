# Header:
# This script performs analysis and visualization of ARI (Adjusted Rand Index) 
# results for various clustering algorithms (ECO, DAMEX, CLEF, MUSCLE, SKMeans) 
# across different dimensions (d) and sparsity indices (s). It reads ARI results 
# from CSV files, computes the mean values, and generates a line plot comparing 
# the performance of the algorithms.

# Import necessary libraries
import numpy as np  # For numerical operations
from scipy.stats import pareto  # For statistical distributions
import pandas as pd  # For data manipulation and reading CSV files
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For enhanced data visualization
from plot_helpers import *  # Custom plotting helpers (not provided here)

# Use a custom plot style for consistency in visualizations
plt.style.use('qb-light.mplstyle')

# Define dimensions and sparsity indices
d = [20, 160]  # List of dimensions to analyze
k = [250]  # Placeholder variable (not used in this code)
sparsity_index = [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3]  # Sparsity levels

# Define markers and line styles for plotting
markers = ['o', '^', '*', 's', 'p']  # Different markers for algorithms
linestyles = ['solid', 'dashdot']  # Line styles for the plots

# Initialize an empty DataFrame to store ARI results
results = pd.DataFrame(columns=['ARI', 'Algorithm', 'd', 's'])

# Loop over each dimension (d)
for d_ in d:
    # Initialize lists to store mean ARI values for different algorithms
    ari_eco_mean = []
    ari_eco_bunea_mean = []
    ari_eco_seco_mean = []
    ari_dmx_mean = []
    ari_clf_mean = []
    ari_muscle_mean = []
    ari_skmeans_mean = []

    # Loop over each sparsity index (s)
    for s_ in sparsity_index:
        # Read ARI results from CSV files for different algorithms
        ari_eco = np.array(pd.read_csv(f"model_1_{int(d_)}_{int(s_)}/ari_eco.csv", index_col=0))
        ari_dmx = np.array(pd.read_csv(f"model_1_{int(d_)}_{int(s_)}/ari_dmx.csv", index_col=0))
        ari_skmeans = np.array(pd.read_csv(f"model_1_{int(d_)}_{int(s_)}/ari_skmeans.csv", index_col=0))
        ari_eco_bunea = np.array(pd.read_csv(f"model_1_{int(d_)}_{int(s_)}/ari_eco_bunea.csv", index_col=0))
        ari_eco_seco = np.array(pd.read_csv(f"model_1_{int(d_)}_{int(s_)}/ari_eco_seco.csv", index_col=0))
        
        # Append mean ARI values for the algorithms
        ari_eco_mean.append(np.mean(ari_eco))
        ari_dmx_mean.append(np.mean(ari_dmx))
        ari_skmeans_mean.append(np.mean(ari_skmeans))
        ari_eco_bunea_mean.append(np.mean(ari_eco_bunea))
        ari_eco_seco_mean.append(np.mean(ari_eco_seco))

        # Read ARI results for CLEF and MUSCLE algorithms
        ari_clf = np.array(pd.read_csv(f"model_1_{int(d_)}_{int(s_)}/ari_clf.csv", index_col=0))
        ari_clf_mean.append(np.mean(ari_clf))
        ari_muscle = np.array(pd.read_csv(f"model_1_{int(d_)}_{int(s_)}/ari_muscle.csv", index_col=0))
        ari_muscle_mean.append(np.mean(ari_muscle))

        # Create a DataFrame with ARI results for the current dimension and sparsity level
        d = {'ARI': [np.mean(ari_eco_seco), np.mean(ari_clf), np.mean(ari_dmx), np.mean(ari_muscle)], 
             'Algorithm': ['ECO', 'CLEF', 'DAMEX', 'MUSCLE'],  
             'd': [d_, d_, d_, d_], 
             's': [s_, s_, s_, s_]}
        inappend = pd.DataFrame(data=d)

        # Concatenate the new results to the main results DataFrame
        results = pd.concat([results, inappend])

# Reset the index of the results DataFrame
results = results.reset_index(drop=True)

# Print the results for verification
print(results)

# Create a plot to visualize the ARI values
fig, ax = plt.subplots()
# Generate a line plot for ARI results with styles and markers
sns.lineplot(data=results, x='s', y='ARI', hue='Algorithm', style='d',
             markers=['o', 's'], 
             palette=['#EF476F', '#06D6A0', '#118AB2', '#FFD166'], 
             lw=1, markersize=5, ax=ax)

# Display the plot
plt.show()

# Save the figure to a PDF file
fig.savefig('ARI_CLEF_DAMEX_ECO_MUSCLE_FLC.pdf')
