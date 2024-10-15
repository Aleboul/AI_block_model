# Supplementary code for "High dimensional variable clustering based on maxima of a weakly dependent random process"

This repository contains CSV files with the results for various dimensions and levels of sparsity for Experiment E1 and Framework F2.

## File descriptions:

* `model_3_d_s.py`: Contains CSV files with the results for Experiment E2 and Framework F2, where d represents the dimension and s the sparsity degree.
* `plot_2.py`: This script performs analysis and visualization of ARI (Adjusted Rand Index) results for various clustering algorithms (ECO, DAMEX, CLEF, MUSCLE, SKMeans) across different dimensions (d) and sparsity indices (s). It reads ARI results from CSV files, computes the mean values, and generates a line plot comparing the performance of the algorithms.
* `plot_helper.py`:  This script contains various utility functions and plotting routines for statistical analysis. It includes functions for plotting trajectories, calculating statistical intervals, hazard rates from run lengths, and fitting a truncated Laplace distribution. Additionally, there are helper functions to configure plot styles, set marker properties, and toggle LaTeX support in Matplotlib.