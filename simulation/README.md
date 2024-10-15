# Supplementary code for "High dimensional variable clustering based on maxima of a weakly dependent random process"

This repository contains implementation of the ECO algorithm described in "High dimensional variable clustering based on maxima of a weakly dependent random process" and codes for generating all figures and table therein. 

## File descriptions:

* `model_i_parmixing_m.py`, this script performs a Monte Carlo simulation to evaluate the accuracy of clustering using the ECO algorithm for framework F3. The simulation is conducted across various block length, with each iteration generating synthetic data, performing clustering, and measuring the exact recovery percentage of the clustering algorithm with i = 3,4,5 the corresponding experiments.
* `model_i_parmixing_k.py`, this script performs a Monte Carlo simulation to evaluate the accuracy of clustering using the ECO algorithm for framework F4. The simulation is conducted across various sample sizes, with each iteration generating synthetic data, performing clustering, and measuring the exact recovery percentage of the clustering algorithm with i = 3,4,5 the corresponding experiments.
* `model_i_parmixing_alpha.py`, this script uses R and Python to generate synthetic data based on copula models and performs clustering evaluation through Monte Carlo simulations for framework F5. It computes the percentage of exact cluster recovery and SECO (Sum of Extremal COefficients) criterion with i = 3,4,5 the corresponding experiments.
* `model_i_parmixing_m.R`, this R code implements a series of functions to generate a copula-based random time series and extract block maxima from the data with i = 3,4,5 the corresponding experiments.
* `eco_alg.py`, this Python module provides functions for statistical analysis and clustering based on extremal value theory, including calculations of the Empirical Cumulative Distribution Function (ECDF), w-madogram, and clustering algorithms that identify clusters using extremal correlation matrices.
* `output`: contains csv files of results and script to produce Figures.

## Dependencies:
The following packages are required to run the code:

- Python packages:
  - **numpy** (version >= 1.24.2)
  - **multiprocessing** (version >= 3.13.0)
  - **pandas** (version >= 1.5.3)
  - **matplotlib** (version >= 3.7.1)
  - **rpy2** (version >= 3.5.13)

- R packages:
  - **copula** (version >= 1.1.2)
  - **stats** (version >= 4.1.2)