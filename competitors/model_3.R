# Source the custom R script 'muscle.R' that contains relevant functions for clustering analysis
source('muscle.R')

# Load the required libraries for statistical methods and clustering
library(MASS)       # Provides functions for statistical analysis and modeling

# Read the dataset from a CSV file, using a dynamically constructed file path
X = read.csv(paste('results_model_3/data/data/data_', 1, '.csv', sep=''))

# Remove the first column from the dataset, typically an identifier or index
X = X[,-1]

# Determine the number of features (dimensions) and observations (data points) in the dataset
d = ncol(X)  # Number of columns (features)
n = nrow(X)  # Number of rows (observations)

# Calculate the number of clusters (k) as the integer square root of the number of observations
k = as.integer(sqrt(n))

# Set the number of clusters based on the number of features, dividing by 4
nclusters = as.integer(d / 4)

# Check if the number of observations exceeds 1000 to proceed with further analysis
if (n > 1000) {
    # Transpose the data matrix so that features are rows and observations are columns
    X = t(X)
    
    # Create a sequence of proportions ranging from 0.01 to 0.15 in increments of 0.005
    prop <- seq(0.01, 0.15, by = 0.005)
    
    # Call the muscle_clusters function to perform clustering and obtain directions for the data
    directions <- muscle_clusters(X, prop)
    
    # Extract the membership matrix from the clustering results, excluding the last row
    M_emp <- as.matrix(directions[[1]][-(d + 1), ])
    
    # Convert the membership matrix to a data frame for easier manipulation
    M_emp = data.frame(M_emp)
    
    # Write the membership matrix to a CSV file for later analysis, without row names
    write.csv(M_emp, file=paste("results_model_3/results_muscle/muscle/Memp_", 1, ".csv", sep=''), row.names=F)
}
