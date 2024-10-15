# Load required R scripts and libraries
source('muscle.R')  # Source the custom R script 'muscle.R' which contains relevant functions
library(MASS)       # Load the MASS library for statistical functions

# Read the CSV file containing data for the model
X = read.csv(paste('results_model_1/data/data/data_', 1, '.csv', sep=''))

# Remove the first column from the dataset, typically an index or identifier
X = X[,-1]

# Determine the number of features (dimensions) and observations (data points)
d = ncol(X)  # Number of columns (features) in the dataset
n = nrow(X)  # Number of rows (observations) in the dataset

# Set the number of clusters (k) as the square root of the number of observations, rounded to the nearest integer
k = as.integer(sqrt(n))

# Check if the number of observations is greater than 1000 before proceeding with clustering
if(n > 1000) {
    # Transpose the data matrix so that features are rows and observations are columns
    X = t(X)
    
    # Create a sequence of proportions to be used in the clustering process
    prop <- seq(0.01, 0.15, by = 0.005)
    
    # Call the muscle_clusters function to perform clustering and get cluster directions
    directions <- muscle_clusters(X, prop)
    
    # Extract the membership matrix (excluding the last row) from the clustering results
    M_emp <- as.matrix(directions[[1]][-(d + 1), ])
    
    # Convert the membership matrix to a data frame for easier manipulation
    M_emp = data.frame(M_emp)
    
    # Write the membership matrix to a CSV file for later analysis, without row names
    write.csv(M_emp, file=paste("results_model_1/results_muscle/muscle/Memp_", 1, ".csv", sep=''), row.names=F)
}
