# Source the custom R script 'muscle.R' that contains functions necessary for clustering analysis
source('muscle.R')

# Load the required libraries for statistical analysis and spherical k-means clustering
library(MASS)       # Provides functions for statistical analysis
library(skmeans)    # Contains functions for spherical k-means clustering

# Read the dataset from a CSV file, constructing the file path dynamically
X = read.csv(paste('results_model_4/data/data/data_', 1, '.csv', sep=''))

# Remove the first column from the dataset, which is usually an identifier or index
X = X[,-1]

# Determine the number of features (dimensions) and observations (data points) in the dataset
d = ncol(X)  # Number of features (columns)
n = nrow(X)  # Number of observations (rows)

# Calculate the number of clusters (k) as the integer square root of the number of observations
k = as.integer(sqrt(n))

# Initialize a numeric vector to store the L2 norms for each observation
norms = numeric(n)

# Calculate the L2 norm (Euclidean norm) for each observation in the dataset
for (i in 1:n) {
    norms[i] = norm(as.matrix(X)[i,], type = "2")  # Compute the L2 norm for each row
}

# Set the number of clusters for the spherical k-means algorithm
nclusters = 5

# Proceed only if the specified number of clusters is less than the computed k
if (nclusters < k) {
    # Identify the indices of observations with the largest L2 norms
    indices = rank(norms) > n - k
    
    # Create a matrix of the observations corresponding to the selected indices
    norms_ext = as.matrix(X[indices,])
    
    # Perform spherical k-means clustering on the selected observations
    km <- skmeans(norms_ext, nclusters)
    
    # Convert the cluster centers to a data frame for easier manipulation
    centers = data.frame(km$prototypes)
    
    # Write the cluster centers to a CSV file for further analysis, without row names
    write.csv(centers, file=paste("results_model_4/results_skmeans/skmeans/centers_", 1, ".csv", sep=''), row.names=F)
}

# Transpose the data matrix so that features are rows and observations are columns
X = t(X)

# Create a sequence of proportions ranging from 0.01 to 0.15 in increments of 0.005
prop <- seq(0.01, 0.15, by = 0.005)

# Call the muscle_clusters function to perform clustering and obtain directions for the transposed data
directions <- muscle_clusters(X, prop)

# Extract the membership matrix from the clustering results, excluding the last row
M_emp <- as.matrix(directions[[1]][-(d + 1), ])

# Convert the membership matrix to a data frame for easier manipulation
M_emp = data.frame(M_emp)

# Write the membership matrix to a CSV file for later analysis, without row names
write.csv(M_emp, file=paste("results_model_4/results_muscle/muscle/Memp_", 1, ".csv", sep=''), row.names=F)

