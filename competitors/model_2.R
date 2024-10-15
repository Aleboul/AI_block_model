# Source the custom R script 'muscle.R' that contains relevant functions for clustering analysis
source('muscle.R')

# Load required libraries for statistical analysis and clustering
library(MASS)       # Contains functions for statistical methods and models
library(skmeans)    # Provides functions for spherical k-means clustering

# Read the dataset from a CSV file, specifying the file path and naming convention
X = read.csv(paste('results_model_2/data/data/data_', 1, '.csv', sep=''))

# Remove the first column from the dataset, which is usually an index or identifier
X = X[,-1]

# Determine the number of features (dimensions) and observations (data points) in the dataset
d = ncol(X)  # Number of columns (features)
n = nrow(X)  # Number of rows (observations)

# Set the number of clusters (k) as the square root of the number of observations, rounded to the nearest integer
k = as.integer(sqrt(n))

# Initialize a numeric vector to store the Euclidean norms of each observation
norms = numeric(n)

# Calculate the Euclidean norm (L2 norm) for each observation in the dataset
for (i in 1:n) {
    norms[i] = norm(as.matrix(X)[i,], type = "2")  # Calculate the L2 norm for the i-th observation
}

# Determine the number of clusters based on the number of features, dividing by 4
nclusters = as.integer(d / 4)

# Check if the calculated number of clusters is less than the previously defined k
if (nclusters < k) {
    # Identify indices of observations with norms that rank in the top (n - k) largest values
    indices = rank(norms) > n - k
    
    # Filter the dataset to only include observations corresponding to the identified indices
    norms_ext = as.matrix(X[indices, ])
    
    # Perform spherical k-means clustering on the filtered dataset with the specified number of clusters
    km <- skmeans(norms_ext, nclusters)
    
    # Create a data frame containing the cluster centers from the k-means clustering
    centers = data.frame(km$prototypes)
    
    # Write the cluster centers to a CSV file for later analysis
    write.csv(centers, file=paste("results_model_2/results_skmeans/skmeans/centers_", 1, ".csv", sep=''), row.names=F)
}

# Check if the number of observations exceeds 1000 before proceeding with additional analysis
if (n > 1000) {
    # Transpose the data matrix so that features are rows and observations are columns
    X = t(X)
    
    # Create a sequence of proportions to be used in the clustering process
    prop <- seq(0.01, 0.15, by = 0.005)
    
    # Call the muscle_clusters function to perform clustering and obtain cluster directions
    directions <- muscle_clusters(X, prop)
    
    # Extract the membership matrix (excluding the last row) from the clustering results
    M_emp <- as.matrix(directions[[1]][-(d + 1), ])
    
    # Convert the membership matrix to a data frame for easier manipulation
    M_emp = data.frame(M_emp)
    
    # Write the membership matrix to a CSV file for later analysis, without row names
    write.csv(M_emp, file=paste("results_model_2/results_muscle/muscle/Memp_", 1, ".csv", sep=''), row.names=F)
}