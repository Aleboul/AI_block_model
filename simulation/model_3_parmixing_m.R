# Load required libraries
library(copula)  # For working with copulas, which are functions that couple multivariate distribution functions
library(stats)   # For basic statistical functions and methods

# Function to generate a copula object
generate_copula <- function() {
    # Obtain a parameter value (tau) for the Clayton copula at a specified value (0.5)
    thetabase = copClayton@iTau(0.5)  
    # Calculate the power copula associated with the Clayton copula
    opow.Clayton = opower(copClayton, thetabase)  
    
    # Define theta parameters for the copula, with three values
    theta = c(1, 10/7, 10/7)  
    
    # Create a nested copula structure using the specified theta values
    copoc = onacopulaL(opow.Clayton, list(theta[1], NULL, list(
        list(theta[2], 1:100),  # Sub-copula with theta[2] applied to the first 100 dimensions
        list(theta[3], 101:200) # Sub-copula with theta[3] applied to the next 100 dimensions
    )))
    return(copoc)  # Return the generated copula
}

# Function to generate a random series based on the copula
generate_randomness <- function(nobservation, p, d, seed, copoc) {
    set.seed(seed)  # Set the random seed for reproducibility
    # Generate random samples from the specified copula
    sample_ = rnacopula(nobservation, copoc)  
    
    # Generate a binary vector I_ with 'nobservation' elements, where each element is 1 with probability p
    I_ = rbinom(nobservation, size = 1, prob = p)  
    
    # Initialize a matrix to hold the time series, with one extra row for initial conditions
    series = matrix(0, nobservation + 1, d)  
    
    # Generate the first observation from the copula
    series[1,] = rnacopula(1, copoc)  
    
    # Loop through each observation to construct the time series
    for (i in 1:nobservation) {
        if (I_[i] == 1) {
            # If the random variable is 1, use the sampled value
            series[i + 1,] = sample_[i,]  
        } else {
            # Otherwise, retain the previous value in the series
            series[i + 1,] = series[i,]  
        }
    }
    return(series[-1,])  # Return the series without the first row (initial condition)
}

# Function to apply a certain transformation to the generated randomness
robservation <- function(randomness, m, d, k) {
    # Reshape the randomness matrix into a matrix with 'm' rows
    mat = matrix(randomness, m)  
    
    # Apply the maximum function across the columns to get the maximum values per dimension
    dat = apply(mat, 2, max)  
    
    # Reshape the maximum values into a matrix of dimensions (k, d), raised to the power of 'm'
    return(matrix(dat, k, d)^m)  
}