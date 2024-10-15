# Load required libraries
library(copula)  # For creating and working with copulas
library(stats)   # For basic statistical functions and methods

# Function to generate a copula based on the specified dimensions and number of groups
generate_copula <- function(d, K, seed) {
    set.seed(seed)  # Set the seed for reproducibility
    
    # Obtain a parameter value (tau) for the Clayton copula at a specified value (0.5)
    thetabase = copClayton@iTau(0.5)  
    
    # Calculate the power copula associated with the Clayton copula
    opow.Clayton = opower(copClayton, thetabase)  
    
    # Define theta parameters for the copula (six values)
    theta = c(1, 10/7, 10/7, 10/7, 10/7, 10/7)  
    
    # Initialize probabilities for each group
    probs = rep(0, K)  
    
    # Calculate probabilities for the first K-1 groups
    for (k in 1:(K-1)) {  # Loop from 1 to K-1
        p = (1/2) ** k  # Probability for group k
        probs[k] = p  # Assign the probability
    }
    
    # Calculate the last probability as the remaining amount to make total sum 1
    probs[K] = 1 - sum(probs)  
    
    # Generate sizes for each group using a multinomial distribution
    sizes = as.vector(rmultinom(n = 1, size = d, prob = probs))  
    
    # Compute cumulative sizes for indexing purposes
    sizes_sum = cumsum(sizes)  
    
    # Create a nested copula structure using the specified theta values and sizes
    copoc = onacopulaL(opow.Clayton, list(theta[1], NULL, list(
        list(theta[2], 1:sizes_sum[1]),  # Sub-copula for first group
        list(theta[3], (sizes_sum[1]+1):sizes_sum[2]),  # Second group
        list(theta[4], (sizes_sum[2]+1):sizes_sum[3]),  # Third group
        list(theta[5], (sizes_sum[3]+1):sizes_sum[4]),  # Fourth group
        list(theta[6], (sizes_sum[4]+1):sizes_sum[5])   # Fifth group
    )))
    
    return(list(copoc, sizes))  # Return the copula and the sizes as a list
}

# Function to generate a random series based on the copula
generate_randomness <- function(nobservation, p, d, seed, copoc) {
    set.seed(seed)  # Set the random seed for reproducibility
    
    # Generate random samples from the specified copula
    sample_ = rnacopula(nobservation, copoc)  
    
    # Generate a binary vector I_ with 'nobservation' elements, where each element is 1 with probability p
    I_ = rbinom(nobservation, size = 1, prob = p)  
    
    # Initialize a matrix to hold the time series, with one extra row for the initial conditions
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

# Function to apply a transformation to the generated randomness
robservation <- function(randomness, m, d, k) {
    # Reshape the randomness matrix into a matrix with 'm' rows
    mat = matrix(randomness, m)  
    
    # Apply the maximum function across the columns to get the maximum values per dimension
    dat = apply(mat, 2, max)  
    
    # Reshape the maximum values into a matrix of dimensions (k, d), raised to the power of 'm'
    return(matrix(dat, k, d) ^ m)  
}