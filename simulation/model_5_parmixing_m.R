# Load required libraries for copula modeling and statistical functions
library(copula)  # Provides functions to work with copulas
library(stats)   # Contains basic statistical functions

# Function to generate a copula and its associated sizes
generate_copula <- function(d, K, seed) {
    set.seed(seed)  # Set the seed for random number generation to ensure reproducibility
    
    # Obtain a parameter (tau) for the Clayton copula at a specified value (0.5)
    thetabase = copClayton@iTau(0.5)  
    
    # Calculate the power copula associated with the Clayton copula
    opow.Clayton = opower(copClayton, thetabase)  
    
    # Initialize probabilities for the first K-5 groups
    probs = rep(0, K - 5)  
    
    # Calculate probabilities for the first four groups
    for (k in 1:4) {
        p = (1/2) ** k  # Probability for group k based on halving
        probs[k] = p  # Assign calculated probability to the list
    }
    
    # The last probability is adjusted to ensure total sums to 1
    probs[K - 5] = 1 - sum(probs)  
    
    # Generate sizes for each group using a multinomial distribution
    sizes = as.vector(rmultinom(n = 1, size = d - 5, prob = probs))  
    
    # Append five units of size 1 to the sizes for additional groups
    sizes = c(sizes, rep(1, 5))  
    
    # Compute cumulative sums of sizes for later indexing
    sizes_sum = cumsum(sizes)  
    
    # Define theta parameters for the copula (11 values total)
    theta = c(1, 10/7, 10/7, 10/7, 10/7, 10/7, 10/7, 10/7, 10/7, 10/7, 10/7)  
    
    # Create a nested copula structure using specified theta values and sizes
    copoc = onacopulaL(opow.Clayton, list(theta[1], NULL, list(
        list(theta[2], 1:sizes_sum[1]),  # Sub-copula for the first group
        list(theta[3], (sizes_sum[1]+1):sizes_sum[2]),  # Second group
        list(theta[4], (sizes_sum[2]+1):sizes_sum[3]),  # Third group
        list(theta[5], (sizes_sum[3]+1):sizes_sum[4]),  # Fourth group
        list(theta[6], (sizes_sum[4]+1):sizes_sum[5]),  # Fifth group
        list(theta[7], (sizes_sum[5]+1):sizes_sum[6]),  # Sixth group
        list(theta[8], (sizes_sum[6]+1):sizes_sum[7]),  # Seventh group
        list(theta[9], (sizes_sum[7]+1):sizes_sum[8]),  # Eighth group
        list(theta[10], (sizes_sum[8]+1):sizes_sum[9]), # Ninth group
        list(theta[11], (sizes_sum[9]+1):sizes_sum[10]) # Tenth group
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