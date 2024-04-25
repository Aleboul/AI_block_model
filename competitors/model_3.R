source('muscle.R')
library(MASS)
library(wordspace)
library(skmeans)

X = read.csv(paste('results_model_3/data/data/data_',1,'.csv', sep=''))

X = X[,-1]
d = ncol(X)
n = nrow(X)
k = as.integer(sqrt(n))
nclusters = as.integer(d / 4)
if(nclusters < k){
    norms = rowNorms(as.matrix(X), method = "euclidean", p = 2)
    indices = rank(norms) > n-k
    norms_ext = as.matrix(X[indices,])
    km <- skmeans(norms_ext, nclusters)
    centers = data.frame(km$prototypes)
    write.csv(centers,file=paste("results_model_3/results_skmeans/skmeans/centers_",1,".csv", sep=''), row.names=F)
}
if(n > 1000){
    X = t(X)
    prop <- seq(0.01, 0.15, by = 0.005)
    directions <- muscle_clusters(X, prop)
    M_emp <- as.matrix(directions[[1]][-(d+1), ])
    M_emp = data.frame(M_emp)
    write.csv(M_emp,file=paste("results_model_3/results_muscle/muscle/Memp_",1,".csv", sep=''), row.names=F)
}
