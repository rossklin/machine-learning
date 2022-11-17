library(tidyverse)
library(FNN)
library(microbenchmark)

n <- 1e4
d <- 2
data <- matrix(rnorm(n * d), nrow = n)

microbenchmark(test <- knn.index(data, k = 3))

data2 <- rbind(data, rnorm(2))
microbenchmark(test2 <- knn.index(data2, k = 3))
