library(tidyverse)
library(Matrix)

## Agent model
fluff.n <- function(x, lim) min(max(1, floor(rnorm(n=1, mean = x, sd = 0.2 * x))), lim)

winit <- function(n) runif(n, 0, 1)

sigmoid <- function(x) 1 / (1 + exp(-x))

w.sparsem <- function(nr, nc, branch, f = winit) {
  n.in <- fluff.n(nr * branch, nr * nc)
  w.in <- sparseMatrix(
    i = sample(1:nr, size = n.in, replace = T),
    j = sample(1:nc, size = n.in, replace = T),
    x = f(n.in),
    dims = c(nr, nc)
  )
}

make.agent <- function(d.in, d.out, d.inner = 1000, branch = 3) {
  if (d.inner < d.in + d.out + 1) stop("Insufficient inner dims!")

  list(
    # w.in = w.sparsem(d.inner, d.in, branch),
    # b.in = rnorm(d.in),
    w.brain = w.sparsem(d.inner, d.inner, branch),
    threshold = abs(rnorm(d.inner)),
    # w.out = w.sparsem(d.out, d.inner, branch, rnorm),
    # b.out = winit(d.out),
    # energy = rep(0, d.inner),
    # activation = rep(0, d.inner),
    activity.tracker = rep(0, d.inner),
    output.tracker = rep(0, d.out),
    output.std = rep(0, d.out),
    gamma = 10,
    branch = branch,
    active.threshold = 0.5,
    dead.threashold = 0.01,
    dim = d.inner,
    performance = 0,
    age = 0,
    noise = 0.1
  )
}

fade <- function(x, y, n) (x * n + y) / (n + 1)

evaluate <- function(data, agent) {
  activation <- rep(0, agent$dim)
  energy <- winit(agent$dim)
  d.in <- length(data$x[[1]])
  d.out <- length(data$y[[1]])
  add.dims <- agent$dim - d.in
  
  for (i in seq_along(data$x)) {
    ## Input mapped directly to first N nodes
    input <- c(sigmoid(data$x[[i]]), rep(0, add.dims))
    energy <- energy + input + agent$w.brain %*% activation + rnorm(agent$dim, 0, agent$noise)
    activation <- energy > agent$threshold
    energy <- ifelse(activation, 0, 0.9 * energy)

    output <- activation[(d.in+1):(d.in+d.out)]
    score <- 1 / (2 * abs(data$y[[i]] - output) + 1) ## TODO: dims

    agent$activity.tracker <- fade(agent$activity.tracker, activation, agent$gamma)
    agent$output.tracker <- fade(agent$output.tracker, output, agent$gamma)
    agent$output.std <- fade(agent$output.std, abs(output - agent$output.tracker), agent$gamma)

    agent <- enforce(agent, score)

    agent$performance <- fade(agent$performance, score, min(i - 1, 10))
  }

  agent
}

## Increase/reduce weights to and from active nodes
enforce <- function(agent, score, tol = 0.01, lim = 3) {
  s <- 0.9 + 0.2 * sigmoid(score - agent$performance)

  ## Find active nodes
  idx <- which(agent$activity.tracker > agent$active.threshold)
  n <- length(idx)

  ## Scale weights to/from nodes
  agent$w.brain[idx,] <- pmax(s * agent$w.brain[idx,] + rnorm(n, 0, 1e-3), 0)
  agent$w.brain[,idx] <- pmax(s * agent$w.brain[,idx] + rnorm(n, 0, 1e-3), 0)

  ## Scale down "burning" nodes
  idx <- which(agent$activity.tracker > 0.8)
  agent$w.brain[idx,] <- 0.95 * agent$w.brain[idx,]

  agent$w.brain <- drop0(agent$w.brain, tol = tol)

  ## Limit weight growth
  overflow <- which(agent$w.brain > lim)
  agent$w.brain[overflow] <- lim

  if (s > 1) {
    agent <- grow(agent)
  } else {
    agent <- ungrow(agent)
  }
  
  agent
}

## Add connections to active nodes
grow <- function(agent, f = winit) {
  idx.act <- which(agent$activity.tracker > agent$active.threshold)
  if (length(idx.act) < 1) return(agent)

  d <- agent$dim
  n <- fluff.n(d / 1000, length(idx.act))

  i <- sample(idx.act, n)
  agent$w.brain[i + d * (sample(1:d, n) - 1)] <- f(n)
  agent$w.brain[sample(1:d, n) + d * (i - 1)] <- f(n)

  agent
}

ungrow <- function(agent) {
  idx.act <- which(agent$activity.tracker > agent$active.threshold)
  if (length(idx.act) < 1) return(agent)

  d <- agent$dim
  n <- fluff.n(d / 100, length(idx.act))

  i <- sample(idx.act, n)
  agent$w.brain[i + d * (sample(1:d, n) - 1)] <- 0
  agent$w.brain[sample(1:d, n) + d * (i - 1)] <- 0
  agent$w.brain <- drop0(agent$w.brain)

  agent
}

## Remove inactive nodes
prune <- function(agent) {
  idx <- which(agent$activity.tracker < agent$dead.threashold)
  if (length(idx) > 0) {
    agent$w.brain[idx,] <- 0
    agent$w.brain[,idx] <- 0
    agent$w.brain <- drop0(agent$w.brain)
  }
  agent
}

train <- function(agent, f.batch, its = 100) {
  for (iteration in 1:its) {
    batch <- f.batch()
    agent <- evaluate(batch, agent)
    agent <- prune(agent)

    cat(sprintf(
      "Iteration %d: score %f, active = %d, dead = %d, burn = %d, out = %f (%f)\n", 
      iteration, agent$performance, 
      sum(agent$activity.tracker > agent$active.threshold), 
      sum(agent$activity.tracker < agent$dead.threashold),
      sum(agent$activity.tracker > 0.8),
      agent$output.tracker,
      agent$output.std
      ))
  }
}

batch.remember <- function(steps = 1) {
  function() {
    x <- rnorm(100) > 0
    y <- c(rep(0, steps), tail(x, -steps))
    list(x = as.list(2 * x - 1), y = as.list(y))
  }
}

main <- function(d = 100) {
  agent <- make.agent(d.in = 1, d.out = 1, d.inner = d, branch = 2)
  train(agent, batch.remember())
}
