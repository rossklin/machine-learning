library(tidyverse)
library(ggplot2)
library(progress)
library(plotly)

## Band widths and gamma for "bad points" term
bw_reward <- 0.3
bw_state <- 0.15
bw_prox <- 0.15
gamma <- 0.3
noise <- 0.2

## Target behaviuor
f <- function(x, z) sin(pi/2 * x) * cos(pi/2 * z)

## Kernel function
kernel <- function(d, h) exp(-(d / h)^2)

## Reward rule
eval <- function(y, x, z) {
    d <- y - f(x, z)
    if (abs(d) < 0.5) {
        kernel(d, bw_reward)
    } else {
        -kernel(1 / d, 1)
    }
}

## Distances and kernel weights utility function
distances <- function(df, x0, z0) {
    df %>%
        mutate(
            delta = sqrt((x0 - x)^2 + (z0 - z)^2),
            w = kernel(delta, bw_state)
        )
}

## Predict using rewards from earlier tests directly
fpred <- function(dfw) {
    res <- optim(par = 0.5, method = "Brent", lower = -1, upper = 1, fn = function(y0) {
        ## Punishment for distance to good points
        err <- dfw %>%
            filter(reward > 0) %>%
            summarize(pred = weighted.mean((y0 - y)^2, w * reward)) %>%
            pull(pred)
        ## Punishment for proximity to bad points
        tau <- dfw %>%
            filter(reward < 0) %>%
            summarize(t = max(abs(reward) * kernel(sqrt(delta^2 + (y0 - y)^2), bw_prox))) %>%
            pull(t)
        err + gamma * tau
    })
    res$par
}

## Start with some initial noisy data
ninit <- 100
df <- tibble(
    y = rnorm(ninit),
    x = rnorm(ninit),
    z = rnorm(ninit),
    t = "init",
    reward = runif(ninit, -0.1, 0.1)
)

## Run: sample responses with some variance, evaluate and update
nrun <- 1000
pb <- progress_bar$new(total = nrun)

for (i in 1:nrun) {
    pb$tick()
    nd <- data.frame(x = rnorm(1), z = rnorm(1))
    dfw <- distances(df, nd$x, nd$z)
    pred <- fpred(dfw)

    ## "cheat" local variance
    var <- dfw %>% summarize(var = weighted.mean((y - pred)^2, w)) %>% pull(var)

    sample <- rnorm(1, pred, sqrt(var / i))
    reward <- eval(sample, nd$x, nd$z) + rnorm(1, noise)
    
    nd <- tibble(nd, i = i, y = sample, t = "enforce", reward = reward, correct = f(nd$x, nd$z))

    df <- bind_rows(df, nd)
}

## Visualize
tail(df)

ggplot(df, aes(x = i, y = reward)) + geom_point() + geom_smooth()

buf <- tibble(x = seq(-1, 1, 0.1), z = 0) %>%
    rowwise() %>%
    do({
        row <- as_tibble(.)
        dfw <- distances(df, row$x, row$z)
        tibble(row, y = fpred(dfw), correct = f(row$x, row$z))
    })

ggplot(buf, aes(x = x, y = y)) + geom_path() + geom_path(aes(y = correct), color = "red")

buf <- tibble(z = seq(-1, 1, 0.1), x = 1) %>%
    rowwise() %>%
    do({
        row <- as_tibble(.)
        dfw <- distances(df, row$x, row$z)
        tibble(row, y = fpred(dfw), correct = f(row$x, row$z))
    })

ggplot(buf, aes(x = z, y = y)) + geom_path() + geom_path(aes(y = correct), color = "red")


nr <- 20
buf <- matrix(sapply(0:(nr * nr - 1), function(idx) {
    row <- idx %% nr
    col <- floor(idx / nr)
    x <- -2 + 4 * col / (nr - 1)
    z <- -2 + 4 * row / (nr - 1)
    distances(df, x, z) %>% fpred()
}), nrow = nr)

buf2 <- matrix(sapply(0:(nr * nr - 1), function(idx) {
    row <- idx %% nr
    col <- floor(idx / nr)
    x <- -2 + 4 * col / (nr - 1)
    z <- -2 + 4 * row / (nr - 1)
    f(x, z)
}), nrow = nr)

## Note: y renamed as z
plot_ly(z = ~buf) %>% add_surface() %>% add_surface(z = ~buf2)
