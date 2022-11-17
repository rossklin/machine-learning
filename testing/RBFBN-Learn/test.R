library(tidyverse)
library(ggplot2)

a <- -1
b <- 1

ninit <- 100
df <- tibble(
    y0 = rnorm(100),
    x0 = rnorm(100),
    z0 = rnorm(100),
    t = "init"
)

h <- 0.3
eval <- function(xs, ys) exp(-((ys - (a * xs + b))/h)^2)

m <- lm(y0~x0*z0, data = df)
origc <- coef(m)
df <- df %>% mutate(tracker = eval(x0, predict(m, newdata = df)))

for (i in 1:100) {
    nd <- data.frame(x0 = rnorm(1), z0 = rnorm(1))
    pred <- predict(m, newdata = nd, interval = "prediction") ## Class: matrix
    err <- min(0.2 * (pred[1, 3] - pred[1,2]), 1)
    sample <- rnorm(1, pred[1, 1], err)
    reward <- eval(nd$x0, sample)

    bw <- 0.2
    tracker <- weighted.mean(df$tracker, exp(-((df$x0 - nd$x0)^2 + (df$z0 - nd$z0)^2) / bw^2))
    
    if (reward > tracker) {
        ## This was an improvement, so just add the data point
        nd <- tibble(nd, y0 = sample, t = "enforce")
    } else {
        ## This was worse, pull model away from this point
        nd <- tibble(nd, y0 = 2 * pred[1,1] - sample, t = "avoid", pair = sample, mid = pred[1,1], base_tracker = tracker)
    }
    nd$tracker <- reward
    df <- bind_rows(df, nd) %>% mutate(idx = pmax(row_number() - ninit, 1))
    m <- update(m, data = df, weights = idx * (0.5 * (as.numeric(t == "init") + as.numeric(t == "avoid")) + as.numeric(t == "enforce")))
    ## m <- lm(y0~x0, data = df)
}

xs <- -2:2
ggplot(df, aes(x = x0, y = y0, group = t, color = t)) +
    geom_point() +
    geom_line(aes(y = tracker - 1, group=NULL, color=NULL), linetype="dashed") + 
    geom_line(data = tibble(x0 = xs, y0=a * xs + b, t="solution")) +
    geom_line(data = tibble(x0 = xs, y0=origc[2] * xs + origc[1], t="original")) +
    geom_line(data = tibble(x0 = xs, y0=coef(m)[2] * xs + coef(m)[1], t="final")) +
    ## geom_crossbar(aes(x = x0, y = mid, ymin = pmin(y0, pair), ymax = pmax(y0, pair)), width=0.1) +
    ggtitle(paste("Tracker", tracker), subtitle = paste("Coefs", paste(coef(m), collapse=",")))
