library(dplyr)
library(ggplot2)
library(reshape2)
library(gganimate)
library(plotly)
library(gridExtra)
library(tidyr)

tree.evaluator.cols <- c("treesize", "lrate")

agent.cols <- c(
    "pid",
    "label",
    "age",
    "score",
    "nancestors",
    "nparents",

    "training.rel.change.mean",
    "training.rel.change.max",
    "training.rel.change.output",
    "training.rate.zero",
    "training.rate.successfull",
    "training.rate.accurate",
    "training.rate.correct.sign",
    "training.rate.optim.failed",

    tree.evaluator.cols
)

pod.game.cols <- c("did.finish", "relative", "speed")

## Pure train
df.pt <- setNames(read.csv("pure_train.meta.csv", header=F, stringsAsFactors=F), c("epoch", agent.cols, pod.game.cols))

top.pids <- (df.pt %>%
    select(age, pid, relative, speed) %>%
    filter(age > 130) %>%
    group_by(pid) %>%
    mutate(rs = weighted.mean(relative, age)) %>%
    group_by %>%
    arrange(-rs) %>%
    filter(pid %in% head(unique(pid), 5)))$pid

df.pt %>%
    select(age, pid, starts_with("training.rate")) %>%
    filter(pid %in% top.pids) %>%
    pivot_longer(-c(age, pid)) %>%
    ggplot(aes(x = age, y = value, group = interaction(name, pid), color = name)) +
    geom_path() +
    facet_grid(pid~name, scales="free")

df.pt %>%
    select(age, pid, starts_with("training.rel")) %>%
    filter(pid %in% top.pids, age > 10) %>%
    mutate(
        training.rel.change.mean = 100 * training.rel.change.mean,
        training.rel.change.max = 100 * training.rel.change.max
    ) %>%
    pivot_longer(-c(age, pid)) %>%
    ggplot(aes(x = age, y = value, group = interaction(name, pid), color = name)) +
    geom_path() +
    facet_grid(pid~name, scales="free")

df.pt %>%
    select(age, pid, relative, speed) %>%
    filter(pid %in% top.pids) %>%
    mutate(speed = speed / 250) %>%
    pivot_longer(-c(age, pid)) %>%
    ggplot(aes(x = age, y = value, group = interaction(name, pid), color = interaction(name, pid))) +
    geom_smooth() +
    facet_grid(pid~name, scales="free")
    

## ****************************************
## POPULATION ANALYSIS
## ****************************************

df.population <- setNames(read.csv("data/population.csv", header=F), c("epoch", "rank", agent.cols))

df.population %>%
    select(age, starts_with("training.rel.change")) %>%
    mutate(
        training.rel.change.mean = training.rel.change.mean * 1000,
        training.rel.change.max = training.rel.change.max * 1000
    ) %>%
    pivot_longer(-age) %>%
    ggplot(aes(x = age, y = value, group = name, color = name)) +
    geom_point() +
    geom_path()

df.population %>%
    select(age, starts_with("training.rate")) %>%
    pivot_longer(-age) %>%
    ggplot(aes(x = age, y = value, group = name, color = name)) +
    geom_point() +
    geom_smooth() +
    ylim(c(-0.1,1.1))

rank.limit <- 8
pop.plot <- df.population %>%
    filter(rank < rank.limit) %>%
    transmute(
        epoch = epoch,
        ancestors = nancestors,
        nparents.per100 = 100 * nparents,
        age.x10 = 0.1 * age,
        treesize = treesize
    ) %>%
    melt(id.vars = "epoch") %>%
    arrange(epoch) %>%
    ggplot(aes(x = epoch, y = value, group = variable, color = variable)) +
    geom_point() +
    geom_smooth()

## ggplot(df.population %>% filter(epoch == max(epoch)), aes(x = rank, y = treesize)) + geom_bin2d()

## ggplot(df.population %>% filter(epoch == max(epoch)), aes(x = rank, y = nancestors)) + geom_bin2d()

## ****************************************
## REFERENCE GAME STATISTICS
## ****************************************

df.meta <- setNames(
    read.csv("data/game.meta.csv", header=F),
    c("epoch", "rank", agent.cols, pod.game.cols)
) %>%
    mutate(win = as.numeric(relative >= 1)) %>%
    filter(epoch >= 27)

## df.meta %>% filter(epoch == max(epoch))

span <- 1
ref.plot <- ggplot(
    df.meta,
    aes(x = epoch, y = relative, shape = as.factor(did.finish), color = as.factor(rank), group = as.factor(rank))
) +
    geom_point(aes(size=age), alpha=0.25) +
    geom_smooth(se=F, span=span) + ## relative speed
    geom_smooth(se=F, span=span, linetype="dashed", aes(y = 2 * win, shape=NULL)) + ## winrate
    geom_hline(yintercept = 0) +
    geom_hline(yintercept = 1) +
    coord_cartesian(ylim=c(-1, 2))

## COMBINE PLOTS
grid.arrange(pop.plot, ref.plot, ncol=2)

summary(lm(win~epoch, data = df.meta %>% filter(rank == 0, epoch > 35)))

df.meta %>%
    mutate(decade = ceiling(epoch / 10)) %>%
    group_by(decade, rank) %>%
    summarize(
        winrate = mean(win),
        rel.speed = mean(relative),
        n = n()
    ) %>%
    filter(rank == 0)


## ****************************************
## SPECIFIC GAME DATA
## ****************************************

df <- setNames(
    read.csv("data/game.csv", header=F),
    c("game.id", "turn", "team", "lap", "player.id", "x", "y", "reward")
)

df = df %>% group_by(game.id, player.id) %>%
    mutate(
        dx = c(0, diff(x)),
        dy = c(0, diff(y)),
        speed = sqrt(dx^2 + dy^2)
    ) %>%
    group_by

## hack: drop last ten games to get rank N player
N <- 0
idx <- 5
gid <- head(tail(unique(df$game.id), 5*(3*idx + 2-N)+1), 1)

df.game <- df %>%
    filter(game.id == gid)

p <- ggplot(
    df.game,
    aes(x, y,
        frame = turn,
        color = speed, 
        shape = as.factor(team),
        size = lap
        )
) + geom_point() 

ggplotly(p)


## ****************************************
## OLD STUFF
## ****************************************

theme_update(text = element_text(size = 30))

p.angle <- function(x) if (x[1] > 0) {atan(x[2] / x[1])} else {pi + atan(x[2] / x[1])}

x <- c(x = 10032.248972211251, y = 7946.7605350283875)
cp <- setNames(as.data.frame(t(matrix(c(11286.7,8791.31,6367.77,3864.65,8625.6,1915.53,5147.34,8753.89,4547.67,2339.09), nrow = 2))), c("x", "y"))

d <- cp[2,] - x
a <- 4.37
a_ncp <- p.angle(d) - a

ggplot(df.game, aes(x = x, y = y, group = player.id)) +
    geom_path() +
    geom_point(aes(shape = as.character(lap), color = as.character(team)), size = 4) +
    geom_point(data = cp, color="red", size=10, aes(group = NULL)) +
    coord_fixed()

## pure train stats
df.pure <- setNames(read.csv("pure_train.meta.csv", header=F), c("epoch", "pid", "finished", "relative", "speed"))

## ggplot(df.pure, aes(x = epoch, y = speed)) + geom_point() + geom_smooth();

## ggplot(df.pure, aes(x = epoch, y = relative)) + geom_point() + geom_smooth();

rank.stat <- function(speed, epoch) {
    m <- lm(speed~epoch)
    predict(m, newdata=data.frame(epoch = max(epoch)))
}

prog <- df.pure %>%
    filter(epoch > max(epoch) - 100) %>%
    group_by(pid) %>%
    summarize(prog = rank.stat(speed, epoch)) %>%
    arrange(-prog) %>%
    mutate(
        rank = row_number(),
        rank = as.numeric(rank < 4) * rank
    )

df.pure <- df.pure %>%
    left_join(prog, by = "pid")

ggplot(df.pure, aes(x = epoch, y = speed, color = as.character(rank))) + geom_point(alpha=0.3) + geom_smooth()

## TODO: why so many relative > 1?
ggplot(df.pure, aes(x = epoch, y = relative, color = as.character(rank))) + geom_point(alpha=0.3) + geom_smooth() + ylim(c(0, 2))

df.pure %>%
    mutate(era = floor(epoch / 50) + 1) %>%
    group_by(era, pid) %>%
    summarize(
        avg.speed = mean(speed),
        ## age = mean(age),
        ## ntree = treesize[1]
    ) %>%
    group_by(era) %>%
    arrange(-avg.speed) %>%
    summarize(
        speed.best = max(avg.speed),
        speed.q90 = quantile(avg.speed, probs = 0.9),
        speed.median = median(avg.speed),
        ## best.ntree = ntree[1],
        best.pid = pid[1],
        second.pid = pid[2],
        third.pid = pid[3],
    )


## df.meta <- df.meta %>% filter(gid > max(gid) - 400)

df.melted <- df.meta %>% select(gid, pid, switch.pid, age, win, relative, nancestors, nparents, treesize) %>%
    melt(id.vars=c("gid", "pid", "switch.pid", "nparents"))

dev.new(width=20, height=11)
ggplot(df.melted, aes(x = gid, y = value, group = variable, color = variable)) + coord_cartesian(ylim = c(0, 2)) +
    geom_point(size = 5, alpha = 0.2, aes(shape = as.character(nparents))) + geom_smooth(n = 300) + 
    geom_hline(yintercept = 1, linetype = 'dashed', color = "red") ## +
    ## geom_vline(data = df.meta %>% filter(switch.pid), aes(xintercept = gid), alpha = 0.3) +
    ## geom_text(data = df.meta %>% filter(switch.pid), aes(x = gid, y = 0, group = pid, label = as.character(pid)), color = "black", angle = 90, vjust = 1.2, size = 8)
