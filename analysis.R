library(dplyr)
library(ggplot2)
library(reshape2)
library(gganimate)
library(plotly)
library(gridExtra)
library(tidyr)


## sample 1, 2, 9:
## step = 1e-9
## gradient ?
## num dy/dx -1e20
## opt exits no iteration: d = 0
## different outcoes

## sample 3, 4, 8:
## step 1e-8
## gradient:       0         0      2796   3072469     35137 117547022
## opt: overshoots local min at 7e-9, one step 1e-9 -> 1e-8, exits: d < 1e-8
## different outcomes

## sample 5, 6, 7
## step: -1e-6 (negative)
## gradient:       0         0     20284   1385766    100761 105622920
## opt: steps backwards, worsens objective

step * 100
ggplot(data.frame(x = step * seq_along(landscape), y = landscape), aes(x = x, y = y)) + geom_path() + geom_vline(xintercept = 100 * step) + geom_vline(xintercept = 1e-9)

ggplot(data.frame(x = step * (1:100), y = head(landscape, 100)), aes(x = x, y = y)) + geom_path()

landscape[1]
landscape[2]
landscape[101]

optpath
ggplot(data.frame(x = seq_along(optpath$ys),  y = optpath$ys), aes(x = x, y = y)) + geom_path()

ggplot(data.frame(x = gradient), aes(x = x)) + geom_density()
summary(abs(gradient))


## analysis

tree.evaluator.cols <- c("treesize", "lrate")

agent.cols <- c(
    "pid",
    "label",
    "age",
    "score",
    "nancestors",
    "nparents",

    "training.rel.change.mean",
    ## "training.rel.change.max",
    "training.rel.change.output",
    ## "training.rate.zero",
    "training.rate.successfull",
    ## "training.rate.accurate",
    ## "training.rate.correct.sign",
    "training.rate.optim.failed",

    tree.evaluator.cols
)

pod.game.cols <- c("did.finish", "relative", "speed")

## Pure train

## cases
## extreme gradient: 1

df.pt <- setNames(read.csv("pure_train.meta.csv", header=F, stringsAsFactors=F), c("epoch", agent.cols, pod.game.cols))

top.pids <- (df.pt %>%
    select(epoch, pid, relative, speed) %>%
    filter(epoch > 5) %>%
    group_by(pid) %>%
    mutate(rs = weighted.mean(relative, epoch)) %>%
    group_by %>%
    arrange(-rs) %>%
    filter(pid %in% head(unique(pid), 5)))$pid %>% unique

df.pt %>%
    filter(did.finish > 0) %>%
    select(epoch, pid, relative, speed) %>%
    filter(pid %in% top.pids) %>%
    mutate(speed = speed / 250, win = relative > 1) %>%
    pivot_longer(-c(epoch, pid)) %>%
    ggplot(aes(x = epoch, y = value)) +
    geom_point() + 
    geom_smooth() +
    geom_hline(yintercept = 0:1) +
    coord_cartesian(ylim = c(-0.5, 1.5)) +
    facet_grid(pid~name)

df.pt %>%
    select(epoch, pid, starts_with("training.rate")) %>%
    filter(pid %in% top.pids) %>%
    pivot_longer(-c(epoch, pid)) %>%
    ggplot(aes(x = epoch, y = value, group = interaction(name, pid), color = name)) +
    geom_path(size=2, alpha=0.3) +
    facet_grid(.~name, scales="free")

df.pt %>%
    select(epoch, pid, starts_with("training.rel")) %>%
    filter(pid %in% top.pids) %>%
    pivot_longer(-c(epoch, pid)) %>%
    ggplot(aes(x = epoch, y = value, group = interaction(name, pid), color = name)) +
    geom_path() +
    facet_grid(pid~name, scales="free")

df.pt %>%
    filter(pid %in% top.pids) %>%
    mutate(cent = ceiling(epoch / 100)) %>%
    group_by(cent, pid) %>%
    summarize(r = mean(speed)) %>%
    group_by(cent) %>%
    filter(r == max(r)) %>%
    arrange(cent) %>%
    data.frame
    

## ****************************************
## POPULATION ANALYSIS
## ****************************************

df.pt <- setNames(read.csv("data/population.csv", header=F, stringsAsFactors=F), c("epoch", "rank", agent.cols))

df.pt %>%
    select(epoch, pid, starts_with("training.rate")) %>%
    ## filter(pid %in% top.pids) %>%
    pivot_longer(-c(epoch, pid)) %>%
    ggplot(aes(x = epoch, y = value, group = interaction(name, pid), color = name)) +
    geom_path(size=2, alpha=0.3) +
    facet_grid(.~name, scales="free")

df.pt %>%
    select(epoch, pid, starts_with("training.rel")) %>%
    ## filter(pid %in% top.pids) %>%
    mutate(
        training.rel.change.mean = 100 * training.rel.change.mean,
        training.rel.change.max = 100 * training.rel.change.max
    ) %>%
    pivot_longer(-c(epoch, pid)) %>%
    ggplot(aes(x = epoch, y = value, color = name)) +
    geom_smooth() +
    facet_grid(.~name)

## df.pt %>%
##     ## filter(pid %in% top.pids) %>%
##     mutate(cent = ceiling(epoch / 100)) %>%
##     group_by(cent, pid) %>%
##     summarize(r = mean(speed)) %>%
##     group_by(cent) %>%
##     filter(r == max(r)) %>%
##     arrange(cent) %>%
##     data.frame

## df.pt %>%
##     select(epoch, pid, relative, speed) %>%
##     ## filter(pid %in% top.pids) %>%
##     mutate(speed = speed / 250) %>%
##     pivot_longer(-c(epoch, pid)) %>%
##     ggplot(aes(x = epoch, y = value)) +
##     geom_smooth() +
##     facet_grid(pid~name)

## df.population %>%
##     select(age, starts_with("training.rel.change")) %>%
##     pivot_longer(-age) %>%
##     ggplot(aes(x = age, y = value, group = name, color = name)) +
##     geom_point() +
##     facet_grid(name~., scales="free")

## df.population %>%
##     select(age, starts_with("training.rate")) %>%
##     pivot_longer(-age) %>%
##     ggplot(aes(x = age, y = value, group = name, color = name)) +
##     geom_point() +
##     facet_grid(name~., scales="free")

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
    mutate(win = as.numeric(relative >= 1))

## df.meta %>% filter(epoch == max(epoch))

span <- 1
ref.plot <- ggplot(
    df.meta,
    aes(x = epoch, y = relative, shape = as.factor(did.finish)## , color = as.factor(rank), group = as.factor(rank)
        )
) +
    geom_point(aes(size=age), alpha=0.25) +
    geom_smooth(se=F, span=span, aes(shape = NULL)) + ## relative speed
    geom_smooth(se=F, span=span, linetype="dashed", aes(y = 2 * win, shape=NULL)) + ## winrate
    geom_hline(yintercept = 0) +
    geom_hline(yintercept = 1) +
    coord_cartesian(ylim=c(-1, 2))

## COMBINE PLOTS
grid.arrange(pop.plot, ref.plot, ncol=2)

summary(lm(win~epoch, data = df.meta %>% filter(rank == 0)))

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
