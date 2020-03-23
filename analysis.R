library(dplyr)
library(ggplot2)
library(reshape2)
library(gganimate)

theme_update(text = element_text(size = 30))

## SPECIFIC GAME DATA
df <- setNames(
    read.csv("game.csv", header=F),
    c("game.id", "turn", "team", "lap", "player.id", "x", "y", "reward")
)

df = df %>% group_by(game.id, player.id) %>%
    mutate(
        dx = c(0, diff(x)),
        dy = c(0, diff(y)),
        speed = sqrt(dx^2 + dy^2)
    )

gid <- tail(df$game.id, 1)
df.game <- df %>% filter(game.id == gid, turn < 1000)

ggplot(df.game, aes(x, y, color = as.character(team), shape = as.character(player.id), size = lap)) + geom_point() + transition_time(turn) + ease_aes('linear')

ggplot(df.game, aes(x = x, y = y, group = interaction(game.id, player.id))) + geom_path() + geom_point(aes(shape = as.character(lap), color = as.character(team)), size = 4) + coord_fixed()

## df.meta <- setNames(read.csv("game.meta.csv", header=F), c("pid", "label", "age", "score", "nancestors", "nparents", "relative", "speed", "nbase", "alpha", "cluster", "bwc", "bws", "dqm", "dqstd"))

## POPULATION STATISTICS
df.population <- setNames(read.csv("population.csv", header=F), c("epoch", "pid", "rank", "nancestors", "nparents", "score", "age", "treesize", "lrate"))

rank.limit <- 10
population.means <- df.population %>% 
    group_by(epoch) %>%
    filter(rank < rank.limit) %>%
    summarize(
        anc.mean = mean(nancestors),
        anc.std = sd(nancestors),
        tree.mean = mean(treesize),
        tree.std = sd(treesize),
        lrate.mean.per1000 = mean(1000 * lrate),
        lrate.std.per1000 = sd(1000 * lrate),
        age.mean = mean(age),
        age.std = sd(age)
    ) %>% melt(id.vars = "epoch") %>%
    arrange(epoch)

## dev.new(width=35, height=18)
ggplot(population.means, aes(x = epoch, y = value, group = variable, color = variable, linetype = variable)) + geom_path(size=2)

ggplot(df.population %>% filter(epoch == max(epoch)), aes(x = rank, y = treesize)) + geom_bin2d()

ggplot(df.population %>% filter(epoch == max(epoch)), aes(x = rank, y = nancestors)) + geom_bin2d()

## pure train stats
df.pure <- setNames(read.csv("pure_train.meta.csv", header=F), c("epoch", "pid", "label", "age", "score", "nancestors", "nparents", "relative", "speed", "treesize", "lrate"))

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
        age = mean(age),
        ntree = treesize[1]
    ) %>%
    group_by(era) %>%
    arrange(-avg.speed) %>%
    summarize(
        speed.best = max(avg.speed),
        speed.q90 = quantile(avg.speed, probs = 0.9),
        speed.median = median(avg.speed),
        best.ntree = ntree[1],
        best.pid = pid[1],
        second.pid = pid[2],
        third.pid = pid[3],
    )

## REFERENCE GAME STATISTICS
df.meta <- setNames(read.csv("game.meta.csv", header=F), c("pid", "label", "age", "score", "nancestors", "nparents", "relative", "speed", "treesize", "lrate"))

df.meta <- df.meta %>%
    mutate(gid = ceiling(row_number()/3)) %>%
    mutate(win = as.numeric(relative >= 1))

df.meta %>% tail

ggplot(df.meta, aes(x = gid, y = speed)) + geom_point() + geom_smooth()
ggplot(df.meta, aes(x = gid, y = relative)) + geom_point() + geom_smooth() + ylim(c(0, 3))
ggplot(df.meta, aes(x = gid, y = win)) + geom_smooth()

df.meta %>% mutate(lev = ceiling(gid/50)) %>% group_by(lev) %>% summarize(s = mean(speed), w = mean(win)) %>% data.frame

## df.meta <- df.meta %>% filter(gid > max(gid) - 400)

df.melted <- df.meta %>% select(gid, pid, switch.pid, age, win, relative, nancestors, nparents, treesize) %>%
    melt(id.vars=c("gid", "pid", "switch.pid", "nparents"))

dev.new(width=20, height=11)
ggplot(df.melted, aes(x = gid, y = value, group = variable, color = variable)) + coord_cartesian(ylim = c(0, 2)) +
    geom_point(size = 5, alpha = 0.2, aes(shape = as.character(nparents))) + geom_smooth(n = 300) + 
    geom_hline(yintercept = 1, linetype = 'dashed', color = "red") ## +
    ## geom_vline(data = df.meta %>% filter(switch.pid), aes(xintercept = gid), alpha = 0.3) +
    ## geom_text(data = df.meta %>% filter(switch.pid), aes(x = gid, y = 0, group = pid, label = as.character(pid)), color = "black", angle = 90, vjust = 1.2, size = 8)
