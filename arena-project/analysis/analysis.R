library(dplyr)
library(ggplot2)
library(tidyr)
library(gridExtra)

source("util.R")

run.id <- "2944455052"
## First run with objective function including all options
## batch.size: ceil(sqrt(epoch))
## agents speed > 1 at epoch 1 (0 games): 0.8%
## agents speed > 1 at epoch 25 (90 games): 0.15%
## agents speed > 1 who started < 1: 0.15%

run.id <- "3398175809"
## First run with init to zero
## 0.4% success from start, all crashed after training

run.id <- "3255738355"
## First run with flat exploration
## Updated initialization to small normal values instead of 0
## Fixed init error limit
## 0.2% success from start, all crashed after training

run.id <- "3850273625"
## Updated conf dists
## future_discount = pow(10, -u01(1, 2));
## w_reg = pow(10, -u01(3, 5));
## learning_rate = pow(10, -u01(4, 6));
## step_limit = pow(10, -u01(2, 4));
## use_f0c = u01() < 0.1;
## RESULTS:
## total: 0.275%, e1: 0.225%,  end: 0.050%, impr: 0.025%, removed: 0.125%,  worse: 0.125%

run.id <- "2358257395"
## Init with supervision data
## n = 100

df.pt <- setNames(read.csv(paste0("data/pure-train-run-", run.id, ".csv"), header=F, stringsAsFactors=F), c("epoch", "supervision", "reinforcement", agent.cols)) %>%
    filter(epoch < 100)

counts(df.pt)

df.pt %>%
    group_by(epoch) %>%
    summarize(
        success = mean(tstats.rate_successfull),
        best = max(score_simple.value_ma),
        best.cur = max(score_simple.current),
        n = n()
    )

df.pt %>%
    filter(epoch == max(epoch), score_simple.value_ma >= 1) %>%
    distinct(id, use.f0c, step.limit, learning.rate, future_discount, supervision, reinforcement) %>%
    summary

df.pt %>%
    mutate(x = score_simple.value_ma) %>%
    group_by(id) %>%
    filter(x[epoch == 1] < 0.75, x >= 1) %>%
    group_by %>%
    distinct(id, step.limit, learning.rate, future_discount, supervision, reinforcement)

top.pids <- (
    df.pt %>%
    filter(epoch > 1) %>%
    arrange(-score_simple.value_ma) %>%
    ## arrange(-optimstats.improvement.value_ma) %>%
    distinct(id) %>%
    head(10)
)$id

df.pt %>%
    filter(id %in% top.pids) %>%
    transmute(
        epoch = epoch,
        id = id,
        speed = score_simple.current,
        speed.ma = score_simple.value_ma,
        success = tstats.rate_successfull,
        ## improve = optimstats.improvement.current,
        ## its = optimstats.its.value_ma / 100,
        wsum = wsum / 100,
        treesize = treesize / 100,
        out.change = tstats.output_change,
        winrate = score_refbot.current
    ) %>%
    arrange(id, epoch) %>%
    ind.graph(c("speed", "speed.ma", "success", "wsum", "treesize", "out.change"))

df.pt %>%
    filter(id %in% top.pids) %>%
    distinct(id, reinforcement, supervision, treesize)


df.pt %>%
    filter(id %in% top.pids) %>%
    transmute(
        epoch = epoch,
        speed = score_simple.current,
        wsum = wsum,
        wchange = tstats.rel_change_mean,
        srate = tstats.rate_successfull,
        success = c(NA, head(srate, -1)) > c(NA, NA, head(srate, -2)) | c(NA, head(srate, -1)) == 1
    ) %>%
    arrange(epoch)

df.pt %>% filter(id == 17) %>%
    select(epoch, w_reg, opt.succ = optimstats.success.value_ma, speed = score_simple.current, dx = optimstats.dx.value_ma, dy = optimstats.dy.value_ma, wsum0, wsum1)

df.pt %>%
    group_by(epoch) %>%
    summarize(m = max(score_simple.current))
    
df.pt %>%
    select(epoch, pid = id, starts_with("tstats"), starts_with("wsum")) %>%
    filter(pid %in% top.pids) %>%
    pivot_longer(-c(epoch, pid)) %>%
    ggplot(aes(x = epoch, y = value, group = interaction(name, pid), color = name)) +
    geom_path(size=2, alpha=0.3) +
    facet_grid(name~., scales="free")
    
df.pt %>%
    select(epoch, pid = id, speed = score_simple.value_ma, change = score_simple.diff_ma) %>%
    mutate(cent = ceiling(epoch / 10)) %>%
    group_by(cent, pid) %>%
    summarize(r = mean(speed)) %>%
    group_by(cent) %>%
    filter(r == max(r)) %>%
    arrange(cent) %>%
    data.frame
    

## ****************************************
## POPULATION ANALYSIS
## ****************************************

## Note: from epoch 204, gamma1 is replaced by mut_tag
run.id <- "2939787966"
filter.epoch <- 380

df.pt <- setNames(read.csv(paste0("data/run-", run.id, "-population.csv"), header=F, stringsAsFactors=F), c("epoch", "rank", agent.cols)) %>%
    group_by(epoch, rank) %>%
    arrange(epoch, rank) %>%
    filter(row_number() == max(row_number())) %>%
    group_by %>%
    filter(epoch >= filter.epoch)

rank.limit <- 8
plot.rate <- df.pt %>%
    filter(rank < rank.limit, epoch > filter.epoch) %>%
    select(epoch, starts_with("training.rate")) %>%
    pivot_longer(-c(epoch)) %>%
    ggplot(aes(x = epoch, y = value)) +
    geom_point() +
    geom_smooth(size=2, alpha=0.3) +
    facet_grid(.~name, scales="free")

plot.rel <- df.pt %>%
    filter(rank < rank.limit, epoch > filter.epoch) %>%
    select(epoch, starts_with("training.rel")) %>%
    pivot_longer(-c(epoch)) %>%
    ggplot(aes(x = epoch, y = value)) +
    geom_smooth() +
    geom_point() +
    facet_grid(.~name)

df.pop.plot <- df.pt %>%
    filter(rank < rank.limit, epoch > max(epoch) - 40) %>%
    mutate(phash = ifelse(epoch >= max(epoch) - 30, phash, "old")) %>%
    transmute(
        epoch = epoch,
        phash = phash,
        ancestors = nancestors,
        age.per10 = age / 10,
        treesize = treesize
    ) %>%
    pivot_longer(-c(epoch, phash)) %>%
    arrange(epoch)

pop.plot <- df.pop.plot %>%
    ggplot(aes(x = epoch, y = value, group = name, color = name)) +
    geom_point(alpha=.3) +
    geom_smooth(data = df.pop.plot %>% filter(phash != "old"), method="lm", aes(group = interaction(name,phash))) +
    coord_cartesian(ylim=c(-1, 1200))

## ****************************************
## REFERENCE GAME STATISTICS
## ****************************************
df.meta <- setNames(
    read.csv(paste0("data/run-", run.id, "-refgames.csv"), header=F, stringsAsFactors=F),
    c("epoch", "opponent", "rank", agent.cols, pod.game.cols)
) %>%
    mutate(win = as.numeric(relative >= 1)) %>%
    group_by(epoch, rank, opponent) %>%
    arrange(epoch, rank, opponent) %>%
    filter(row_number() == max(row_number())) %>%
    group_by %>%
    filter(epoch >= filter.epoch)

span <- 1
ref.plot <- df.meta %>%
    filter(rank < 3, opponent == "refbot", epoch > max(epoch) - 40) %>%
    ## mutate(phash = ifelse(epoch >= max(epoch) - 20, phash, "old")) %>%
    ggplot(
        aes(x = epoch, y = relative, shape = as.factor(did.finish), color = phash, group = phash)
    ) +
    geom_point(aes(size=age), alpha=0.3) +
    geom_smooth(method="lm", se=F, span=span, size=1, aes(shape = NULL)) + ## relative speed
    geom_smooth(method="lm", se=F, span=span, size=1, linetype="dashed", aes(y = 2 * win, shape=NULL)) + ## winrate
    geom_smooth(color = "black", se=F, span=span, size=1, aes(shape = NULL, color = NULL, group = NULL)) + ## relative speed
    geom_smooth(color = "black", se=F, span=span, size=1, linetype="dashed", aes(y = 2 * win, shape=NULL, color = NULL, group = NULL)) + ## winrate
    geom_hline(yintercept = 0) +
    geom_hline(yintercept = 1) +
    coord_cartesian(ylim=c(-1, 2))

ref.plot2.df <- df.meta %>%
    filter(rank < 3, epoch > max(epoch) - 40)

ref.plot2 <- ref.plot2.df %>% ggplot(
        aes(x = epoch, y = relative, shape = as.factor(did.finish), color = opponent, group = opponent)
        ) +
    
    geom_point(aes(size=age), alpha=0.3) +

    geom_smooth(se=F, span=span, size=1, aes(shape = NULL)) + ## relative speed
    geom_smooth(se=F, span=span, size=1, linetype="dashed", aes(y = 2 * win, shape=NULL)) + ## winrate
    
    ## geom_smooth(data = ref.plot2.df %>% filter(opponent == "refbot"), color = "black", se=F, span=span, size=1, aes(shape = NULL, color = NULL, group = NULL)) + ## total relative speed
    ## geom_smooth(data = ref.plot2.df %>% filter(opponent == "refbot"), color = "black", se=F, span=span, size=1, linetype="dashed", aes(y = 2 * win, shape=NULL, color = NULL, group = NULL)) + ## total winrate
    
    geom_hline(yintercept = 0) +
    geom_hline(yintercept = 1) +
    
    coord_cartesian(ylim=c(-1, 2))

## COMBINE PLOTS
grid.arrange(ref.plot, ref.plot2, pop.plot, plot.rate, ncol=2)

df.pt %>% filter(epoch == max(epoch)) %>% summary

df.pt %>%
    filter(epoch == max(epoch)) %>%
    select(rank, age, treesize, lrate, gamma0, gamma1, score) %>%
    arrange(score) %>%
    data.frame

summary(lm(win~epoch, data = df.meta %>% filter(rank == 1, opponent == "refbot")))

df.meta %>%
    filter(rank < 3, opponent == "refbot") %>%
    mutate(decade = floor(epoch / 10) + 1) %>%
    group_by(decade, rank) %>%
    summarize(
        winrate = mean(win),
        rel.speed = mean(relative),
        n = n()
    ) 


## ****************************************
## SPECIFIC GAME DATA
## ****************************************

df <- setNames(
    read.csv(paste0("data/run-", run.id, "-game.csv"), header=F),
    c(
        "run.id", "epoch", "rank", "opponent",
        "game.id", "turn", "team", "lap", "player.id", "x", "y",
        "angle", "shield", "boost.count", "reward",
        "cp.xs", "cp.ys"
    )
)

arrow.size <- 400
df = df %>% group_by(game.id, player.id) %>%
    mutate(
        dx = c(0, diff(x)),
        dy = c(0, diff(y)),
        speed = sqrt(dx^2 + dy^2),
        arrow.x0 = x - arrow.size * cos(angle),
        arrow.y0 = y - arrow.size * sin(angle),
        arrow.x1 = x + arrow.size * cos(angle),
        arrow.y1 = y + arrow.size * sin(angle),
    ) %>%
    group_by

df.game <- df %>%
    filter(epoch == max(epoch), rank == 0) %>%
    filter(game.id == sample(game.id, 1))

df.cps <- df.game %>%
    head(1) %>%
    do(data.frame(
        cp.xs = as.numeric(strsplit(as.character(.$cp.xs), " ")[[1]]),
        cp.ys = as.numeric(strsplit(as.character(.$cp.ys), " ")[[1]])
    )) %>%
    mutate(cp.idx = 1:n())

df.game <- df.game %>% select(-cp.xs, -cp.ys)

draw.frame <- function(tmax) {
    ggplot(
        df.game %>% filter(turn < tmax, turn > tmax - 10, turn %% 2 == 0),
        aes(x, y,
            color = as.factor(team), 
            ## shape = as.factor(lap),
            ## size = reward,
            )
    ) +

    ## arrow
    geom_segment(
        aes(
            x = arrow.x0,
            y = arrow.y0,
            xend = arrow.x1,
            yend = arrow.y1
        ), arrow = arrow(length = unit(0.03, "npc"))
    ) +

    ## pod
    geom_point(
        size = 2,
        aes(shape = as.factor(player.id))
    ) +

    ## shield
    geom_point(
        data = df.game %>% filter(shield, turn < 40),
        color = "blue",
        alpha = 0.4
    ) +

    ## checkpoints
    geom_point(
        data = df.cps,
        size = 10,
        aes(x = cp.xs, y = cp.ys, frame = NULL, color = as.factor(cp.idx), size = NULL, shape = NULL)
    ) 
}



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
