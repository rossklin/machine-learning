
dval.cols <- c("current", "last", "value_ma", "sd_ma", "diff_ma", "n")
tree.evaluator.cols <- c("gamma", "wsum", "treesize")
## team.evaluator.cols <- c("treesize", "lrate", "mut_tag", ...)

agent.cols <- c("id",
"parent_hash",
"class_id",
"original_id",
"label",
paste("score_tmt", dval.cols, sep="."),
paste("score_simple", dval.cols, sep="."),
paste("score_refbot", dval.cols, sep="."),
"rank",
"last_rank",
"age",
"mut_age",
"future_discount",
"w_reg",
"mem_limit",
"mem_curve",
"inspiration_age_limit",
"learning.rate",
"step.limit",
"use.f0c",
"tstats.rel_change_mean",
"tstats.output_change",
"tstats.rate_successfull",
"tstats.rate_optim_failed",
paste("optimstats.success", dval.cols, sep="."),
paste("optimstats.improvement", dval.cols, sep="."),
paste("optimstats.its", dval.cols, sep="."),
paste("optimstats.overshoot", dval.cols, sep="."),
paste("optimstats.dx", dval.cols, sep="."),
paste("optimstats.dy", dval.cols, sep="."),
"parents.size",
"ancestors.size",
## team.evaluator.cols
tree.evaluator.cols
)

pod.game.cols <- c("did.finish", "relative", "speed")

ind.graph <- function(df.pt, cols) {
    df.pt %>%
        select(epoch, pid = id, one_of(cols)) %>%
        pivot_longer(-c(epoch, pid)) %>%
        ggplot(aes(x = epoch, y = value, group = name, color = name)) +
        geom_path(se=F) +
        geom_point() + 
        geom_hline(yintercept = 0:1) +
        coord_cartesian(ylim=c(-1, 3)) +
        facet_grid(.~pid, scales="free")
}

counts <- function(df.pt) {
    df.pt <- df.pt %>% mutate(x = score_simple.value_ma)
    n <- df.pt %>% distinct(id) %>% nrow
    lapply(list(
        e.all = df.pt %>% filter(x >= 1) %>% distinct(id) %>% nrow,
        e.1 = df.pt %>%
            filter(epoch == 1, x >= 1) %>%
            nrow,
        e.end = df.pt %>%
            filter(epoch == max(epoch), x >= 1) %>%
            nrow,
        e.impr = df.pt %>%
            group_by(id) %>%
            filter(x[epoch == 1] < 0.75, x >= 1) %>%
            group_by %>%
            distinct(id) %>%
            nrow,
        e.rem = df.pt %>%
            mutate(emax = max(epoch)) %>%
            group_by(id) %>%
            filter(max(epoch) < emax, x >= 1) %>%
            group_by %>%
            distinct(id) %>%
            nrow,
        e.worsen = df.pt %>%
            group_by(id) %>%
            filter(x[epoch == max(epoch)] < 0.75, x >= 1) %>%
            group_by %>%
            distinct(id) %>%
            nrow
    ), function(x) 100 * x/n
    )
}
