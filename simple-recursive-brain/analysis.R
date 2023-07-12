library(dplyr)
library(readr)
library(ggplot2)

run <- function() {
    df <- read_csv("simple-recursive-brain/output.csv") %>%
        mutate(test_id = as.integer(test_id), thread_id = as.integer(thread_id)) %>%
        group_by(thread_id) %>%
        mutate(is_max_test_id = test_id == max(test_id)) %>%
        ungroup()

    tmax <- max(df$iteration)

    ggplot(data = df) +
        geom_path(aes(x = iteration, y = rolling_rate, group = test_id, color = as.factor(thread_id), alpha = 0.2 + 0.8 * as.integer(is_max_test_id))) +
        geom_path(data = tibble(x = c(0, tmax), y = c(0.25, 0.25 + 0.00002 * tmax)), aes(x = x, y = y))

    ggsave(filename = "res.png", width = 16)
}
