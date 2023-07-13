library(dplyr)
library(readr)
library(ggplot2)

run <- function() {
    while (T) {
        df <- read_csv("simple-recursive-brain/output.csv") %>%
            mutate(test_id = as.integer(test_id), thread_id = as.integer(thread_id)) %>%
            group_by(thread_id) %>%
            mutate(is_max_test_id = test_id == max(test_id)) %>%
            ungroup()

        tmax <- max(df$iteration)
        ts <- 0:tmax
        limits <- 1 / (1 + exp(-ts / 2200)) - 0.357

        ggplot() +
            geom_path(data = df %>% filter(is_max_test_id), alpha = 1, aes(x = iteration, y = rolling_rate, group = test_id, color = as.factor(thread_id))) +
            geom_path(data = df %>% filter(!is_max_test_id), alpha = 0.2, aes(x = iteration, y = rolling_rate, group = test_id, color = as.factor(thread_id))) +
            geom_path(data = tibble(x = ts, y = limits), aes(x = x, y = y))

        ggsave(filename = "res.png", width = 16)

        Sys.sleep(30)
    }
}
