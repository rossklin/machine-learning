library(dplyr)
library(readr)
library(ggplot2)

run <- function() {
    df <- read_csv("simple-recursive-brain/output.csv") %>% 
        mutate(test_id = as.character(test_id))

    ggplot(df, aes(x = iteration, y = test, group = test_id, color = test_id)) + 
        geom_smooth(se = F)
}
