library(tidyverse)

## Make a linear model using the columns from scaled_data
make_model <- function(var_bounds, init_data) {
    buf <- scaled_data(var_bounds, init_data)
    vars <- colnames(buf)
    f <- as.formula(paste("y ~", paste(vars, collapse="+")))
    lm(formula = f, data = buf %>% mutate(y = init_data$y))
}

## Returns a table with one col for each scaled variant of terms defined in var_bounds
## using data from df
scaled_data <- function(var_bounds, df) {
    n <- nrow(df)
    do.call(bind_rows, apply(var_bounds, 1, function(row) {
        v <- row[1]
        x0 <- as.numeric(row[2])
        x1 <- as.numeric(row[3])
        delta <- x1 - x0
        x <- df[[v]]
        tibble(
            obs_id = 1:n,
            index = v,
            term1 = sin(2 * pi * x / delta),
            term2 = sin(4 * pi * x / delta),
            term3 = cos(2 * pi * x / delta),
            term4 = cos(4 * pi * x / delta),
            )
    })) %>%
        pivot_wider(id_cols = obs_id, names_from = index, values_from = starts_with("term")) %>%
        select(starts_with("term"))
}

make_node <- function(var_bounds, output_range = c(-1, 1)) {
    n_vars <- nrow(var_bounds)
    n_deps <- sample(1:n_vars, 1, replace = F, prob = (1:n_vars)^(-1))
    n_deps <- max(n_deps, 1)
    deps <- sample_n(var_bounds, n_deps)
    vars <- deps$idx

    n_init <- 100
    init_data <- deps %>%
        select(-idx) %>%
        as.matrix() %>%
        apply(1, function(row) runif(n_init, row[1], row[2])) %>%
        as_tibble(.name_repair="minimal") %>%
        setNames(vars) %>%
        mutate(y = runif(n_init, output_range[1], output_range[2]))

    m <- make_model(deps, init_data)

    predictor <- function(model, variables) {
        input <- scaled_data(deps, as_tibble(variables))
        pred <- predict(model, newdata = input, interval = "prediction") ## Class: matrix
        rnorm(1, pred[1, 1], 0.2 * (pred[1,3] - pred[1, 2]))
    }

    list(
        model = m,
        deps = deps,
        predictor = predictor,
        data = init_data,
        feedback = NULL
    )
}

name_vars <- function(idx) paste("V", idx, sep="")

## var bounds format, idx should be integer
##     idx lower upper
##   <int> <dbl> <dbl>
## 1     1    -3     2
## 2     2    -3     2
build_network <- function(var_bounds, n_layers = 2, layer_width = 2) {
    var_bounds <- var_bounds %>% mutate(idx = name_vars(idx))
    nodes <- list()
    for (i in 1:n_layers) {
        bounds_buf <- tibble()
        for (j in 1:layer_width) {
            idx <- paste("node", i, j, sep=".")
            nodes[[idx]] <- make_node(var_bounds)
            bounds_buf <- bind_rows(
                bounds_buf,
                tibble(idx = idx, lower = -1, upper = 1)
            )
        }
        var_bounds <- bind_rows(var_bounds, bounds_buf)
    }

    idx <- "node.output"
    nodes[[idx]] <- make_node(var_bounds)

    list(
        nodes = nodes,
        output_idx = idx,
        memory = list(),
        feedback_level = NULL,
        age = 100
    )
}

## Give feedback to network
## data <- list(run_id => feedback)
## Feedback should be in (-1, 1)
give_feedback <- function(network, data) {
    for (i in names(network$nodes)) {
        node <- network$nodes[[i]]
        nvars <- nrow(node$deps)

        pos <- tibble()
        neg <- tibble()
        for (run_id in as.integer(names(data))) {
            feedback <- data[[run_id]] ## number
            variables <- network$memory[[run_id]] ## named list
            input <- as_tibble(variables[node$deps$idx]) ## tibble with vars as cols
            output <- variables[[i]] ## number

            ## TODO: correct to compare to mean feedback level?
            delta <- feedback - network$feedback_level
            buf <- tibble(y = output, w = abs(delta), input)
            if (delta > 0) {
                pos <- bind_rows(pos, buf)
            } else if (delta < 0) {
                neg <- bind_rows(neg, buf)
            }
        }

        ## Update model by adding memories as new data points
        new_model <- node$model
        print("Old: ")
        print(coef(new_model))
        if (nrow(pos) > 0) {
            print(paste("Positive feedback", nrow(pos)))
            old_data <- new_model$model %>% mutate(w = 1)
            new_model <- update(new_model, data = bind_rows(old_data, pos), weights = w)
        }
        if (nrow(neg) > 0) {
            print(paste("Negative feedback", nrow(neg)))
            ## Pretend to update model to incorporate the negative feedback points
            old_data <- new_model$model %>% mutate(w = 1)
            buf_model <- update(new_model, data = bind_rows(old_data, neg), weights = w)
            
            ## Move away from these coefficients!
            neg_coefs <- new_model$coefficients + (new_model$coefficients - buf_model$coefficients)
            ## TODO: fix.coef does not work, messes up model
            ## new_model <- fix.coef(buf_model, neg_coefs)
            new_model <- buf_model
            new_model$coefficients <- neg_coefs
        }

        ## TODO: how to update SE estimates? Add newdata to model$model?
        print("New: ")
        print(coef(new_model))
        
        network$nodes[[i]]$model <- new_model
    }

    network$age <- network$age + length(data)

    network
}

## input indexed vector
network_predict <- function(net, input, run_id) {
    variables <- setNames(as.list(input), name_vars(seq_along(input)))
    for (i in names(net$nodes)) {
        variables[[i]] <- net$nodes[[i]]$predictor(net$nodes[[i]]$model, variables)
    }

    net$memory[[run_id]] <- variables
    
    list(
        updated_network = net,
        output = as.numeric(tail(variables, 1))
    )
}

## TODO: Seems like no positive or negative feedback is generated, maybe run_id is missing?
run <- function() {
    var_bounds <- tibble(idx = 1, lower = 0, upper = 1)
    net <- build_network(var_bounds, 1, 1)
    
    a <- runif(1)
    b <- runif(1)
    for (rid in 1:10) {
        x <- runif(1)
        y <- a * x + b
        res <- network_predict(net, x, rid)
        net <- res$updated_network
        feedback <- list()
        feedback[[rid]] <- 1 / abs(res$output - y)
        net <- give_feedback(net, feedback)
        print(c(res$output - y, feedback[[rid]]))
    }
}
