grid_colsample = seq(0.3, 0.9, 0.1)
grid_subsample = seq(0.3, 0.9, 0.1)
grid_result = data.frame()

for (colsample in grid_colsample) {

    for (subsample in grid_subsample) {

        set.seed(0)
        xgb_params = list(
            eta = 0.1,
            colsample_bytree = colsample,
            subsample = subsample,
            max_depth = 12,
            min_child_weight = 100,
            gamma = 1,
            alpha = 1,
            lambda = 0,
            base_score = 7,
            seed = 0,
            objective = xgb_obj
        )
      
        row = c(res$params$colsample_bytree,
                res$params$subsample,
                res$best_iteration,
                res$evaluation_log[res$best_iteration]$test_error_mean,
                res$evaluation_log[res$best_iteration]$test_error_std
                )

        grid_result = rbind(grid_result, row)

    }
}
