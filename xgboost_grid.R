grid_colsample = seq(0.3, 0.9, 0.1)
grid_subsample = seq(0.3, 0.9, 0.1)
grid_result = data.frame()

for (colsample in grid_colsample) {

    for (subsample in grid_subsample) {

        set.seed(0)
        xgb_params = list(
            seed = 0,
            colsample_bytree = colsample,
            subsample = subsample,
            eta = 0.1,
            objective = 'reg:linear',
            max_depth = 6,
            min_child_weight = 9,
            gamma = 0,
            alpha = 0,
            lambda = 1)

        res = xgb.cv(
            data = dtrain,
            params = xgb_params,
            nround = 1000,
            early_stopping_rounds = 20,
            nfold = 5,
            feval = xgb_mae,
            verbose = 0,
            maximize = FALSE)

        row = c(res$params$colsample_bytree,
                res$params$subsample,
                res$best_iteration,
                res$evaluation_log[res$best_iteration]$test_error_mean,
                res$evaluation_log[res$best_iteration]$test_error_std)

        grid_result = rbind(grid_result, row)

    }
}
