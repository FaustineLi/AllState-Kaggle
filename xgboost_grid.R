# grid search on max_depth and min_child_weight
grid_max_depth = seq(3,10)
grid_child_weight = seq(1,5)
grid_result = data.frame()

for (max_depth in grid_max_depth) {

    for (child_weight in grid_child_weight) {

        xgb_params = list(
            seed = 0,
            colsample_bytree = 0.8,
            subsample = 0.8,
            eta = 0.1,
            objective = 'reg:linear',
            max_depth = max_depth,
            min_child_weight = child_weight,
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
            maximize = FALSE)

        row = c(res$params$max_depth,
                res$params$min_child_weight,
                res$best_iteration,
                res$evaluation_log[res$best_iteration]$test_error_mean,
                res$evaluation_log[res$best_iteration]$test_error_std)

        grid_result = rbind(grid_result, row)

    }
}

grid_result = setNames(grid_result, c('max_depth', 'child_weight',
                                      'niter', 'cv_error', 'cv_std'))
