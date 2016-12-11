library(xgboost)

# result from best
best = round(res$best_iteration / 0.8)

# evaluation function
xgb_eval = function (yhat, dtrain) {
    y = getinfo(dtrain, 'label')
    err = mean(abs(exp(y) - exp(yhat)))
    return (list(metric = 'error', value = err))
}

# log cosh objective function
xgb_obj = function(yhat, dtrain) {
    y = getinfo(dtrain, 'label')
    grad = tanh(yhat - y)
    hess = 1 - grad * grad
    return(list(grad = grad, hess = hess))
}

# -------------------------------------------------

set.seed(0)

xgb_params = list(
    eta = 0.003,
    colsample_bytree = 0.4,
    subsample = 0.8,
    max_depth = 12,
    min_child_weight = 100,
    gamma = 1,
    alpha = 1,
    lambda = 0,
    base_score = 7,
    seed = 1,
    objective = xgb_obj
)


model1 = xgboost(dtrain,
                params = xgb_params,
                nrounds = best,
                feval = xgb_eval,
                maximize = FALSE
)

yhat1 = exp(predict(model1, xtest)) - shift

write.table(data.frame(id = id, loss = yhat1),
            file = 'submit1.csv', sep=',', row.names = FALSE)

#-------------------------------

set.seed(0)

xgb_params = list(
    eta = 0.003,
    colsample_bytree = 0.4,
    subsample = 0.8,
    max_depth = 12,
    min_child_weight = 100,
    gamma = 1,
    alpha = 1,
    lambda = 0,
    base_score = 7,
    seed = 2,
    objective = xgb_obj
)


model2 = xgboost(dtrain,
                params = xgb_params,
                nrounds = best,
                feval = xgb_eval,
                maximize = FALSE
)

yhat2 = exp(predict(model2, xtest)) - shift

write.table(data.frame(id = id, loss = yhat2),
            file = 'submit2.csv', sep=',', row.names = FALSE)

# ------------------------------------

set.seed(0)

xgb_params = list(
    eta = 0.003,
    colsample_bytree = 0.4,
    subsample = 0.8,
    max_depth = 12,
    min_child_weight = 100,
    gamma = 1,
    alpha = 1,
    lambda = 0,
    base_score = 7,
    seed = 3,
    objective = xgb_obj
)


model3 = xgboost(dtrain,
                params = xgb_params,
                nrounds = best,
                feval = xgb_eval,
                maximize = FALSE
)

yhat3 = exp(predict(model3, xtest)) - shift

write.table(data.frame(id = id, loss = yhat3),
            file = 'submit3.csv', sep=',', row.names = FALSE)

# ------------------------------------------

yhat_ensemble = rowMeans(cbind(yhat1, yhat2, yhat3))

write.table(data.frame(id = id, loss = yhat_ensemble),
            file = 'submit.csv', sep=',', row.names = FALSE)
