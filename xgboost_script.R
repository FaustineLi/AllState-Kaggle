library('xgboost')

xtrain = readRDS('xtrain_lab.RDS')
xtest = readRDS('xtest_lab.RDS')
id = readRDS('testid.RDS')
ytrain = readRDS('ytrain.RDS')

dtrain = xgb.DMatrix(as.matrix(xtrain), label=data.matrix(log(ytrain + 200)))
dtest =  xgb.DMatrix(as.matrix(xtest))

# mae function
xgb_eval = function (yhat, dtrain) {
    y = getinfo(dtrain, 'label')
    err = mean(abs(exp(y) - exp(yhat)))
    return (list(metric = 'error', value = err))
}

xgb_obj = function(yhat, dtrain) {
    y = getinfo(dtrain, 'label')
    grad = tanh(yhat - y)
    hess = 1 - grad * grad
    return(list(grad = grad, hess = hess))
}

#-------------------------------------------------------------------

set.seed(0)
xgb_params = list(
    seed = 0,
    colsample_bytree = 0.4,
    subsample = 0.9,
    eta = 0.001,
    objective = xgb_obj,
    max_depth = 6,
    min_child_weight = 9,
    gamma = 0,
    alpha = 0,
    lambda = 1,
    base_score = 7)

res = xgb.cv(data = dtrain,
             params = xgb_params,
             nround = 5000,
             nfold = 5,
             early_stopping_rounds = 20,
             feval = xgb_mae,
             maximize = FALSE)

# ------------------------------------------------------------------
set.seed(0)
best = res$best_iteration
model = xgboost(xtrain,
                data.matrix(log(ytrain)),
                params = xgb_params,
                nrounds = best,
                feval = xgb_eval,
                maximize = FALSE)

yhat = exp(predict(model, xtest))

write.table(data.frame(id = id, loss = yhat),
            file = 'submit.csv', sep=',', row.names = FALSE)
