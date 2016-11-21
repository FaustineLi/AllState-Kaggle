library('xgboost')

xtrain = readRDS('xtrain_lab.RDS')
xtest = readRDS('xtest_lab.RDS')
id = readRDS('testid.RDS')
ytrain = readRDS('ytrain.RDS')

shift = 200
dtrain = xgb.DMatrix(as.matrix(xtrain), label=data.matrix(log(ytrain + shift)))
dtest =  xgb.DMatrix(as.matrix(xtest))

# mae evaluation function
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

#-------------------------------------------------------------------

set.seed(0)
xgb_params = list(
    eta = 0.01,
    colsample_bytree = 0.5,
    subsample = 0.8,
    max_depth = 12,
    min_child_weight = 100,
    gamma = 1,
    alpha = 1,
    lambda = 0,
    base_score = 7,
    seed = 0,
    objective = xgb_obj
    )

res = xgb.cv(data = dtrain,
             params = xgb_params,
             nround = 100000,
             nfold = 5,
             early_stopping_rounds = 50,
             feval = xgb_eval,
             maximize = FALSE)

# ------------------------------------------------------------------
set.seed(0)
best = round(res$best_iteration)
model = xgboost(dtrain,
                params = xgb_params,
                nrounds = best,
                feval = xgb_eval,
                maximize = FALSE)

yhat = exp(predict(model, xtest)) - shift

write.table(data.frame(id = id, loss = yhat),
            file = 'submit.csv', sep=',', row.names = FALSE)
