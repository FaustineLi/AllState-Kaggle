library('xgboost')

xtrain = readRDS('xtrain_lab.RDS')
xtest = readRDS('xtest_lab.RDS')
id = readRDS('testid.RDS')
ytrain = readRDS('ytrain.RDS')

dtrain = xgb.DMatrix(as.matrix(xtrain), label=data.matrix(log(ytrain)))
dtest =  xgb.DMatrix(as.matrix(xtest))

# mae function
xgb_mae = function (yhat, dtrain) {
    y = getinfo(dtrain, 'label')
    err = mean(abs(exp(y) - exp(yhat)))
    return (list(metric = 'error', value = err))
}

#-------------------------------------------------------------------

xgb_params = list(
    seed = 0,
    colsample_bytree = 0.7,
    subsample = 0.7,
    eta = 0.075,
    objective = 'reg:linear',
    max_depth = 6,
    num_parallel_tree = 1,
    min_child_weight = 1,
    base_score = 7
)

res = xgb.cv(data = dtrain,
             params = xgb_params,
             nround=500,
             early_stopping_rounds = 20,
             nfold = 4,
             feval = xgb_mae,
             maximize = FALSE)

# ------------------------------------------------------------------
best = res$best_iteration
model = xgboost(xtrain,
                data.matrix(log(ytrain)),
                params = xgb_params,
                nrounds = best,
                feval = xgb_mae,
                maximize = FALSE)

yhat = exp(predict(model, xtest))

write.table(data.frame(id = id, loss = yhat),
            file = 'submit.csv', sep=',', row.names = FALSE)
