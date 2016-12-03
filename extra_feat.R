library('caret')

# import data
xtrain = readRDS('xtrain_lab.RDS')
ytrain = readRDS('ytrain.RDS')
xtest = readRDS('xtest_lab.RDS')
id = readRDS('testid.RDS')

# from xgboost, find most important features
importance = readRDS('feat_import.RDS')
catvar = importance$Feature[grepl('cat', importance$Feature)]
contvar = importance$Feature[grepl('cont', importance$Feature)]

# pick top 16 catagorical variables and encode interaction terms
cat_features = combn(catvar[1:16],2)
features = combn(importance$Feature[1:20],2)

# interaction for top 16 features
for (i in seq_len(ncol(cat_features))) {
    feat1 = cat_features[1,i]
    feat2 = cat_features[2,i]
    new_feat_name = paste0(feat1, feat2)

    new_feat = as.matrix(xtrain[,feat1] + xtrain[,feat2])
    colnames(new_feat) =  new_feat_name
    xtrain = cbind(xtrain, new_feat)

    new_feat = as.matrix(xtest[,feat1] + xtest[,feat2])
    colnames(new_feat) =  new_feat_name
    xtest = cbind(xtest, new_feat)
}

# remove unused features
importance_extra = readRDS('extra_feat_import.RDS')
used_features = importance_extra$Feature
xtrain = subset(xtrain, select=used_features)
xtest  = subset(xtest,  select=used_features)

# scale and unskew continuous features
for (j in contvar) {
    # unskew using boxcox transformation
    boxcox_model = BoxCoxTrans(xtrain[,j])
    xtrain[,j] = predict(boxcox_model, xtrain[,j])
    xtest[,j]  = predict(boxcox_model, xtest[,j])

    # scale data
    mean = mean(xtrain[,j])
    sd = sd(xtrain[,j])
    xtrain[,j] = (xtrain[,j] - mean) / sd
    xtest[,j] = (xtest[,j] - mean) / sd
}
