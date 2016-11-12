library('dplyr')
xtest  = read.csv('pml_test_features.csv')
train  = read.csv('pml_train.csv')
testid = xtest$id
ytrain = train[,ncol(train)]
xtrain = train[,c(-1, -ncol(train))]
xtest =  xtest[,-1]
rm(train)

# labeling factors using the union of the test and train
factor = which(sapply(xtrain, is.factor))
for (i in factor) {
    all_labs = union(levels(xtrain[,i]), levels(xtest[,i]))
    all_labs = sort_char(all_labs)
    setattr(xtrain[,i], 'levels', all_labs)
    setattr(xtest[,i], 'levels', all_labs)
}

# one hot encoding of train / test data
one_hot = sparse.model.matrix(~ .-1, data = xtrain,
                contrasts.arg = lapply(
                        xtrain[,sapply(xtrain, is.factor)],
                        contrasts, contrasts=FALSE
                    )
                )

one_hot_test = sparse.model.matrix(~ .-1, data = xtest,
                                   contrasts.arg = lapply(
                                       xtest[,sapply(xtest, is.factor)],
                                       contrasts, contrasts=FALSE
                                   )
)

# numeric label encoding of train / test data
label_map = function(char) {
    char_list = unlist(strsplit(char, ''))
    num_char = sapply(char_list, function(x) which(LETTERS == x))
    base = 26^(rev(seq_along(char_list)) - 1)
    sum(base * num_char)
}
label_map = Vectorize(label_map)

# map a matrix of factors into integers
map_mat = function(mat) {
    ans = matrix(ncol = ncol(mat), nrow = nrow(mat))
    for (i in seq_len(ncol(mat))) {
        if (is.factor(mat[,i])) {
            ans[,i] = label_map(as.character(mat[,i]))
        } else {
            ans[,i] = mat[,i]
        }
    }
    ans
}

label_train = map_mat(xtrain)
saveRDS(label_train, file = 'xtrain_lab.RDS')
label_test = map_mat(xtest)
saveRDS(label_test, file = 'xtest_lab.RDS')
