# import data
xtest  = read.csv('data//pml_test.csv')
train  = read.csv('data//pml_train.csv')
testid = xtest$id
ytrain = train[,ncol(train)]
xtrain = train[,c(-1, -ncol(train))]
xtest =  xtest[,-1]
names = colnames(xtrain)

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

# save train data
label_train = map_mat(xtrain)
colnames(label_train) = names
saveRDS(label_train, file = 'xtrain.RDS')

label_test = map_mat(xtest)
colnames(label_test) = names
saveRDS(label_test, file = 'xtest.RDS')
