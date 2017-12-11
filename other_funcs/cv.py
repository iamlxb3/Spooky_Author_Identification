import math

def create_k_fold(X, k_fold=5):
    X_len = len(X)
    one_fold_len = int(math.floor(X_len/k_fold))
    last_fold_len = X_len - (k_fold-1)*one_fold_len
    for i in range(k_fold):
        if i == k_fold - 1:
            X_test = X[X_len-last_fold_len:]
            X_train = X[:X_len-last_fold_len]
        else:
            X_test = X[i * one_fold_len:(i + 1) * one_fold_len]
            X_train = X[0: i * one_fold_len] + X[(i + 1) * one_fold_len:]
        yield X_train, X_test


if __name__ == '__main__':
    X = list(range(100))
    for X_train, X_test in create_k_fold(X, k_fold=10):
        print ("X_train: ", X_train)
        print ("X_test: ", X_test)



