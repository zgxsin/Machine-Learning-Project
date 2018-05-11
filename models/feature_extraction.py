from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
import numpy as np
# import pandas as pd


class Histog(BaseEstimator, TransformerMixin):
    def __init__(self, block=10, bins_value=100):
        self.block = block
        self.bins_value = bins_value

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = check_array(X)
        index1 = X.shape[0]
        X = X.reshape(index1, 176, 208, 176)
        X = X[:, 10:160, 30:180, 10:160]
        n = self.block  # n*n*n blocks
        slice1 = round(150 / n)
        A = np.zeros(shape=(index1, n ** 3, slice1, slice1, slice1))

        for i in range(index1):
            num = 0
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        A[i, num, :, :, :] = X[
                                             i, slice1 * j: slice1 * (j + 1),
                                             slice1 * k: slice1 * (k + 1),
                                             slice1 * l: slice1 * (l + 1)]
                        num = num + 1
        bins_value1 = self.bins_value
        X_new = np.zeros(shape=(index1, n ** 3 * bins_value1))

        for i in range(index1):
            for h in range(n ** 3):
                hist, bins = np.histogram(
                             A[i, h, :, :, :].flatten(),
                             bins=bins_value1,
                             range=(10, 2000))
                X_new[i, bins_value1 * h:bins_value1 * (h + 1)] = hist

        return X_new

# READ_ME
# -----------------------------------------------
# -----------------------------------------------
# I process y into a single label array. This is
# how I do this. First I assign 1 2 3 4 to the probability
# according to the probability(biggest probability with 4).
# Since y is a 290*4 array, for each row of y, I join the element
# together so that y is a 290*1 array. Then each element of y can
# be regarded as a 'label' for the corresponding sample.
# After preprocessing X, I use SVM to train the model.
# With the model, I use 'predict' command to predict future
# labels with future data. SVM can give me the class label for
# each future data. However, the output prediction file is not
# consistent with the requirement of Kaggle. So I just split the
# y in the prediction file to form a 162*4 array and submit the
# transformed one to Kaggle.

# ----------------------------
# code for preprocessing y
#     def fit(self, X, y):
#         y = check_array(y)
#         y_new = np.zeros(shape=(X.shape[0], 1), dtype=int)
#         # print(y.shape)
#         # print(X.shape)
#         for i in range( X.shape[0]):
#             b_order = np.argsort(y[i])
#             a = np.array( [1, 2, 3, 4], dtype=int )
#             for j in range(4):
#                 y[i][b_order[j]] = a[j]
#                 temp = y[i].astype(int)
#             # convert [1,2,3,4] to ['1234']
#             y_new[i] = ''.join(map(str, temp))
#             # convert['1234'] to [1234]
#         # y_new = [int( numeric_string ) for numeric_string in y_new]
#         print(y_new)
#         np.savetxt( "data/y_new.csv", y_new, delimiter=",")
