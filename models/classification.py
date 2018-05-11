import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
# import scipy.stats as sp
from sklearn.linear_model import LogisticRegression


class MeanPredictor(BaseEstimator, TransformerMixin):
    """docstring for MeanPredictor"""
    def __init__(self, classifier=None):
        self.classifier = classifier

    def fit(self, X, y):
        # self.mean = y.mean(axis=0)
        # processing y
        y_new = np.zeros(shape=(X.shape[0], 1))
        y = check_array(y)
        for i in range(X.shape[0]):
            b = y[i, :].split()
            b = list(map(float, b))
            y_new[i] = b.index(max(b))
        print(y_new)
        self.classifier = LogisticRegression(solver='newton-cg')
        # train your model
        self.classifier.fit(X, y)
        return self

    def predict_proba(self, X):
        check_array(X)
        check_is_fitted(self, ["Logistic"])
        # n_samples, _ = X.shape
        # return np.tile(self.mean, (n_samples, 1))
        return self.classifier.predict_proba(X)

    # def spearman(self, X, y_prob_true, sample_weight=None):
    #     res = 0
    #     for (i, j) in zip( y_true, y_pred ):
    #         ret_score = sp.spearmanr( i, j )[0]
    #         res += ret_score if not np.isnan( ret_score ) else 0.0
    #     return res / len( y_true )
    # GridSearchCV( ......., scoring=metrics.make_scorer( self.spearman ) )
