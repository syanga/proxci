import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils._param_validation import Interval, StrOptions
from numbers import Integral, Real
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import GridSearchCV


class Test(BaseEstimator):
    def __init__(self, a=1, b=2):
        self.a = a
        self.b = b

    def fit(self, data):
        self.X_ = data[0]["X"]
        self.y_ = data[0]["y"]

        return self

    def predict(self, data):
        return self.y_ + data[0]["y"]

    def score(self, data):
        return np.mean(self.predict(data) == data[0]["y"])


if __name__ == "__main__":
    t = Test()

    data = [{"X": np.random.randn(), "y": np.random.randn()} for _ in range(100)]
    search = GridSearchCV(Test(), {"a": [1, 2, 3], "b": [4, 5, 6]})
    search.fit(data)
