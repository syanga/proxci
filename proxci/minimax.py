import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils._param_validation import Interval
from numbers import Integral, Real
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import GridSearchCV
import warnings

from .proxci_dataset import *
from .minimax_core import *


def MinimaxRKHSCV(hdim, fdim, lambdas=None, gammas=None, cv=2, n_jobs=1, verbose=0):
    """MinimaxRKHS with hyperparameter tuning via gridsearch"""
    # set defaults for hyperparameter tuning
    if lambdas is None:
        lambdas = [10**i for i in range(-5, 1, 1)]
    if gammas is None:
        gammas = [1]

    args = {"lambda_h": lambdas, "lambda_f": lambdas, "gamma": gammas}
    return GridSearchCV(
        MinimaxRKHS(hdim, fdim, n_jobs=n_jobs),
        args,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
    )


class MinimaxRKHS(BaseEstimator):

    """Base class for minimax RKHS estimators

    The goal is to find a function h(r1) (along with function f(r2)) that solves

    h = \argmin_{h} \max_f [\mathbb{E}(f(r2)*(h(r1)*g1 + g2) - f(r2)**2) - (lambda_f/n^0.8) *\|f\|_{RKHS}^2 + (lambda_h//n^0.8) *\|h\|_{RKHS}^2]

    where h and f are functions in the RKHS of the metric specified by their kernel functions.
    """

    _parameter_constraints: dict = {
        "lambda_h": [Interval(Real, 0.0, None, closed="left")],
        "lambda_hf": [Interval(Real, 0.0, None, closed="left")],
        "lambda_q": [Interval(Real, 0.0, None, closed="left")],
        "lambda_qf": [Interval(Real, 0.0, None, closed="left")],
        "gamma": [Interval(Real, 0.0, None, closed="neither")],
        "n_jobs": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(
        self,
        hdim,
        fdim,
        *,
        n_jobs=1,
        lambda_h=1,
        lambda_f=1,
        gamma=1,
    ):
        # variable dimensions
        self.hdim = hdim
        self.fdim = fdim

        # hyperparameters
        self.lambda_h = lambda_h
        self.lambda_f = lambda_f

        # kernel parameters
        self.gamma = gamma
        self.n_jobs = n_jobs

    def _create_kernel(self, X, Y=None):
        return pairwise_kernels(
            X, Y, metric="rbf", gamma=self.gamma, n_jobs=self.n_jobs
        )

    def _unpack_data(self, data):
        r1 = data[:, : self.hdim]
        r2 = data[:, self.hdim : self.hdim + self.fdim]
        g1 = data[:, self.hdim + self.fdim]
        g2 = data[:, self.hdim + self.fdim + 1]
        return r1, r2, g1, g2

    def fit(self, data):
        """
        # data contains r1, r2, g1, g2
        # fit alpha_, beta_, coefficients for h and f respectively
        """
        r1, r2, g1, g2 = self._unpack_data(data)

        # compute rkhs basis
        self.Kh_ = self._create_kernel(r1)
        self.Kf_ = self._create_kernel(r2)

        # solve minimax problem
        self.alpha_, self.beta_ = kkt_solve(
            self.Kh_, self.Kf_, g1, g2.copy(), self.lambda_h, self.lambda_f)

        # fitted h function
        self.r1_ = r1
        self.h_ = lambda r: self.alpha_ @ self._create_kernel(self.r1_, r)

        # fitted f function
        self.r2_ = r2
        self.f_ = lambda r: self.beta_ @ self._create_kernel(self.r2_, r)

        return self

    def predict(self, data):
        """Evaluate fitted function h at data points"""
        check_is_fitted(self)
        r1, _, _, _ = self._unpack_data(data)
        return self.h_(r1)

    def score(self, data):
        """Score used for hyperparameter tuning"""
        r1, r2, g1, g2 = self._unpack_data(data)

        # fix h, solve max E[f(h*g1 + g2) - f^2] - lambda_f|f|^2
        h_val = self.h_(r1)
        Kf_val = self._create_kernel(r2)

        return score_nuisance_function(h_val, Kf_val, g1, g2.copy(), self.lambda_f)
