from .proxci_dataset import *
from .minimax import *


class ProximalInference:
    def __init__(
        self,
        proxci_dataset,
        crossfit_folds=1,
        lambdas=None,
        gammas=None,
        cv=2,
        n_jobs=1,
        verbose=0,
        print_best_params=False,
    ):
        assert crossfit_folds >= 1
        self.crossfit_folds = crossfit_folds
        self.data = proxci_dataset

        # minimax and gridsearch parameters
        self.hdim = self.data.r1.shape[1]
        self.fdim = self.data.r2.shape[1]
        self.lambdas = lambdas
        self.gammas = gammas
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.print_best_params = print_best_params

        # if doing cross-fitting, split data into fit and eval sets
        self.cf_inds = proxci_dataset.create_crossfit_split(crossfit_folds)

        # generate ensemble of h and q functions for cross-fitting
        self.h = [
            {a: self.estimate_h(a, fold=i) for a in range(2)}
            for i in range(crossfit_folds)
        ]
        self.q = [
            {a: self.estimate_q(a, fold=i) for a in range(2)}
            for i in range(crossfit_folds)
        ]

    def estimate_h(self, a, fold=0):
        """Estimate nuisance function h"""
        g1 = -1 * (self.data.A == a)
        g2 = self.data.Y * (self.data.A == a)
        data = join_data(self.data.r1, self.data.r2, g1, g2)[
            self.cf_inds[fold]["train"]
        ]
        search = MinimaxRKHSCV(
            self.hdim,
            self.fdim,
            lambdas=self.lambdas,
            gammas=self.gammas,
            cv=self.cv,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )
        search.fit(data)
        if self.print_best_params > 0:
            print("h, a=%d" % a, search.best_params_)
        return search.best_estimator_.h_

    def estimate_q(self, a, fold=0):
        """Estimate nuisance function q"""
        g1 = -1 * (self.data.A == a)
        g2 = np.ones(len(self.data.Y))
        data = join_data(self.data.r2, self.data.r1, g1, g2)[
            self.cf_inds[fold]["train"]
        ]
        search = MinimaxRKHSCV(
            self.hdim,
            self.fdim,
            lambdas=self.lambdas,
            gammas=self.gammas,
            cv=self.cv,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )
        search.fit(data)
        if self.print_best_params > 0:
            print("q, a=%d" % a, search.best_params_)
        return search.best_estimator_.h_

    def por(self):
        """Estimator based on function h"""
        est = 0.0
        for fold in range(self.crossfit_folds):
            r = self.data.r1[self.cf_inds[fold]["eval"]]
            est += np.mean(self.h[fold][1](r)) - np.mean(self.h[fold][0](r))
        return est / self.crossfit_folds

    def pipw(self):
        """Estimator based on function q"""
        est = 0.0
        for fold in range(self.crossfit_folds):
            fold_idx = self.cf_inds[fold]["eval"]
            I0 = self.data.A[fold_idx] == 0
            I1 = self.data.A[fold_idx] == 1
            r = self.data.r2[fold_idx]
            y = self.data.Y[fold_idx]
            est += np.mean(I1 * y * self.q[fold][1](r)) - np.mean(
                I0 * y * self.q[fold][0](r)
            )
        return est / self.crossfit_folds

    def dr(self):
        """Doubly robust estimator"""
        est = 0.0
        for fold in range(self.crossfit_folds):
            fold_idx = self.cf_inds[fold]["eval"]
            I0 = self.data.A[fold_idx] == 0
            I1 = self.data.A[fold_idx] == 1
            r1 = self.data.r1[fold_idx]
            r2 = self.data.r2[fold_idx]
            y = self.data.Y[fold_idx]

            h0 = self.h[fold][0](r1)
            h1 = self.h[fold][1](r1)
            q0 = self.q[fold][0](r2)
            q1 = self.q[fold][1](r2)
            est += np.mean(I1 * (y - h1) * q1 + h1) - np.mean(I0 * (y - h0) * q0 + h0)
        return est / self.crossfit_folds
