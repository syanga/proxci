import numpy as np
from scipy.special import expit
from proxci.proxci_dataset import ProxCIData


""" 2d dataset from Cui 2020 / Ghassami 2021 """


def generate_data(n):
    # generate X
    Gammax = np.array([0.25, 0.25])
    Sigmax = np.diag([0.25, 0.25]) ** 2
    X = np.random.multivariate_normal(Gammax, Sigmax, size=n)

    # generate A|X
    p = expit(-X.dot(np.array([0.125, 0.125])))
    A = np.random.binomial(1, p)

    # generate Z,W,U|A,X
    alpha0 = 0.25
    alphaa = 0.125
    alphax = np.array([0.25, 0.25])

    mu0 = 0.25
    mua = 0.25
    mux = np.array([0.25, 0.25])

    kappa0 = 0.25
    kappaa = 0.25
    kappax = np.array([0.25, 0.25])

    Mean = np.vstack(
        [
            alpha0 + alphaa * A + X.dot(alphax),
            mu0 + mua * A + X.dot(mux),
            kappa0 + kappaa * A + X.dot(kappax),
        ]
    ).T
    Sigma = np.array([[1, 0.25, 0.5], [0.25, 1, 0.5], [0.5, 0.5, 1]])
    ZWU = np.vstack([np.random.multivariate_normal(Mean[i], Sigma) for i in range(n)])
    Z, W, U = ZWU[:, 0], ZWU[:, 1], ZWU[:, 2]

    # generate E[Y|Z,W,U,A,X]
    b0 = 2
    ba = 2

    bx = np.array([0.25, 0.25])
    bomega = 4
    omega = 2

    EW = mu0 + X.dot(mux) + (Sigma[2, 1] / Sigma[2, 2]) * (U - kappa0 - X.dot(kappax))
    EY = b0 + ba * A + X.dot(bx) + omega * W + (bomega - omega) * EW
    Y = np.random.normal(EY, np.sqrt(0.25))

    target = ba
    return ProxCIData(X, Z, W, U, Y, A), target
