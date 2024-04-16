import numpy as np
from numba import njit


@njit(fastmath=True)
def kkt_solve(gram1, gram2, g1, g2, lambda1, lambda2):
    """
    The goal is to find kernel functions h(r1) (and function f(r2)) that solves:

    h = \argmin_{h} \max_f [
        \mathbb{E}(f(r2)*(h(r1)*g1 + g2) - f(r2)**2) 
        - lambda_2 * \|f\|_{RKHS}^2 + lambda_1 *\|h\|_{RKHS}^2
    ]

    The function h is in the RKHS of the metric specified by the kernel function
    with gram matrix gram1, and the function f is in the RKHS of the metric
    specified by the kernel function with gram matrix gram2.
    """
    n = gram1.shape[0]

    if np.linalg.matrix_rank(gram1) < n or np.linalg.matrix_rank(gram2) < n:
        # form full KKT system
        kkt_12 = gram1 @ (g1 * gram2).T
        kkt_22 = -2 * (gram2 @ gram2 + lambda2 * gram2)
        kkt_matrix = np.zeros((3 * n, 3 * n))
        kkt_matrix[:n, :n] = 2 * lambda1 * gram1
        kkt_matrix[:n, n:2*n] = kkt_12
        kkt_matrix[:n, 2*n:] = kkt_12
        kkt_matrix[n:2*n, :n] = kkt_12.T
        kkt_matrix[n:2*n, n:2*n] = kkt_22
        kkt_matrix[2*n:, 2*n:] = kkt_22

        kkt_vec = np.zeros(3 * n)
        kkt_vec[n:2*n] = -gram2 @ g2
    else:
        # form reduced KKT system
        kkt_matrix = np.zeros((2 * n, 2 * n))
        kkt_matrix[:n, n:] = (g1 * gram2).T
        kkt_matrix[n:, :n] = (g1 * gram1).T
        kkt_matrix[:n, :n] = -2 * gram2
        for i in range(n):
            kkt_matrix[i, i] += 2 * lambda1
            kkt_matrix[n + i, n + i] -= 2 * lambda2

        kkt_vec = np.zeros(2 * n)
        kkt_vec[n:] = -g2

    # solve the KKT system
    sol = np.linalg.solve(kkt_matrix, kkt_vec)
    alpha, beta = sol[:n], sol[n:2*n]
    return alpha, beta


@njit(fastmath=True)
def score_nuisance_function(h_values, gram2, g1, g2, lambda2):
    """
    Score a fitted nuisance function (function h) by solving the minimax problem
    with respect to the beta vector (function f), with the values h(r1) fixed.
    Returns a score (higher is better).
    """
    n = gram2.shape[0]
    if np.linalg.matrix_rank(gram2) < n:
        # form full KKT system
        kkt_matrix = 2 * (gram2 @ gram2 + lambda2 * gram2)
        kkt_vector = gram2 @ (g1 * h_values + g2)
        beta = np.linalg.solve(kkt_matrix, kkt_vector)
        f_values = beta @ gram2
    else:
        # reduced KKT system
        kkt_matrix = 2 * gram2
        for i in range(n):
            kkt_matrix[i, i] += 2 * lambda2
        kkt_vector = g1 * h_values + g2
        beta = np.linalg.solve(kkt_matrix, kkt_vector)
        f_values = beta @ gram2

    metric = np.mean((g1 * h_values + g2) * f_values - f_values**2)
    return -metric
