import numpy as np


def confidence_interval(point_estimators, true_value, width=1.96):
    mu = np.mean(point_estimators)
    sigma = np.std(point_estimators)
    ci_width = (width / np.sqrt(len(point_estimators))) * sigma
    upper_bound = mu + ci_width
    lower_bound = mu - ci_width
    return (lower_bound, upper_bound)


def test_coverage(point_estimators, true_value, width=1.96):
    """check if true value is covered by point estimates. By default, with 95% confidence"""
    lb, ub = confidence_interval(point_estimators, true_value, width=1.96)
    return true_value >= lb and true_value <= ub


def estimate_bias(point_estimators, true_value):
    """calculate bias of point estimators"""
    return np.mean(point_estimators) - true_value


def mse(point_estimators, true_value):
    """calculate MSE of point estimators"""
    return np.mean((point_estimators - true_value) ** 2)
