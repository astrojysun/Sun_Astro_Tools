from __future__ import (
    division, print_function, absolute_import, unicode_literals)

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor


def interp_gpr(
        X_train, Y_train, X_pred, kernel=None, verbose=False,
        return_std=False, return_cov=False, return_regressor=False,
        **regressor_kwargs):
    """
    A light wrapper around GaussianProcessRegressor in sklearn.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        Feature vectors or other representations of training data.
    Y_Train : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values
    X_pred : array-like of shape (n_samples, n_features)
        Query points where the GP is evaluated.
    kernel : kernel instance, default=None
        The kernel specifying the covariance function of the GP.
    verbose : bool, default=False
        If True, information about the detailed steps taken inside this
        function will be displayed in the terminal.
    return_std : bool, default=False
        If True, the standard-deviation of the predictive distribution at
        the query points is returned along with the mean.
    return_cov : bool, default=False
        If True, the covariance of the joint predictive distribution at
        the query points is returned along with the mean.
    return_regressor : bool, default=False
        If True, the regressor object is returned along with the outputs
        from `regressor.predict()`
    **regressor_kwargs
        Other keyword arguments to be passed to `GaussianProcessRegressor`
    """
    if kernel is None:
        raise ValueError(
            "Please specify `kernel` for the Gaussian Process Regressor!"
            "\nSee following pages for instructions on kernel choice:\n"
            "https://scikit-learn.org/stable/auto_examples/"
            "gaussian_process/plot_gpr_noisy_targets.html\n"
            "https://scikit-learn.org/stable/modules/"
            "gaussian_process.html#kernels-for-gaussian-processes")

    if verbose:
        print("Training a GPR model...")
    gpr = GaussianProcessRegressor(
        kernel=kernel, **regressor_kwargs)
    gpr.fit(X_train, Y_train)

    if verbose:
        print(f"Model: {gpr.kernel_}")

    if verbose:
        print("Making predictions with the model...")
    results = gpr.predict(
        X_pred, return_std=return_std, return_cov=return_cov)

    if return_regressor:
        return results, gpr
    else:
        return results
