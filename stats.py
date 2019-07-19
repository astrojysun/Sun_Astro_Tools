from __future__ import (
    division, print_function, absolute_import, unicode_literals)

import numpy as np
from scipy.stats import binned_statistic


def running_mean(x, y, xbins):
    """
    Compute the mean value of y for all data points in each x-bin.

    This is a simple wrapper around `scipy.stats.binned_statistic`,
    tuned to handle NaN values.

    Parameters
    ----------
    x : array_like
        x values of all data points
    y : array_like
        y values of all data points
    xbins : array_like
        Bin edges, including the rightmost edge

    Returns
    -------
    means : numpy array
        Mean values of y in every x-bin.
    """
    return binned_statistic(
        np.atleast_1d(x).ravel(), np.atleast_1d(y).ravel(),
        bins=xbins, statistic=np.nanmean)[0]


def running_percentile(x, y, xbins, q):
    """
    Compute the qth percentile of y for all data points in each x-bin.

    Parameters
    ----------
    x : array_like
        x values of all data points
    y : array_like
        y values of all data points
    xbins : array_like
        Bin edges, including the rightmost edge
    q : float in range of [0,100] (or sequence of floats)
        Percentile to compute, to be passed along to `np.percentile`

    Returns
    -------
    percentiles : ndarray
        0th axis corresponds to the percentiles,
        1st axis corresponds to the bins.
    """
    nbin = len(xbins) - 1
    percentiles = np.full([np.atleast_1d(q).size, nbin], np.nan)
    for ibin in range(nbin):
        mask = ((x >= xbins[ibin]) & (x < xbins[ibin + 1]) &
                np.isfinite(y))
        if mask.sum() == 0:
            continue
        yinbin = y[mask]
        percentiles[:, ibin] = np.percentile(yinbin, q)
    return percentiles


def OLS_fitter(x, y, method='OLS (Y|X)', weight=None):
    """
    Simple ordinary least square (OLS) fitter.

    This regression method should only be used when the measurement
    uncertainties on both X and Y are negligible compared to the
    intrinsic scatter about the true, linear relation between them.

    This function supports multiple OLS regression methods:
    - 'OLS (Y|X)':
        The standard OLS regression of Y on X (i.e., treating X as the
        independent variable)
    - 'OLS (X|Y)':
        OLS regression of X on Y (i.e., treating Y as the independent
        variable)
    - 'OLS bisector':
        The bisector of the OLS (Y|X) line and the OLS (X|Y) line

    Parameters
    ----------
    x : array_like
        An array of x values
    y : array_like
        An array of y values, of the sample length as x.
    method : {'OLS (Y|X)', 'OLS (X|Y)', 'OLS bisector'}, optional
        OLS regression methods (see description above)
        Default is 'OLS (Y|X)'
    weight : array_like, optional
        An array of weights, of the same length as x.

    Returns
    -------
    beta : float
        Slope of the OLS regression line
    alpha : float
        Intercept of the OLS regression line
    """
    if ~(np.isfinite(x).all()) or ~(np.isfinite(y).all()):
        raise ValueError("Input variables contain NaN or Inf")
    if len(x) != len(y):
        raise ValueError("Input variables have different lengths")
    if weight is None:
        w = np.ones_like(x).astype('float')
        w_tot = w.sum()
    else:
        w = weight.reshape(x.ravel().shape)
        w_tot = w.sum()
        if ~(np.isfinite(w).all()) or ~((w >= 0).all()) or w_tot == 0:
            raise ValueError(
                "Weight contains NaN, Inf, negative value, "
                "or all zeros")
    x_mean = (x.ravel() * w).sum() / w_tot
    y_mean = (y.ravel() * w).sum() / w_tot
    S_matrix = np.cov(
        x.ravel(), y.ravel(), ddof=0, aweights=w) * w_tot
    beta1 = S_matrix[0, 1] / S_matrix[0, 0]
    beta2 = S_matrix[1, 1] / S_matrix[0, 1]
    if method == 'OLS (Y|X)':
        beta = beta1
    elif method == 'OLS (X|Y)':
        beta = beta2
    elif method == 'OLS bisector':
        beta = ((beta1*beta2 - 1 +
                 np.sqrt((1+beta1**2)*(1+beta2**2))) /
                (beta1 + beta2))
    else:
        raise ValueError("Unknown fitting method: {}".format(method))
    alpha = y_mean - beta * x_mean
    return beta, alpha


def gaussNd_pdf(x, norm=1., mean=0., cov_matrix=None,
                return_log=False):
    """
    N-dimensional Gaussian PDF.

    Parameters
    ----------
    x : array_like[..., ndim]
        The independent variable.
    norm : scalar or array_like (optional)
        The normalization factor (i.e. 0th moment). Default: 1.0
        Shape should be compatible (broadcastable) with `x`.
    mean : scalar or array_like (optional)
        The position of the peak (i.e. 1st moment). Default: 0.0
        Shape should be compatible (broadcastable) with `x`.
    cov_matrix : array_like[..., ndim, ndim]
        Covariance matrix of the ND independent variable.
        Array shape requirement:
        cov_matrix.shape[-1] and cov_matrix.shape[-2] should both
        equal x.shape[-1], and cov_matrix.shape[:-2] should be
        compatible (broadcastable) with x.shape[:-1].
    return_log : bool (optional)
        If set to True, return the natural log of the function values.
        This is particularly useful for calculating log likelihood.

    Returns
    -------
    y : array_like
        The function value at the positions of `x`.
        Shape will be the broadcasted shape from `x`, `norm`, `mean`
        and `var`.

    See Also
    --------
    ~scipy.stats.multivariate_normal.pdf()
    ~scipy.stats.multivariate_normal.logpdf()
    """
    if cov_matrix is None:
        raise ValueError("No `cov_matrix` specified.")
    distsq = np.einsum('...i,...ij,...j',
                       x - mean,
                       np.linalg.inv(cov_matrix),
                       x - mean)  # square of the Mahalanobis distance
    ndim = x.shape[-1]
    if not return_log:
        return (np.exp(-0.5 * distsq) /
                np.sqrt((2 * np.pi) ** ndim *
                        np.linalg.det(cov_matrix)) *
                norm)
    else:
        return (-0.5 * (distsq + np.log(2 * np.pi) * ndim +
                        np.log(np.linalg.det(cov_matrix))) +
                np.log(norm))
