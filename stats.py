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
