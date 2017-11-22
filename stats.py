from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np


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
                np.sqrt((2*np.pi)**ndim *
                        np.linalg.det(cov_matrix)) *
                norm)
    else:
        return (-0.5 * (distsq + np.log(2*np.pi)*ndim +
                        np.log(np.linalg.det(cov_matrix))) +
                np.log(norm))
