from __future__ import (
    division, print_function, absolute_import, unicode_literals)

import numpy as np
from corner import corner


def easy_corner(chain, range=None, bins=50, smooth=2,
                quantiles=[0.16, 0.50, 0.84], show_titles=True,
                **kwargs):
    """
    `~corner.corner` function with some pre-configued parameters.

    Parameters
    ----------
    chain : array_like[nsamples, ndim]
        The samples to be visualized.
    range : iterable (ndim,)
        default: [0.95] * ndim
    bins : int or array_like[ndim,]
        default: 50
    smooth : float
        default: 2
    quantiles : iterable
        default: [0.16, 0.50, 0.84]
    show_titles : bool
        default: True
    **kwargs
        Other keyword arguments to send to `~corner.corner`

    Returns
    -------
    fig : `~matplotlib.figure.Figure` object
    """
    nsamples, ndim = np.shape(chain)
    if range is None:
        range = [0.95] * ndim
    return corner(chain, range=range, bins=bins, smooth=smooth,
                  quantiles=quantiles, show_titles=show_titles,
                  **kwargs)


