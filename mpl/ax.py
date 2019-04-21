from __future__ import (
    division, print_function, absolute_import, unicode_literals)

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib as mpl
import matplotlib.pyplot as plt


def log_contourplot(
        x, y, weight_by=None, xlim=None, ylim=None,
        overscan=(0.1, 0.1), logbin=(0.02, 0.02), smooth_nbin=(3, 3),
        levels=(0., 0.393, 0.865, 0.989), alphas=(0.75, 0.50, 0.25),
        color='k', ax=None, **contourfkw):
    """
    Generate data density contours in log-log space.

    Parameters
    ----------
    x, y : array_like
        x & y coordiantes of the data points
    weight_by : {None, 'x', 'y', function object}, optional
        Weight applied to the data density in each x-y bin.
        Uniform weight (weight_by=None) is used by default.
        Other options include weighting by 'x', 'y', or a
        function that takes x & y as arguments and calculate
        the weight at each x-y location.
    xlim, ylim : array_like, optional
        Range to calculate and generate contour.
        Default is to use a range wider than the data range
        by a factor of F on both sides, where F is specified by
        the keyword 'overscan'.
    overscan : array_like (2 elements), optional
        Factor by which 'xlim' and 'ylim' are wider than
        the data range on both sides. Default is 0.1 dex wider,
        meaning that xlim = (Min(x) / 10**0.1, Max(x) * 10**0.1),
        and the same case for ylim.
    logbin : array_like (2 elements), optional
        Bin widths (in dex) used for generating the 2D histogram.
        Usually the default value (0.02 dex) is enough, but it
        might need to be higher for complex distribution shape.
    smooth_nbin : array_like (2 elements), optional
        Number of bins to smooth over along x & y direction.
        To be passed to `~scipy.ndimage.gaussian_filter`
    levels : array_like, optional
        Contour levels to be plotted, specified as levels in CDF.
        By default levels=(0., 0.393, 0.865, 0.989), which corresponds
        to the integral of a 2D normal distribution within 1-sigma,
        2-sigma, and 3-sigma range (i.e., Mahalanobis distance).
        Note that for an N-level contour plot, 'levels' must have
        length=N+1, and its leading element must be 0.
    alphas : array_like, optional
        Transparancy of the contours. Default: (0.75, 0.50, 0.25)
    color : mpl color, optional
        Base color of the contours. Default: 'k'
    ax : `~matplotlib.axes.Axes` object, optional
        The Axes object to plot contours in.
    **contourfkw
        Keywords to be passed to `~matplotlib.pyplot.contourf`.

    Returns
    -------
    ax : `~matplotlib.axes.Axes` object
        The Axes object in which contours are plotted.
    """
    
    if xlim is None:
        xlim = (10**(np.nanmin(np.log10(x))-overscan[0]),
                10**(np.nanmax(np.log10(x))+overscan[0]))
    if ylim is None:
        ylim = (10**(np.nanmin(np.log10(y))-overscan[1]),
                10**(np.nanmax(np.log10(y))+overscan[1]))

    if ax is None:
        ax = plt.axes([0.02, 0.08, 0.96, 0.90])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    # force to change to log-log scale
    ax.set_xscale('log')
    ax.set_yscale('log')

    # generate 2D histogram
    lxedges = np.arange(
        np.log10(xlim)[0], np.log10(xlim)[1]+logbin[0], logbin[0])
    lyedges = np.arange(
        np.log10(ylim)[0], np.log10(ylim)[1]+logbin[1], logbin[1])
    hist, lxedges, lyedges = np.histogram2d(
        np.log10(x), np.log10(y), bins=[lxedges, lyedges])
    xmids = 10**(lxedges[:-1] + 0.5*logbin[0])
    ymids = 10**(lyedges[:-1] + 0.5*logbin[1])
    
    # weight 2D histogram
    if weight_by == 'x':
        hist *= xmids.reshape(-1, 1)
    elif weight_by == 'y':
        hist *= ymids
    elif weight_by is not None:
        hist *= weight_by(xmids.reshape(-1, 1), ymids)

    # smooth 2D histogram
    pdf = gaussian_filter(hist, smooth_nbin).T
    
    # calculate cumulative density distribution (CDF)
    cdf = np.zeros_like(pdf).ravel()
    for i, density in enumerate(pdf.ravel()):
        cdf[i] = pdf[pdf >= density].sum()
    cdf = (cdf/cdf.max()).reshape(pdf.shape)

    # plot contourf
    ax.contourf(
        xmids, ymids, cdf, levels,
        colors=[mpl.colors.to_rgba(color,a) for a in alphas],
        **contourfkw)
    
    return ax


def minimal_barplot(
        seq, percent=[16., 50., 84.], pos=None,
        colors=None, labels=None, labelloc='up', labelpad=0.1,
        ax=None, barkw={}, labelkw={}, **symkw):
    """
    Generate barplot, in the spirit of minimalism.

    Parameters
    ----------
    seq : sequence of array_like
        Samples to be represented by the bar plot.
    percent : array_like of floats between 0 and 100, optional
        Percentiles to compute, default: [16., 50., 84.]
    pos : array_like of floats, optional
        Positions at which to plot the bars for each sample.
        Default: np.arange(len(seq)) + 1
    colors : array_like of colors, optional
        Colors to use for each sample, default: 'k'
    labels : array_like of strings, optional
        Labels for each sample, default is no label.
    labelloc : {'up', 'down'}, optional
        Locations to put label relative to bar, default: 'up'
    labelpad : float, optional
        Pad (in data unit) between labels and bars, default: 0.1
    ax : `~matplotlib.axes.Axes` object, optional
        Overplot onto the provided axis object.
        If not available, a new axis will be created.
    barkw : dict, optional
        Keyword arguments to control the behavior of the bars.
        Will be passed to `~matplotlib.axes.Axes.hlines`.
    labelkw : dict, optional
        Keyword arguments to control the behavior of the labels.
        Will be passed to `~matplotlib.axes.Axes.text`.
    **symkw
        Keyword arguments to control the behavior of the symbols.
        Will be passed to `~matplotlib.axes.Axes.plot`.

    Examples
    --------
    >>> seq = [np.random.randn(1000) + i for i in range(5)]
    >>> ax = minimal_barplot(seq,
    >>>                      percent=[5, 16, 25, 50, 75, 84, 95],
    >>>                      labels=['a', 'b', 'c', 'd', 'e'])
    """

    percentiles = np.array([np.percentile(x, percent) for x in seq])
    nsample, nlim = percentiles.shape

    if pos is None:
        pos = np.arange(nsample) + 1
    else:
        pos = np.atleast_1d(pos)
    if colors is None:
        colors = ['k'] * nsample
    else:
        colors = np.atleast_1d(colors)
    if labels is None:
        labels = [''] * nsample
    else:
        labels = np.atleast_1d(labels)
    if labelloc == 'up':
        ha = 'center'
        va = 'bottom'
        dy = labelpad
    elif labelloc == 'down':
        ha = 'center'
        va = 'top'
        dy = -labelpad
    else:
        raise ValueError("`labelloc` must be one of "
                         "('top', 'bottom')")

    if ax is None:
        ax = plt.axes([0.02, 0.08, 0.96, 0.90])
    lw = 0.
    ibar = -1
    for ibar in range(nlim // 2):
        lw += plt.rcParams['lines.linewidth']
        ax.hlines(pos, percentiles[:, ibar], percentiles[:, -ibar-1],
                  colors=colors, linewidth=lw, **barkw)
    if nlim % 2 == 1:
        for isample in range(nsample):
            ax.plot(percentiles[isample, ibar+1], pos[isample],
                    ms=lw+2, mfc='w', mew=lw, mec=colors[isample],
                    lw=0.0, **symkw)
            ax.text(percentiles[isample, ibar+1],
                    pos[isample] + dy,
                    labels[isample], ha=ha, va=va,
                    color=colors[isample], **labelkw)
    else:
        for isample in range(nsample):
            ax.text(percentiles[isample, ibar:ibar+2].mean(),
                    pos[isample] + dy,
                    labels[isample], ha=ha, va=va,
                    color=colors[isample], **labelkw)

    ax.tick_params(axis='both', left='off', top='off', right='off',
                   labelleft='off', labeltop='off', labelright='off')
    for side in ['top', 'left', 'right']:
        ax.spines[side].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_smart_bounds(True)

    return ax
