from __future__ import (
    division, print_function, absolute_import, unicode_literals)

import numpy as np
import matplotlib.pyplot as plt


def minimal_barplot(seq, percent=[16., 50., 84.],
                    pos=None, colors=None,
                    labels=None, labelloc='up', labelpad=0.1,
                    ax=None, barkw={}, labelkw={}, **symkw):
    """
    Create barplot, in the spirit of minimalism.

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
