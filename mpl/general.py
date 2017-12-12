from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import matplotlib.pyplot as plt


def mpl_setup(figwidth='half', aspect=0.8, lw=1.0, ms=1.0, fs=11.0,
              style='seaborn-paper', rcParams={}):

    """
    Configure matplotlib rcParams.

    Parameters
    ----------
    figwidth : {'full', 'half', 'onethird', float}
        This parameter specifies the width of a figure:
            * 'full': suitable for figures spanning the full width of
                      a letter size page.
            * 'half': 1/2 the width of a letter size page.
            * 'onethird': 1/3 the width of a letter size page.
            * float: absolute width (in inches).
    aspect : float, optional
        Aspect ratio (height/width) of a figure.
        Default is 0.8.
    lw : float, optional
        Line width (in points). Default is 1.
    ms : float, optional
        Marker size (in points). Default is 1.
    fs : float, optional
        Font size (in points). Default is 11.
    style : string, optional
        Style name to be passed to `~matplotlib.pyplot.style.use`.
        Default is 'seaborn-paper'.
    rcParams : dict, optional
        Other `~matplotlib.rcParams`
    """

    # style
    plt.style.use(style)

    # figure
    letter_page_size = (8.5, 11.0)  # in inches
    if figwidth == 'full':
        fw = 0.84 * letter_page_size[0]
    elif figwidth == 'half':
        fw = 0.40 * letter_page_size[0]
    elif figwidth == 'onethird':
        fw = 0.27 * letter_page_size[0]
    else:
        fw = figwidth
    plt.rcParams['figure.figsize'] = (fw, fw*aspect)
    plt.rcParams['figure.dpi'] = 200

    # image
    plt.rcParams['image.cmap'] = 'gray'
    plt.rcParams['image.interpolation'] = 'none'

    # linewidth
    for key in ['lines.linewidth', 'axes.linewidth',
                'patch.linewidth', 'hatch.linewidth',
                'grid.linewidth', 'lines.markeredgewidth',
                'xtick.major.width', 'xtick.minor.width',
                'ytick.major.width', 'ytick.minor.width']:
        plt.rcParams[key] = lw

    # errorbar cap size
    plt.rcParams['errorbar.capsize'] = 2.0

    # markersize
    plt.rcParams['lines.markersize'] = ms

    # fontsize
    plt.rcParams['font.size'] = fs
    plt.rcParams['axes.labelsize'] = 'small'
    for key in ['axes.titlesize', 'legend.fontsize',
                'figure.titlesize']:
        plt.rcParams[key] = 'medium'

    # font
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times'
    plt.rcParams['font.monospace'] = 'Terminal'

    # axes
    plt.rcParams['xtick.top'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['xtick.major.size'] = 4 * lw
    plt.rcParams['xtick.minor.size'] = 2 * lw
    plt.rcParams['xtick.major.pad'] = 0.3 * fs
    plt.rcParams['xtick.minor.pad'] = 0.2 * fs
    plt.rcParams['ytick.right'] = True
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['ytick.major.size'] = 4 * lw
    plt.rcParams['ytick.minor.size'] = 2 * lw
    plt.rcParams['ytick.major.pad'] = 0.2 * fs
    plt.rcParams['ytick.minor.pad'] = 0.1 * fs

    # legend
    plt.rcParams['legend.handlelength'] = 1.0
    plt.rcParams['legend.handletextpad'] = 0.5
    plt.rcParams['legend.framealpha'] = 0.2
    plt.rcParams['legend.edgecolor'] = u'0.0'

    # hatch
    plt.rcParams['hatch.color'] = 'gray'

    for key in rcParams:
        plt.rcParams[key] = rcParams[key]

    return
