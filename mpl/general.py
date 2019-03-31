from __future__ import (
    division, print_function, absolute_import, unicode_literals)

import matplotlib.pyplot as plt


def mpl_setup(figtype='paper-1/2', aspect=0.8,
              lw=None, ms=None, fs=None,
              style='seaborn-paper', rcParams={}):

    """
    Configure matplotlib rcParams.

    Parameters
    ----------
    figtype : {'paper-1/1', 'paper-1/2', 'paper-1/3', 'talk', float}
        This parameter specifies the type and width of the figure(s):
            * 'paper-1/1': figure width = full textwidth (7.1 inch)
                           (normal lw, ms, fs)
            * 'paper-1/2': figure width = 1/2 textwidth
                           (normal lw, ms, fs)
            * 'paper-1/3': figure width = 1/3 textwidth
                           (normal lw, ms, fs)
            * 'talk': figure height = 4.0 inches
                      (larger lw, ms, fs)
            * float: specifying figure width (in inches)
                     (normal lw, ms, fs)
    aspect : float, optional
        Aspect ratio (height/width) of a figure.
        Default is 0.8.
    lw : float, optional
        Line width (in points). Default is 1 (1.5 for figtype='talk').
    ms : float, optional
        Marker size (in points). Default is 1 (1.5 for figtype='talk').
    fs : float, optional
        Font size (in points). Default is 11 (18 for figtype='talk').
    style : string, optional
        Style name to be passed to `~matplotlib.pyplot.style.use`.
        Default is 'seaborn-paper'.
    rcParams : dict, optional
        Other `~matplotlib.rcParams`
    """

    # style
    plt.style.use(style)

    # figure
    textwidth = 7.1  # inches
    slideheight = 4.0  # inches
    if figtype == 'paper-1/1':
        fw = 0.98 * textwidth
    elif figtype == 'paper-1/2':
        fw = 0.48 * textwidth
    elif figtype == 'paper-1/3':
        fw = 0.31 * textwidth
    elif figtype == 'talk':
        fw = slideheight / aspect
    else:
        fw = figtype
    plt.rcParams['figure.figsize'] = (fw, fw*aspect)
    plt.rcParams['figure.dpi'] = 200

    # default sizes
    if figtype == 'talk':
        if lw is None:
            lw = 1.5
        if ms is None:
            ms = 1.5
        if fs is None:
            fs = 18.0
    else:
        if lw is None:
            lw = 1.0
        if ms is None:
            ms = 1.0
        if fs is None:
            fs = 11.0

    # font
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times', 'Times New Roman',
                                  'DejaVu Serif', 'serif']
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial',
                                       'DejaVu Sans', 'sans-serif']
    plt.rcParams['font.monospace'] = ['Terminal', 'monospace']

    # font weight 
    if figtype == 'talk':
        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['figure.titleweight'] = 'bold'

    # fontsize
    plt.rcParams['font.size'] = fs
    plt.rcParams['xtick.labelsize'] = 'small'
    plt.rcParams['ytick.labelsize'] = 'small'
    plt.rcParams['legend.fontsize'] = 'medium'
    plt.rcParams['axes.labelsize'] = 'large'
    plt.rcParams['axes.titlesize'] = 'large'
    plt.rcParams['figure.titlesize'] = 'large'

    # linewidth
    for key in ['lines.linewidth', 'axes.linewidth',
                'patch.linewidth', 'hatch.linewidth',
                'grid.linewidth', 'lines.markeredgewidth',
                'xtick.major.width', 'xtick.minor.width',
                'ytick.major.width', 'ytick.minor.width']:
        plt.rcParams[key] = lw

    # errorbar cap size
    plt.rcParams['errorbar.capsize'] = 2 * lw

    # markersize
    plt.rcParams['lines.markersize'] = ms

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

    # image
    plt.rcParams['image.cmap'] = 'gray'
    plt.rcParams['image.interpolation'] = 'none'

    plt.rcParams.update(rcParams)

    return
