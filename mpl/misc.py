from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np


def label_line(line, label, x, y, **kwargs):

    """
    Add a label to a straight line, at the proper angle.

    Parameters
    ----------
    line : `~matplotlib.lines.Line2D` object,
    label : str
    x : float
        x-position to place center of text (in data coordinates)
    y : float
        y-position to place center of text (in data coordinates)
    **kwargs
        other keyword arguments to be passed to
        `~matplotlib.pyplot.annotate`

    Notes
    -----
    Copied (with minor modifications) from:
    https://stackoverflow.com/questions/18780198/how-to-rotate-matplotlib-annotation-to-match-a-line
    """

    xdata, ydata = line.get_data()
    x1 = xdata[0]
    x2 = xdata[-1]
    y1 = ydata[0]
    y2 = ydata[-1]

    if 'va' in kwargs:
        kwargs.pop('va')

    ax = line.get_axes()
    text = ax.annotate(label, xy=(x, y), xytext=(0, 0),
                       textcoords='offset points',
                       verticalalignment='center_baseline',
                       **kwargs)

    sp1 = ax.transData.transform_point((x1, y1))
    sp2 = ax.transData.transform_point((x2, y2))

    rise = (sp2[1] - sp1[1])
    run = (sp2[0] - sp1[0])

    slope_degrees = np.degrees(np.arctan2(rise, run))
    text.set_rotation(slope_degrees)

    return text
