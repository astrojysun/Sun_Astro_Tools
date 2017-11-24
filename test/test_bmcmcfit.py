from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import matplotlib.pyplot as plt

from ..bmcmcfit import ThickLine
from ..mpl import mpl_setup


def test_ThickLine():
    """
    Sanity check for the ThickLine model.
    """
    # generate fake data
    npoints = 5000
    cov = np.array([[0.09, -0.03], [-0.03, 0.09]])
    x0 = 0.0
    xwidth = 3.0
    slope = 0.75
    inter = 1.00
    scatter = 1.25
    x = np.random.rand(npoints) * 2 * xwidth - xwidth + x0
    y = slope * x + inter + np.random.randn(npoints) * scatter
    data = (np.vstack((x, y)).T +
            np.random.multivariate_normal([0, 0], cov, npoints))

    # MCMC fit
    eargs = {'data': data, 'cov': cov, 'x0': 0.0,
             'guess': [0.5, np.median(data[1]), 1.0],
             'stepsz': [1., 1., 1.],
             'bounds': [[0, np.inf], [-2, 4], [1e-1, 1e1]]}
    thickline = ThickLine(eargs=eargs)
    thickline.sample(['slope', 'intercept', 'scatter'],
                     50000, ptime=10000)

    # visualize data and MCMC result
    mpl_setup('half')
    plt.plot(data[:, 0], data[:, 1], 'ko')
    xref = np.percentile(data[:, 0], (0, 100))
    beta, y0, sigma = thickline.best_fit()
    plt.plot(xref, beta * (xref - x0) + y0, 'b-')
    plt.show()
    plt.close()
    thickline.plot(burn=1000)
    plt.show()
    plt.close()

    # check if true values are within the 3-sigma (99.73%) range
    # burn = 10000
    # slope_range = np.percentile(thickline.chain['slope'][burn:],
    #                             [0.135, 99.865])
    # assert slope_range[0] < slope < slope_range[1]
    # inter_range = np.percentile(thickline.chain['intercept'][burn:],
    #                             [0.135, 99.865])
    # assert inter_range[0] < inter < inter_range[1]
    # scatt_range = np.percentile(thickline.chain['scatter'][burn:],
    #                             [0.135, 99.865])
    # assert scatt_range[0] < scatter < scatt_range[1]
