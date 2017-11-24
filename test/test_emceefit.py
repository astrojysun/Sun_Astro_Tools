from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import matplotlib.pyplot as plt

from ..emceefit import (modelfit_emcee, lnpost_line, lnpost_gauss2d,
                        lnpost_line_censored)
from ..mpl import mpl_setup
from ..mpl.fig import easy_corner


def test_lnpost_line():
    """
    Sanity check for the lnpost_line model.
    """
    # generate fake data
    npoints = 5000
    cov = np.array([[0.09, -0.03], [-0.03, 0.09]])
    x0 = 0.0
    xwidth = 3.0
    slope = np.tan(np.pi/4)
    inter = 1.0
    scatter = 1.
    x = np.random.rand(npoints) * 2 * xwidth - xwidth + x0
    y = slope * x + inter + np.random.randn(npoints) * scatter
    data = (np.vstack((x, y)).T +
            np.random.multivariate_normal([0, 0], cov, npoints))

    # visualize data
    mpl_setup('half')
    plt.plot(x, y, 'ko')
    plt.show()
    plt.close()

    # MCMC fit
    guess = [0.0, np.mean(data[:, 1]), 0.0]
    param_bounds = np.array([[-np.pi*0.4, np.pi*0.4],
                             [-5, 5], [-1, 1]])
    samples = modelfit_emcee(lnpost_line, guess,
                             ndiscard=1000, nstep=5000,
                             data=data, cov=cov,
                             priorargs=(param_bounds, ),
                             intrinsic_scatter='vert')
    samples[:, 0] = np.tan(samples[:, 0])
    samples[:, 2] = 10**(samples[:, 2])

    # visualize MCMC result
    labels = (r"$\beta$", r"$A$", r"$\sigma$")
    nparam = len(labels)
    fig, axes = plt.subplots(nparam, nparam, figsize=(6, 6))
    fig = easy_corner(samples,
                      truths=(slope, inter, scatter),
                      fig=fig, labels=labels)
    plt.show()
    plt.close()


def test_lnpost_gauss2d():
    """
    Sanity check for the lnpost_line model.
    """
    # generate fake data
    npoints = 5000
    cov = np.array([[0.09, -0.03], [-0.03, 0.09]])
    x0 = 0.0
    y0 = 1.0
    scatt_maj = 3.0
    scatt_min = 1.0
    pa = np.pi/4
    cov_model = np.diag([scatt_maj**2, scatt_min**2])
    R_matrix = np.array([[np.cos(pa), -np.sin(pa)],
                         [np.sin(pa), np.cos(pa)]])
    cov_model = np.einsum('ik,kl,jl',
                          R_matrix, cov_model, R_matrix)
    data = (np.random.multivariate_normal([x0, y0], cov_model,
                                          npoints) +
            np.random.multivariate_normal([0, 0], cov, npoints))

    # visualize data
    plt.plot(data[:, 0], data[:, 1], 'ko')
    plt.show()
    plt.close()

    # MCMC fit
    guess = [np.mean(data[:, 0]), np.mean(data[:, 1]),
             np.log10(np.std(data[:, 0])),
             np.log10(np.std(data[:, 1])),
             np.pi/4]
    param_bounds = np.array([[-10, 10],
                             [-5, 5],
                             [-1.0, 1.0],
                             [-1.0, 1.0],
                             [0.0, np.pi/2]])
    samples = modelfit_emcee(lnpost_gauss2d, guess,
                             ndiscard=1000, nstep=5000,
                             data=data, cov=cov,
                             priorargs=(param_bounds, ))
    samples[:, 2] = 10**samples[:, 2]
    samples[:, 3] = 10**samples[:, 3]
    samples[:, 4] = np.tan(samples[:, 4])

    # visualize MCMC result
    labels = (r"$x_0$", r"$y_0$", r"$\sigma_{\rm major}$",
              r"$\sigma_{\rm minor}$", r"$\beta$")
    nparam = len(labels)
    fig, axes = plt.subplots(nparam, nparam, figsize=(6, 6))
    fig = easy_corner(samples, fig=fig, labels=labels,
                      truths=(x0, y0, scatt_maj,
                              scatt_min, np.tan(pa)))
    plt.show()
    plt.close()


def test_lnpost_line_censored():
    """
    Sanity check for the lnpost_line_censored model.
    """
    # generate fake data
    npoints = 5000
    cov = np.array([[0.09, -0.03], [-0.03, 0.09]])
    x0 = 0.0
    xwidth = 3.0
    slope = np.tan(np.pi/4)
    inter = 1.0
    scatter = 1.
    xmin = -12
    xmax = 20
    ymin = lambda x: np.array(x) - 9
    ymax = lambda x: 2 * np.array(x) + 3
    x = np.random.randn(npoints) * xwidth + x0
    y = slope * x + inter + np.random.randn(npoints) * scatter
    data = (np.vstack((x, y)).T +
            np.random.multivariate_normal([0, 0], cov, npoints))
    censored_data = data[data[:, 1] < ymax(data[:, 0]), :]

    # visualize data
    plt.plot(censored_data[:, 0], censored_data[:, 1], 'ko')
    plt.plot([xmin, xmax], [ymin(xmin), ymin(xmax)], color='r')
    plt.plot([xmin, xmax], [ymax(xmin), ymax(xmax)], color='r')
    plt.vlines([xmin, xmax], [ymin(xmin), ymin(xmax)],
               [ymax(xmin), ymax(xmax)], colors='r')
    plt.show()
    plt.close()

    # MCMC fit
    guess = [np.arctan(slope), inter, np.log10(scatter),
             x0, np.log10(xwidth)]
    param_bounds = np.array([[-np.pi*0.4, np.pi*0.4],
                             [-5, 5],
                             [-1, 1],
                             [-10, 10],
                             [-1, 1]])
    samples = modelfit_emcee(lnpost_line_censored, guess,
                             skip_minimize=True,
                             ndiscard=1000, nstep=5000,
                             data=censored_data, cov=cov,
                             priorargs=(param_bounds, ),
                             bounds=(xmin, xmax, ymin, ymax),
                             intrinsic_scatter='vert')
    samples[:, 0] = np.tan(samples[:, 0])
    samples[:, 2] = 10**(samples[:, 2])
    samples[:, 4] = 10**(samples[:, 4])

    # visualize MCMC result
    labels = (r"$\beta$", r"$A$", r"$\sigma$",
              r"$x_0$", r"$\sigma_x$")
    nparam = len(labels)
    fig, axes = plt.subplots(nparam, nparam, figsize=(6, 6))
    fig = easy_corner(samples, fig=fig, labels=labels,
                      truths=(slope, inter, scatter, x0, xwidth))
    plt.show()
    plt.close()
