from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np

from .stats import gaussNd_pdf


# --------------------------------------------------------------------
# MCMC driver
# --------------------------------------------------------------------


def modelfit_emcee(lnpost, guess, skip_minimize=False,
                   threads=1, nwalkers=20, dwalker=1e-3, nstep=1000,
                   verbose=True, ninfo=100, **kwargs):

    """
    Interface for MCMC model fitting.
    """
    import scipy.optimize as op
    from emcee import EnsembleSampler

    # input parameter check
    if not np.isfinite(lnpost(guess, **kwargs)):
        raise ValueError("Zero posterior probability for `guess`!")
    nparam = len(guess)

    # find maximum posterior solution
    if not skip_minimize:
        nlp = lambda *args: -lnpost(*args, **kwargs)
        start = op.minimize(nlp, guess)['x']
        if verbose:
            print("MAP solution found.")
    else:
        start = guess

    # initialize MCMC sampler
    sampler = EnsembleSampler(nwalkers, nparam, lnpost,
                              kwargs=kwargs, threads=threads)
    pos = [start + dwalker * np.random.randn(nparam)
           for i in range(nwalkers)]
    if verbose:
        print("MCMC sampler initialized.")

    # run MCMC
    count = ninfo
    while count < nstep:
        if count == ninfo:
            pos0 = pos
        else:
            pos0 = None
        sampler.run_mcmc(pos0, ninfo)
        if verbose:
            laststeps = sampler.chain[:, -ninfo:, :]
            summary = np.percentile(laststeps.reshape(-1, nparam),
                                    [16, 50, 84], axis=0)
            print("Summary for the {} - {} steps:"
                  "".format(count-ninfo, count))
            print(summary)
        count += ninfo

    sampler.run_mcmc(None, nstep-count+ninfo)
    if verbose:
        summary = np.percentile(sampler.flatchain,
                                [16, 50, 84], axis=0)
        print("Summary for all {} steps:".format(nstep))
        print(summary)

    return sampler.flatchain


# --------------------------------------------------------------------
# Prior
# --------------------------------------------------------------------


def lnprior_flat(params, bounds):
    """
    Natural logarithm of a flat prior probability.
    """
    for param, (lbound, ubound) in zip(params, bounds):
        if not (lbound < param < ubound):
            return -np.inf
    return 0.0


# --------------------------------------------------------------------
# Posterior
# --------------------------------------------------------------------


def lnpost_line(params, data=None, cov=None, intrinsic_scatter='none',
                lnprior=None, priorargs=()):
    """
    Natural logarithm of the posterior probability of a line model.
    """
    from scipy.stats import norm

    if lnprior is None:
        lnprior = lnprior_flat
    lnpr = lnprior(params, *priorargs)
    if not np.isfinite(lnpr):
        return -np.inf

    if data is None:
        raise ValueError("Data needed!")

    if cov is None:
        cov = np.zeros(2, 2)

    if intrinsic_scatter == 'none':
        if np.all(cov == 0):
            raise ValueError("`intrinsic_scatter` should not be "
                             "'none' if `cov` is not provided or "
                             "set to be all zero.")
        incl, inter = params
        scatter = 0.0
    elif intrinsic_scatter in ['vert', 'perp']:
        incl, inter, logs = params
        scatter = 10**logs
    else:
        raise ValueError("`intrinsic_scatter` should be 'none', "
                         "'vert' or 'perp'.")
    slope = np.tan(incl)
    if intrinsic_scatter == 'perp':
        scatter = scatter * np.sqrt(slope**2 + 1)
    normarr = np.array([-slope, 1])
    dy = np.einsum('...i,i', data, normarr) - inter
    var = np.einsum('i,...ij,j', normarr, cov, normarr) + scatter**2
    lnlike = np.sum(norm.logpdf(dy, scale=var**0.5))
    return lnpr + lnlike


def lnpost_gauss2d(params, data=None, cov=None,
                   lnprior=None, priorargs=()):
    """
    Natural logarithm of the posterior probability of a 2D Gaussian model.
    """
    if lnprior is None:
        lnprior = lnprior_flat
    lnpr = lnprior(params, *priorargs)
    if not np.isfinite(lnpr):
        return -np.inf

    if data is None:
        raise ValueError("Data needed!")

    if cov is None:
        cov = np.zeros(2, 2)

    x0, y0, logsmaj, logsmin, pa = params
    scatt_maj = 10**logsmaj
    scatt_min = 10**logsmin
    cov_model = np.diag([scatt_maj**2, scatt_min**2])
    R_matrix = np.array([[np.cos(pa), -np.sin(pa)],
                         [np.sin(pa), np.cos(pa)]])
    cov_model = np.einsum('ik,kl,jl', R_matrix, cov_model, R_matrix)
    cov_whole = cov_model + cov
    lnlike = np.sum(gaussNd_pdf(np.array(data),
                                mean=np.array([x0, y0]),
                                cov_matrix=cov_whole,
                                return_log=True))
    return lnpr + lnlike


def lnpost2D_censored(params, lnpost_func=None,
                      data=None, cov=None, bounds=None,
                      **kwargs):
    """
    Adapt 2D lnpost functions to account for data-censoring.
    """
    if lnpost_func is None:
        raise ValueError("`lnpost_func` needed!")

    lnpost = lnpost_func(params, data=data, cov=cov, **kwargs)

    if bounds is None or np.isinf(lnpost):
        return lnpost

    from scipy.integrate import dblquad

    xmin, xmax, ymin, ymax = bounds
    if callable(ymin):
        yminval = ymin(data[:, 0])
    else:
        yminval = np.full(data.shape[0], ymin)
    if callable(ymax):
        ymaxval = ymax(data[:, 0])
    else:
        ymaxval = np.full(data.shape[0], ymax)
    if (((data[:, 0] < xmin) |
         (data[:, 0] > xmax) |
         (data[:, 1] < yminval) |
         (data[:, 1] > ymaxval)).any()):
        raise ValueError("Data found in censored area")

    def integrand(y, x, cov):
        lnpost = lnpost_func(params, data=np.array([[x, y]]),
                             cov=cov, **kwargs)
        return np.exp(lnpost)

    npoint = data.shape[0]
    if cov.size == 4:
        integral = dblquad(integrand, xmin, xmax, ymin, ymax,
                           args=(cov, ))[0]
        return lnpost - npoint * np.log(integral)
    else:
        for ipoint in range(npoint):
            integral = dblquad(integrand, xmin, xmax, ymin, ymax,
                               args=(cov[ipoint, :, :], ))[0]
            lnpost -= np.log(integral)
        return lnpost
