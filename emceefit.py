from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
from scipy.stats import norm

from .stats import gaussNd_pdf


# --------------------------------------------------------------------
# MCMC driver
# --------------------------------------------------------------------


def modelfit_emcee(lnpost, guess, skip_minimize=False,
                   threads=1, nwalkers=20, dwalker=1e-3,
                   nstep=1000, ndiscard=100, verbose=False,
                   **kwargs):

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

    # run MCMC
    sampler = EnsembleSampler(nwalkers, nparam, lnpost,
                              kwargs=kwargs, threads=threads)
    pos = [start + dwalker * np.random.randn(nparam)
           for i in range(nwalkers)]
    sampler.run_mcmc(pos, ndiscard, storechain=False)
    if verbose:
        print("First {} steps finished and discarded."
              "".format(ndiscard))
    sampler.run_mcmc(pos, nstep)
    if verbose:
        print("{} steps of MCMC sampling finished."
              "".format(nstep))
    return sampler.chain.reshape((-1, nparam))


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


def lnpost_line_censored(params, data=None, cov=None,
                         intrinsic_scatter='none',
                         lnprior=None, priorargs=(),
                         bounds=None, force_homoscedastic=True):
    """
    Natural logarithm of the posterior probability of a line model,
    with the presence of data censoring.
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

    if intrinsic_scatter == 'none':
        if np.all(cov == 0):
            raise ValueError("`intrinsic_scatter` should not be 'none' "
                             "if `cov` is not provided or all zero.")
        incl, inter, x0, logsx = params
        scatter = 0.0
    elif intrinsic_scatter in ['vert', 'perp']:
        incl, inter, logs, x0, logsx = params
        scatter = 10**logs
    else:
        raise ValueError("`intrinsic_scatter` should be 'none', "
                         "'vert' or 'perp'.")
    slope = np.tan(incl)
    y0 = slope * x0 + inter
    xwidth = 10**logsx
    normarr = np.array([[1.0, slope], [slope, slope**2]])
    cov_model = normarr * xwidth**2
    if intrinsic_scatter == 'vert':
        cov_model[1, 1] += scatter**2
    else:
        normarr = np.array([[slope**2, -slope], [-slope, 1.0]])
        cov_model += normarr * scatter**2 / (1 + slope**2)
    cov_whole = cov + cov_model.reshape(1, 2, 2)
    lnlike = np.sum(gaussNd_pdf(np.array(data),
                                mean=np.array([x0, y0]),
                                cov_matrix=cov_whole,
                                return_log=True))
    if bounds is None:
        return lnlike + lnpr

    from scipy.integrate import dblquad

    xmin, xmax, ymin, ymax = bounds

    def integrand(y, x, cov_matrix):
        return gaussNd_pdf(np.array((x, y)),
                           mean=np.array([x0, y0]),
                           cov_matrix=cov_matrix)

    npoint = data.shape[0]
    if cov_whole.shape[0] == 1 or force_homoscedastic:
        cov_whole = np.median(cov_whole, axis=0)
        norm_factor = dblquad(integrand, xmin, xmax, ymin, ymax,
                              args=(cov_whole, ))[0]
        lnlike -= npoint * np.log(norm_factor)
        return lnpr + lnlike
    else:
        for ipoint in range(npoint):
            norm_factor = dblquad(integrand, xmin, xmax, ymin, ymax,
                                  args=(cov_whole[ipoint, :, :], ))[0]
            lnlike -= np.log(norm_factor)
        return lnpr + lnlike


# def lnpost2D_censored(params, lnpost_func,
#                       data=None, cov=None, bounds=None,
#                       **kwargs):
#     """
#     Adapt 2D lnpost functions to correct for data-censoring.
#     """
#     lnpost = lnpost_func(params, data=data, cov=cov, **kwargs)

#     if bounds is None:
#         return lnpost

#     from scipy.integrate import dblquad

#     xmin, xmax, ymin, ymax = bounds
#     # check data in bounds

#     def integrand(y, x, cov):
#         lnpost = lnpost_func(params, data=np.array([[x, y]]),
#                              cov=cov, **kwargs)
#         return np.exp(lnpost)

#     npoint = data.shape[0]
#     if cov.size == 4:
#         return lnpost - npoint * np.log(dblquad(integrand,
#                                                 xmin, xmax,
#                                                 ymin, ymax,
#                                                 args=(cov, ))[0])
#     else:
#         for ipoint in range(npoint):
#             lnpost -= np.log(dblquad(integrand,
#                                      xmin, xmax, ymin, ymax,
#                                      args=(cov[ipoint, :, :], ))[0])
#         return lnpost
