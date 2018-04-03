from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
from scipy.stats import norm

import bmcmc


class ThickLine(bmcmc.Model):

    # setup descriptor
    def set_descr(self):
        # self.descr[varname] = [type, value, sigma, LaTeXname,
        #                        min_val, max_val]
        guess = self.eargs['guess']
        stepsz = self.eargs['stepsz']
        bounds = self.eargs['bounds']
        self.descr['slope'] = ['l0', guess[0], stepsz[0],
                               r'$\beta$',
                               bounds[0][0], bounds[0][1]]
        self.descr['intercept'] = ['l0', guess[1], stepsz[1],
                                   r'$A$',
                                   bounds[1][0], bounds[1][1]]
        self.descr['scatter'] = ['l0', guess[2], stepsz[2],
                                 r'$\sigma$',
                                 bounds[2][0], bounds[2][1]]

    # setup arguments
    def set_args(self):
        # During sampling, the code will pass self.args as an argument
        # to self.lnfunc()
        if 'data' not in self.eargs:
            raise ValueError("No data!?")
        else:
            if self.eargs['data'].shape[-1] != 2:
                raise ValueError("`data` array should have its "
                                 "shape like (..., 2)")
            self.args['data'] = self.eargs['data']
        if 'cov' not in self.eargs:
            self.args['cov'] = np.zeros(2, 2)
        else:
            if self.eargs['cov'].shape[-2:] != (2, 2):
                raise ValueError("`cov` array should have its "
                                 "shape like (..., 2, 2)")
            self.args['cov'] = self.eargs['cov']
        if 'x0' not in self.eargs:
            self.args['x0'] = 0.0
        else:
            self.args['x0'] = self.eargs['x0']
        if 'scatter_along' not in self.eargs:
            self.args['scatter_along'] = 'y'
        else:
            if self.eargs['scatter_along'] not in ['y', 'perp']:
                raise ValueError("`scatter_along` should be either"
                                 "'y' or 'perp'.")

    # define the function which needs to be sampled (log posterior)
    def lnfunc(self, args):
        beta = args['slope']
        y0 = args['intercept']
        if args['scatter_along'] == 'perp':
            sigma = args['scatter'] * np.sqrt(beta**2 + 1)
        else:
            sigma = args['scatter']
        x0 = args['x0']
        normarr = np.array([-beta, 1])
        dy = (np.einsum('...i,i', args['data'], normarr) -
              (y0 - beta * x0))
        var = (np.einsum('i,...ij,j', normarr, args['cov'], normarr) +
               sigma**2)
        return np.sum(norm.logpdf(dy, scale=var**0.5))
