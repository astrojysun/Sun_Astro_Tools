from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
from astropy import units as u, constants as const


def JyPerSteradian2K(x, freq_Hz):
    x_JyPerSr = np.array(x).copy() * u.Jy/u.sr
    factor_RJ = 2 * freq_Hz**2 * u.Hz**2 * const.k_B / const.c**2
    x_K = x_JyPerSr / factor_RJ
    x_K = x_K.to(u.K, equivalencies=u.dimensionless_angles())
    return x_K.value


def JyPerBeam2K(x, beam_params, freq_Hz):
    x_JyPerB = np.array(x).copy() * u.Jy/u.beam
    bmaj, bmin = beam_params
    area_SqdegPerB = (np.pi * bmaj * bmin / np.log(2) / 4 *
                      u.deg**2/u.beam)
    factor_RJ = 2 * freq_Hz**2 * u.Hz**2 * const.k_B / const.c**2
    x_K = x_JyPerB / area_SqdegPerB / factor_RJ
    x_K = x_K.to(u.K, equivalencies=u.dimensionless_angles())
    return x_K.value
