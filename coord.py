from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np


def deproject(params, wcs_cel, naxis, return_xymap=False):

    """
    Compute deprojected radii and projected angle (in deg)
    based on center coordinates, inclination, and position angle.

    Transplanted (with minor modification) by J. Sun from the IDL
    function `deproject`, which is included in the `cpropstoo` package
    written by A. Leroy.
    """

    pa_deg = params['POSANG_DEG'].item()
    i_deg = params['INCL_DEG'].item()
    x0_deg = params['RA_DEG'].item()
    y0_deg = params['DEC_DEG'].item()

    # create ra and dec grids
    rgrid = np.arange(naxis[0])
    dgrid = np.arange(naxis[1]).reshape(-1, 1)
    rmap, dmap = wcs_cel.wcs_pix2world(rgrid, dgrid, 0)

    # recast the ra and dec arrays in term of the center coordinates
    # arrays are now in degrees from the center
    dxmap_deg = (rmap - x0_deg) * np.cos(np.deg2rad(y0_deg))
    dymap_deg = dmap - y0_deg

    # rotation angle (rotate x-axis up to the major axis)
    rotangle = np.pi/2 - np.deg2rad(pa_deg)

    # create deprojected coordinate grids
    deprojdx_deg = (dxmap_deg * np.cos(rotangle) +
                    dymap_deg * np.sin(rotangle))
    deprojdy_deg = (dymap_deg * np.cos(rotangle) -
                    dxmap_deg * np.sin(rotangle))
    deprojdy_deg /= np.cos(np.deg2rad(i_deg))

    # make map of deprojected distance from the center
    rmap_deg = np.sqrt(deprojdx_deg**2 + deprojdy_deg**2)

    # make map of angle w.r.t. position angle
    amap_deg = np.rad2deg(np.arctan2(deprojdy_deg, deprojdx_deg))

    if return_xymap:
        return rmap_deg, amap_deg, dxmap_deg, dymap_deg
    else:
        return rmap_deg, amap_deg
