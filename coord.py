from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import astropy.units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord


def deproject(header=None, wcs=None, naxis=None,
              center_coord=None, incl=0*u.deg, pa=0*u.deg,
              return_xymap=False):

    """
    Generate deprojected radii and projected angle maps (in deg)
    based on a center, inclination, and position angle.

    This is the Python version of an IDL function `deproject`
    in the `cpropstoo` package. See URL below:
    
    https://github.com/akleroy/cpropstoo/blob/master/cubes/deproject.pro
    """

    if header is not None:
        wcs_cel = WCS(header).celestial
        naxis1 = header['NAXIS1']
        naxis2 = header['NAXIS2']
    else:
        if (wcs is None) or (naxis is None):
            raise ValueError(
                "If no header is provided, then both"
                "`wcs` and `naxis` should be specified.")
        wcs_cel = wcs.celestial
        naxis1, naxis2 = naxis

    if center_coord is None:
        raise ValueError(
            "`center_coord` should be either a SkyCoord object "
            "or an array-like object with two elements.")
    if isinstance(center_coord, SkyCoord):
        x0_deg = center_coord.ra.degree
        y0_deg = center_coord.dec.degree
    else:
        x0_deg, y0_deg = center_coord
        if hasattr(x0_deg, 'unit'):
            x0_deg = x0_deg.to(u.deg).value
            y0_deg = y0_deg.to(u.deg).value

    if hasattr(incl, 'unit'):
        incl_deg = incl.to(u.deg).value
    else:
        incl_deg = incl
    if hasattr(pa, 'unit'):
        pa_deg = pa.to(u.deg).value
    else:
        pa_deg = pa

    # create ra and dec grids
    ragrid = np.arange(naxis1)
    decgrid = np.arange(naxis2).reshape(-1, 1)
    ramap, decmap = wcs_cel.wcs_pix2world(ragrid, decgrid, 0)

    # recast the ra and dec arrays in term of the center coordinates
    # arrays are now in degrees from the center
    dxmap_deg = (ramap - x0_deg) * np.cos(np.deg2rad(y0_deg))
    dymap_deg = decmap - y0_deg

    # rotation angle (rotate x-axis up to the major axis)
    rotangle = np.pi/2 - np.deg2rad(pa_deg)

    # create deprojected coordinate grids
    deprojdx_deg = (dxmap_deg * np.cos(rotangle) +
                    dymap_deg * np.sin(rotangle))
    deprojdy_deg = (dymap_deg * np.cos(rotangle) -
                    dxmap_deg * np.sin(rotangle))
    deprojdy_deg /= np.cos(np.deg2rad(incl_deg))

    # make map of deprojected distance from the center
    rmap_deg = np.sqrt(deprojdx_deg**2 + deprojdy_deg**2)

    # make map of angle w.r.t. position angle
    amap_deg = np.rad2deg(np.arctan2(deprojdy_deg, deprojdx_deg))

    if return_xymap:
        return rmap_deg, amap_deg, dxmap_deg, dymap_deg
    else:
        return rmap_deg, amap_deg
