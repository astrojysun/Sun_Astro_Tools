from __future__ import \
    division, print_function, absolute_import, unicode_literals

import numpy as np
import astropy.units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from reproject import reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs


def deproject(
        center_coord=None, incl=0*u.deg, pa=0*u.deg,
        header=None, wcs=None, naxis=None, ra=None, dec=None,
        return_offset=False):

    """
    Calculate deprojected coordinates from sky coordinates.

    This function deals with sky images of astronomical objects with
    an intrinsic disk geometry. Given disk center coordinates,
    inclination, and position angle, it calculates the deprojected
    coordinates in the disk plane (radius and azimuthal angle) from
    (1) a FITS header (`header`), or
    (2) a WCS object with specified axis sizes (`wcs` + `naxis`), or
    (3) RA and DEC coodinates (`ra` + `dec`).
    Note that the deprojected azimuthal angle is defined w.r.t. the
    line of nodes (given by the position angle). For (1) and (2), the
    outputs are 2D images; for (3), the outputs are arrays with shapes
    matching the broadcasted shape of `ra` and `dec`.

    Parameters
    ----------
    center_coord : `~astropy.coordinates.SkyCoord` object or 2-tuple
        Sky coordinates of the disk center
    incl : `~astropy.units.Quantity` object or number, optional
        Inclination angle of the disk (0 degree means face-on)
        Default is 0 degree.
    pa : `~astropy.units.Quantity` object or number, optional
        Position angle of the disk (red/receding side, North->East)
        Default is 0 degree.
    header : `~astropy.io.fits.Header` object, optional
        FITS header specifying the WCS and size of the output 2D maps
    wcs : `~astropy.wcs.WCS` object, optional
        WCS of the output 2D maps
    naxis : array-like (with two elements), optional
        Size of the output 2D maps
    ra : array-like, optional
        RA coordinate of the sky locations of interest
    dec : array-like, optional
        DEC coordinate of the sky locations of interest
    return_offset : bool, optional
        Whether to return the angular offset coordinates together with
        deprojected radii and angles. Default is to not return.

    Returns
    -------
    deprojected_coordinates : list of arrays
        If `return_offset` is set to True, the returned arrays include
        deprojected radii, projected angles, as well as angular offset
        coordinates along East-West and North-South direction;
        otherwise only the former two arrays will be returned.

    Notes
    -----
    This is the Python version of an IDL function `deproject` included
    in the `cpropstoo` package. See URL below:
    https://github.com/akleroy/cpropstoo/blob/master/cubes/deproject.pro
    """

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

    if header is not None:
        wcs_cel = WCS(header).celestial
        naxis1 = header['NAXIS1']
        naxis2 = header['NAXIS2']
        # create ra and dec grids
        ix = np.arange(naxis1)
        iy = np.arange(naxis2).reshape(-1, 1)
        ra_deg, dec_deg = wcs_cel.wcs_pix2world(ix, iy, 0)
    elif (wcs is not None) and (naxis is not None):
        wcs_cel = wcs.celestial
        naxis1, naxis2 = naxis
        # create ra and dec grids
        ix = np.arange(naxis1)
        iy = np.arange(naxis2).reshape(-1, 1)
        ra_deg, dec_deg = wcs_cel.wcs_pix2world(ix, iy, 0)
    else:
        ra_deg, dec_deg = np.broadcast_arrays(ra, dec)
        if hasattr(ra_deg, 'unit'):
            ra_deg = ra_deg.to(u.deg).value
            dec_deg = dec_deg.to(u.deg).value

    # recast the ra and dec arrays in term of the center coordinates
    # arrays are now in degrees from the center
    dx_deg = (ra_deg - x0_deg) * np.cos(np.deg2rad(y0_deg))
    dy_deg = dec_deg - y0_deg

    # rotation angle (rotate x-axis up to the major axis)
    rotangle = np.pi/2 - np.deg2rad(pa_deg)

    # create deprojected coordinate grids
    deprojdx_deg = (dx_deg * np.cos(rotangle) +
                    dy_deg * np.sin(rotangle))
    deprojdy_deg = (dy_deg * np.cos(rotangle) -
                    dx_deg * np.sin(rotangle))
    deprojdy_deg /= np.cos(np.deg2rad(incl_deg))

    # make map of deprojected distance from the center
    radius_deg = np.sqrt(deprojdx_deg**2 + deprojdy_deg**2)

    # make map of angle w.r.t. position angle
    projang_deg = np.rad2deg(np.arctan2(deprojdy_deg, deprojdx_deg))

    if return_offset:
        return radius_deg, projang_deg, dx_deg, dy_deg
    else:
        return radius_deg, projang_deg


def deproject_image(
        in_hdu, incl=0*u.deg, pa=0*u.deg, return_footprint=False,
        **kwargs):

    """
    Deproject an image given inclination & position angles.

    This function deals with sky images of astronomical objects with
    an intrinsic disk geometry. Given disk inclination and position
    angle, it rotates and stretches an image in order to 'deproject'
    it into the disk plane. Note that the line of nodes coincides with
    the y-axis in the output image (with positive-y matching the side
    identified by the position angle).

    This function essentially resamples the original image into a
    new WCS. This is implemented with `reproject.reproject_interp`.

    Parameters
    ----------
    in_hdu : FITS HDU object
        Input image
    incl : `~astropy.units.Quantity` object or number, optional
        Inclination angle of the disk (0 degree means face-on)
        Default is 0 degree.
    pa : `~astropy.units.Quantity` object or number, optional
        Position angle of the disk (red/receding side, North->East)
        Default is 0 degree.
    return_footprint : bool, optional
        If True, also return the footprint of the original image.
        Default: False
    kwargs :
        Other keywords to be passed to `reproject.reproject_interp`

    Returns
    -------
    out_image : numpy array
        Deprojected image
    out_wcs : `astropy.wcs.WCS` object
        Corresponding WCS for the deprojected image
    footprint : numpy array
        Present if `return_footprint=True`
    """

    if hasattr(incl, 'unit'):
        incl_deg = incl.to(u.deg).value
    else:
        incl_deg = incl
    if hasattr(pa, 'unit'):
        pa_deg = pa.to(u.deg).value
    else:
        pa_deg = pa

    scaling_array = np.array([1/np.cos(np.deg2rad(incl_deg)), 1])
    scaling_ratio = scaling_array[1] / scaling_array[0]
    rot_rad = np.deg2rad(pa_deg)
    rot_matrix = np.array(
        [[np.cos(rot_rad), -np.sin(rot_rad) / scaling_ratio],
         [np.sin(rot_rad) * scaling_ratio, np.cos(rot_rad)]])

    new_wcs, new_shape = find_optimal_celestial_wcs([in_hdu])
    # check if the new pixel scale matrix is diagonal
    mask = ~np.eye(new_wcs.pixel_scale_matrix.shape[0]).astype('?')
    if any(new_wcs.pixel_scale_matrix[mask]):
        raise ValueError(
            "Unable to simplify WCS pixel scale matrix...")
    # modify WCS to handle rotation and stretch
    if new_wcs.wcs.has_pc():
        new_wcs.wcs.pc = rot_matrix
        new_wcs.wcs.cdelt /= scaling_array
    else:
        raise ValueError("I can only handle PC matrix for now...")
    new_wcs.wcs.crpix *= scaling_array
    new_hdr = new_wcs.to_header()
    new_hdr['NAXIS'] = new_hdr['WCSAXES']
    new_hdr['NAXIS2'], new_hdr['NAXIS1'] = (
        np.array(new_shape) * scaling_array[::-1]).astype('int')

    if return_footprint:
        new_img, footprint = reproject_interp(
            in_hdu, new_hdr, return_footprint=return_footprint)
        return new_img, new_wcs, footprint
    else:
        new_img = reproject_interp(
            in_hdu, new_hdr, return_footprint=return_footprint)
        return new_img, new_wcs
