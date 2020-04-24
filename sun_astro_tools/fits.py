from __future__ import (
    division, print_function, absolute_import, unicode_literals)

import re
import warnings
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp


def clean_header(
    hdr, auto=None, remove_keys=[], keep_keys=[],
    simplify_scale_matrix=True):
    """
    Clean a fits header and retain only the necessary keywords.

    Parameters
    ----------
    hdr : fits header object
        Header object to be cleaned
    auto : {None, 'image', 'cube'}, optional
        If 'image' : retain only relevant WCS keywords for 2D images
        If 'cube' : retain only relevant WCS keywords for 3D cubes
    remove_keys : iterable, optional
        Keys to manually remove
    keep_keys : iterable, optional
        Keys to manually keep
    simplify_scale_matrix : bool, optional
        Whether to reduce CD or PC matrix if possible (default: True)

    Returns
    -------
    newhdr : fits header object
        Cleaned header
    """
    newhdr = hdr.copy()

    # remove keywords
    for key in remove_keys + [
        'WCSAXES', 'OBSGEO-X', 'OBSGEO-Y', 'OBSGEO-Z',
        'OBS-RA', 'OBS-DEC', 'MJD-OBS', 'DATE-OBS']:
        newhdr.remove(key, ignore_missing=True, remove_all=True)

    # auto clean
    if auto == 'image':
        naxis = 2
        newwcs = WCS(newhdr).reorient_celestial_first().sub(naxis)
    elif auto == 'cube':
        naxis = 3
        newwcs = WCS(newhdr).reorient_celestial_first().sub(naxis)
    else:
        naxis = -1
        newwcs = WCS(newhdr)

    # simplify pixel scale matrix
    if simplify_scale_matrix:
        mask = ~np.eye(newwcs.pixel_scale_matrix.shape[0]).astype('?')
        if not any(newwcs.pixel_scale_matrix[mask]):
            cdelt = newwcs.pixel_scale_matrix.diagonal()
            del newwcs.wcs.pc
            del newwcs.wcs.cd
            newwcs.wcs.cdelt = cdelt
        else:
            warnings.warn(
                "WCS pixel scale matrix has non-zero "
                "off-diagonal elements. 'CDELT' keywords "
                "might not reflect actual pixel size.")

    # construct new header
    newhdr = newwcs.to_header()
    # insert mandatory keywords
    if newwcs.pixel_shape is not None:
        for i in range(newwcs.pixel_n_dim):
            newhdr.insert(
                i, ('NAXIS{}'.format(i+1), newwcs.pixel_shape[i]))
    newhdr.insert(0, ('NAXIS', newhdr['WCSAXES']))
    newhdr.remove('WCSAXES')
    for key in ['BITPIX', 'SIMPLE']:
        if key in hdr:
            newhdr.insert(0, (key, hdr[key]))
    # retain old keywords
    for key in keep_keys:
        if key not in hdr:
            continue
        newhdr[key] = hdr[key]

    return newhdr


def regrid_header(
    hdr, new_crval=None, new_cdelt=None, keep_old_cdelt_sign=True,
    keep_non_celestial_axes=False, keep_keys=[]):
    """
    Regrid a FITS header while perserving its sky footprint.

    Note that only the celestial coordinates are regridded.

    Parameters
    ----------
    hdr : fits header object
        Header object to be regridded
    new_crval : array-like with two elements, optional
        Coordinates of the new reference point
    new_cdelt : scalar or array-like, optional
        New coordinate increment. Need to be either a scalar
        or an array-like object with its length equal to the
        number of celestial coordinates
    keep_old_cdelt_sign : bool, optional
        If True (default), perserve the signs of the original
        CDELT parameters; otherwise use the signs specified by
        the input parameter `new_cdelt`
    keep_non_celestial_axes : bool, optional
        If False (default), only retain information about the
        celestial coordinates in the returned header;
        otherwise retain all WCS coordinates
    keep_keys : iterable, optional
        Keys to manually keep

    Returns
    -------
    newhdr : fits header object
        Regridded header
    """
    inwcs = WCS(hdr).reorient_celestial_first()
    naxes = np.array(inwcs.pixel_shape)
    # check pixel scale matrix
    mask = ~np.eye(inwcs.pixel_scale_matrix.shape[0]).astype('?')
    if any(inwcs.pixel_scale_matrix[mask]):
        raise ValueError(
            "Unable to simplify WCS pixel scale matrix: "
            "non-zero off-diagonal elements")
    # simplify pixel scale matrix
    cdelt = inwcs.pixel_scale_matrix.diagonal()
    del inwcs.wcs.pc
    del inwcs.wcs.cd
    inwcs.wcs.cdelt = cdelt

    # regrid celestial coordinates
    wcs_cel = inwcs.celestial.copy()
    footprint_world = wcs_cel.calc_footprint()
    if new_crval is not None:
        wcs_cel.wcs.crval = np.asfarray(new_crval)
    if new_cdelt is not None:
        if keep_old_cdelt_sign:
            wcs_cel.wcs.cdelt = (
                np.abs(new_cdelt) *
                np.sign(wcs_cel.wcs.cdelt))
        else:
            wcs_cel.wcs.cdelt = (
                np.asfarray(new_cdelt) *
                np.ones_like(wcs_cel.wcs.cdelt))
    # adjust CRPIX so that the "lower left corner" is near (1, 1)
    footprint_pix = wcs_cel.all_world2pix(footprint_world, 1)
    wcs_cel.wcs.crpix -= np.floor(footprint_pix.min(axis=0)) - 1
    # find new NAXISi
    footprint_pix = wcs_cel.all_world2pix(footprint_world, 1)
    naxes_cel = np.ceil(footprint_pix.max(axis=0)).astype('int')
    naxes = np.hstack((naxes_cel, naxes[wcs_cel.pixel_n_dim:]))

    # construct new header
    newhdr = wcs_cel.to_header()
    if not keep_non_celestial_axes:
        # add 'NAXISi'
        for i in range(wcs_cel.pixel_n_dim):
            newhdr.insert(
                i, ('NAXIS{}'.format(i+1), naxes_cel[i]))
        # add 'NAXIS'
        newhdr.insert(0, ('NAXIS', wcs_cel.pixel_n_dim))
    else:
        # add WCS keywords for non-celestial axes
        allhdr = inwcs.to_header()
        for key in allhdr:
            if key not in newhdr:
                newhdr[key] = (allhdr[key], allhdr.comments[key])
        newhdr = WCS(newhdr).to_header()
        # add 'NAXISi'
        for i in range(inwcs.pixel_n_dim):
            newhdr.insert(
                i, ('NAXIS{}'.format(i+1), naxes[i]))
        # add 'NAXIS'
        newhdr.insert(0, ('NAXIS', inwcs.pixel_n_dim))
    # add other mandatory keywords
    for key in ['BITPIX', 'SIMPLE']:
        if key in hdr:
            newhdr.insert(0, (key, hdr[key]))
    newhdr.remove('WCSAXES')
    # retain old keywords
    for key in keep_keys:
        newhdr[key] = hdr[key]

    return newhdr


def regrid_image_hdu(
    inhdu, newhdr, keep_header_keys=[], **kwargs):
    """
    Regrid a FITS image HDU (2D or 3D) according to a specified header.

    This is a simple wrapper around `~reproject.reproject_interp`.

    Parameters
    ----------
    inhdu : FITS HDU object
        Input FITS HDU
    newhdr : FITS header object
        New header that defines the new grid
    keep_header_keys : array-like, optional
        List of keys to keep from the old header in 'inhdu'
        Default is an empty list.
    **kwargs
        Keywords to be passed to `~reproject.reproject_interp`

    Returns
    -------
    outhdu : FITS HDU object
        Regridded HDU
    """
    if inhdu.header['NAXIS'] < 2 or inhdu.header['NAXIS'] > 4:
        raise ValueError("Input HDU has 'NAXIS'={}"
                         "".format(inhdu.header['NAXIS']))
    hdr = newhdr.copy()
    for key in keep_header_keys:
        hdr[key] = inhdu.header[key]
    data_proj, footprint = reproject_interp(inhdu, hdr, **kwargs)
    data_proj[~footprint.astype('?')] = np.nan
    newhdu = inhdu.__class__(data_proj, hdr)
    return newhdu
