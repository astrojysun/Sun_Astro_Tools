from __future__ import (
    division, print_function, absolute_import, unicode_literals)

import re
import warnings
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp


def clean_header(hdr, remove_keys=[], keep_keys=[],
                 simplify_scale_matrix=True):
    """
    Clean a fits header and retain only the necessary keywords.

    Parameters
    ----------
    hdr : fits header object
        Header object to be cleaned
    remove_keys : {'3D', '2D', 'celestial', iterable}, optional
        List of keys to remove before feeding the header to WCS
        If set to '3D', remove keys irrelevant to 3D data cubes;
        if set to '2D', remove keys irrelevant to 2D images;
        if set to 'celestial', remove keys irrelevant to the
        celestial coordinate frame.
    keep_keys : iterable, optional
        List of keys to keep
    simplify_scale_matrix : bool, optional
        Whether to reduce CD or PC matrix if possible (default: True)

    Returns
    -------
    newhdr : fits header object
        Cleaned header
    """
    newhdr = hdr.copy()
    if remove_keys == '3D':
        newhdr['NAXIS'] = 3
        rmkeys = [
            'WCSAXES', 'NAXIS4', 'OBSGEO-X', 'OBSGEO-Y', 'OBSGEO-Z',
            'OBS-RA', 'OBS-DEC', 'MJD-OBS', 'DATE-OBS',
        ]
        for key in rmkeys:
            newhdr.remove(key, ignore_missing=True, remove_all=True)
        for key in hdr:
            if re.match('^C[A-Z]*[4]$', key) is not None:
                newhdr.remove(key, remove_all=True)
            if re.match('^PC[0-9_]*[4][0-9_]*', key) is not None:
                newhdr.remove(key, remove_all=True)
    elif remove_keys == '2D':
        newhdr['NAXIS'] = 2
        rmkeys = [
            'WCSAXES', 'SPECSYS', 'RESTFRQ', 'NAXIS3', 'NAXIS4',
            'OBS-RA', 'OBS-DEC', 'MJD-OBS', 'DATE-OBS',
            'OBSGEO-X', 'OBSGEO-Y', 'OBSGEO-Z',
        ]
        for key in rmkeys:
            newhdr.remove(key, ignore_missing=True, remove_all=True)
        for key in hdr:
            if re.match('^C[A-Z]*[34]$', key) is not None:
                newhdr.remove(key, remove_all=True)
            if re.match('^PC[0-9_]*[34][0-9_]*', key) is not None:
                newhdr.remove(key, remove_all=True)
    elif remove_keys == 'celestial':
        rmkeys = ['WCSAXES', 'OBSGEO-X', 'OBSGEO-Y', 'OBSGEO-Z']
        for key in rmkeys:
            newhdr.remove(key, ignore_missing=True, remove_all=True)
        newhdr = WCS(newhdr).celestial.to_header()
        rmkeys = ['WCSAXES', 'SPECSYS', 'RESTFRQ',
                  'MJD-OBS', 'DATE-OBS', 'OBS-RA', 'OBS-DEC']
        for key in rmkeys:
            newhdr.remove(key, ignore_missing=True, remove_all=True)
    else:
        rmkeys = (remove_keys +
                  ['WCSAXES', 'OBSGEO-X', 'OBSGEO-Y', 'OBSGEO-Z',
                   'OBS-RA', 'OBS-DEC', 'MJD-OBS', 'DATE-OBS'])
    newwcs = WCS(newhdr)
    if simplify_scale_matrix:
        # simplify pixel scale matrix
        mask = ~np.eye(newwcs.pixel_scale_matrix.shape[0]).astype('?')
        if not any(newwcs.pixel_scale_matrix[mask]):
            cdelt = newwcs.pixel_scale_matrix.diagonal()
            del newwcs.wcs.pc
            del newwcs.wcs.cd
            newwcs.wcs.cdelt = cdelt
        else:
            warnings.warn("WCS pixel scale matrix has non-zero "
                          "off-diagonal elements. 'CDELT' keywords "
                          "might not reflect actual pixel size.")
    newhdr = newwcs.to_header()
    newhdr.remove('WCSAXES')
    for key in keep_keys:
        if key in hdr:
            newhdr[key] = hdr[key]
    return newhdr


def regrid_image_hdu(inhdu, newhdr, keep_header_keys=[],
                     **kwargs):
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
