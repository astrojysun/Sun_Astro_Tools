from __future__ import (
    division, print_function, absolute_import, unicode_literals)

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from spectral_cube import SpectralCube, Projection
from .cube import convolve_projection, convolve_cube


def clean_header(hdr, remove_keys=[], keep_keys=[]):
    """
    Clean a fits header and retain only the necessary keywords.

    Parameters
    ----------
    hdr : fits header object
        Header object to be cleaned
    remove_keys : {'3D', '2D', iterable}
        List of keys to remove before feeding the header to WCS
        If set to '3D', remove keys irrelevant to 3D data cubes;
        if set to '2D', remove keys irrelevant to 2D images.
    keep_keys : iterable
        List of keys to keep

    Returns
    -------
    newhdr : fits header object
        Cleaned header
    """
    newhdr = hdr.copy()
    if remove_keys == '3D':
        newhdr['NAXIS'] = 3
        rmkeys = ['WCSAXES',
                  'OBSGEO-X', 'OBSGEO-Y', 'OBSGEO-Z',
                  'OBS-RA', 'OBS-DEC', 'MJD-OBS', 'DATE-OBS',
                  'NAXIS4', 'CTYPE4', 'CUNIT4',
                  'CRVAL4', 'CDELT4', 'CRPIX4', 'CROTA4',
                  'PC4_1', 'PC4_2', 'PC4_3', 'PC4_4',
                  'PC1_4', 'PC2_4', 'PC3_4']
    elif remove_keys == '2D':
        newhdr['NAXIS'] = 2
        rmkeys = ['WCSAXES', 'SPECSYS', 'RESTFRQ',
                  'OBSGEO-X', 'OBSGEO-Y', 'OBSGEO-Z',
                  'OBS-RA', 'OBS-DEC', 'MJD-OBS', 'DATE-OBS',
                  'NAXIS3', 'CTYPE3', 'CUNIT3',
                  'CRVAL3', 'CDELT3', 'CRPIX3', 'CROTA3',
                  'NAXIS4', 'CTYPE4', 'CUNIT4',
                  'CRVAL4', 'CDELT4', 'CRPIX4', 'CROTA4',
                  'PC1_3', 'PC1_4', 'PC2_3', 'PC2_4',
                  'PC3_1', 'PC3_2', 'PC3_3', 'PC3_4',
                  'PC4_1', 'PC4_2', 'PC4_3', 'PC4_4']
    else:
        rmkeys = (remove_keys +
                  ['WCSAXES', 'OBSGEO-X', 'OBSGEO-Y', 'OBSGEO-Z',
                   'OBS-RA', 'OBS-DEC', 'MJD-OBS', 'DATE-OBS'])
    for key in rmkeys:
        newhdr.remove(key, ignore_missing=True, remove_all=True)
    newhdr = WCS(newhdr).to_header()
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


def convolve_image_hdu(inhdu, newbeam, append_raw=False,
                       allow_huge_operations=True, **kwargs):
    """
    Convolve a FITS image HDU (2D or 3D) to a specified beam.

    This is a simple wrapper around `.cube.convolve_cube` and
    `.cube.convolve_projection`

    Parameters
    ----------
    inhdu : FITS HDU object
        Input FITS HDU
    newbeam : radio_beam.Beam object
        Target beam to convolve to
    append_raw : bool, optional
        Whether to append the raw convolved image and weight image
        Default is not to append.
    allow_huge_operations : bool optional
        Whether to set cube.allow_huge_operations=True
        Default is True.
    **kwargs
        Keywords to be passed to either `.cube.convolve_cube`
        or `.cube.convolve_projection`

    Returns
    -------
    outhdu : FITS HDU or HDUList object
        Convolved HDU (when append_raw=False), or HDUList comprising
        3 HDUs (when append_raw=True)
    """
    naxis = inhdu.header['NAXIS']
    if naxis < 2 or naxis >= 4:
        raise ValueError("Cannot handle input HDU with NAXIS={}"
                         "".format(inhdu.header['NAIXS']))
    if naxis == 2:
        oldimg = Projection.from_hdu(inhdu)
        newimg = convolve_projection(
            oldimg, newbeam, append_raw=append_raw, **kwargs)
    else:
        oldimg = SpectralCube(inhdu.data, wcs=WCS(inhdu.header),
                              header=inhdu.header)
        oldimg.allow_huge_operations = allow_huge_operations
        newimg = convolve_cube(
            oldimg, newbeam, append_raw=append_raw, **kwargs)
    if newimg is None:
        return
    newhdr = inhdu.header.copy(strip=True)
    newhdr.remove('WCSAXES', ignore_missing=True)
    if not append_raw:
        for key in ['BMAJ', 'BMIN', 'BPA']:
            newhdr[key] = newimg.header[key]
        newhdu = fits.PrimaryHDU(newimg.hdu.data, newhdr)
        return newhdu
    else:
        for key in ['BMAJ', 'BMIN', 'BPA']:
            newhdr[key] = newimg[0].header[key]
        newhdu = fits.PrimaryHDU(newimg[0].hdu.data, newhdr)
        convhdu = fits.ImageHDU(newimg[1].hdu.data, newhdr)
        newhdr.remove('BUNIT', ignore_missing=True)
        wthdu = fits.ImageHDU(newimg[2].hdu.data, newhdr)
        return fits.HDUList([newhdu, convhdu, wthdu])
