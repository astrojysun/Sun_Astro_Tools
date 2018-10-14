from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from spectral_cube import SpectralCube, Projection


def clean_header(hdr, remove_keys=[], keep_keys=[]):
    """
    Clean a fits header and retain only the necessary keywords.

    Parameters
    ----------
    hdr : fits header object
        Header object to be cleaned
    remove_keys : iterable
        List of keys to remove before feeding the header to WCS
    keep_keys : iterable
        List of keys to keep

    Returns
    -------
    newhdr : fits header object
        Cleaned header
    """
    remove_keys += ['WCSAXES', 'OBSGEO-X', 'OBSGEO-Y', 'OBSGEO-Z',
                    'OBS-RA', 'OBS-DEC', 'MJD-OBS', 'DATE-OBS']
    newhdr = hdr.copy()
    for key in remove_keys:
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
    from reproject import reproject_interp
        
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


def convolve_image_hdu(inhdu, newbeam, append_raw=False, **kwargs):
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
        from .cube import convolve_projection as convolve
    else:
        oldimg = SpectralCube(inhdu.data, wcs=WCS(inhdu.header),
                              header=inhdu.header)
        from .cube import convolve_cube as convolve
    newimg = convolve(oldimg, newbeam, append_raw=append_raw,
                      **kwargs)
    newhdr = inhdu.header.copy(strip=True)
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
