from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from spectral_cube import Projection


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


def regrid_hdu(inhdu, newhdr, keep_old_header_keys=[], **kwargs):
    """
    Regrid a FITS HDU (either 2D or 3D) according to the input header.

    This is just a simple wrapper around `reproject.reproject_interp`.

    Parameters
    ----------
    inhdu : FITS HDU object
        Input FITS HDU
    newhdr : FITS header object
        New header that defines the new grid
    keep_old_header_keys : array-like, optional
        List of keys to keep from the old header in 'inhdu'
        Default is an empty list.
    **kwargs
        Keyword arguments to be passed to `reproject.reproject_interp`

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
    for key in keep_old_header_keys:
        hdr[key] = inhdu.header[key]

    data_proj, footprint = reproject_interp(inhdu, hdr, **kwargs)
    data_proj[~footprint.astype('?')] = np.nan
    newhdu = inhdu.__class__(data_proj, hdr)
    
    return newhdu


def convolve_image_hdu(inhdu, newbeam, res_tol=0.0, min_coverage=0.8,
                       append_raw=False, verbose=False,
                       suppress_error=False):
    """
    Convolve a FITS 2D image to a specified beam.

    Parameters
    ----------
    inhdu : FITS HDU object
        Input FITS HDU
    newbeam : radio_beam.Beam object
        Target beam to convolve to
    res_tol : float, optional
        Tolerance on the difference between input/output resolution
        By default, a convolution is performed on the input image
        whenever its native resolution is different from (sharper than)
        the target resolution. Use this keyword to specify a tolerance
        on resolution, within which no convolution will be performed.
        For example, res_tol=0.1 will allow a 10% tolerance.
    min_coverage : float or None, optional
        When the convolution meets NaN values or edges, the output is
        calculated based on beam-weighted average. This keyword specifies
        the minimum beam covering fraction of valid (np.finite) values.
        All pixels with less beam covering fraction will be assigned NaNs.
        Default is 80% beam covering fraction (min_coverage=0.8).
        If the user would rather use the interpolation strategy in
        `astropy.convolution.convolve_fft`, set this keyword to None.
        Note that the NaN pixels will be kept as NaN.
    append_raw : bool, optional
        Whether to append the raw convolved image and weight image
        Default is not to append.
    verbose : bool, optional
        Whether to print the detailed processing information in terminal
        Default is to not print.
    suppress_error : bool, optional
        Whether to suppress the error when convolution is unsuccessful
        Default is to not suppress.

    Returns
    -------
    outhdu : FITS HDU or HDUList object
        Convolved HDU (when append_raw=False), or HDUList comprising
        3 HDUs (when append_raw=True)
    """

    from functools import partial
    from astropy.convolution import convolve_fft

    if min_coverage is None:
        # Skip coverage check and preserve NaN values.
        # This uses the default interpolation strategy
        # implemented in 'astropy.convolution.convolve_fft'
        convolve_func = partial(convolve_fft, preserve_nan=True)
    else:
        # Do coverage check to determine the mask on the output
        convolve_func = partial(convolve_fft, nan_treatment='fill',
                                boundary='fill', fill_value=0.,
                                allow_huge=True, quiet=~verbose)

    if inhdu.header['NAXIS'] != 2:
        raise ValueError("Input HDU is not a 2D image (NAXIS != 2)")

    if (res_tol > 0) and (newbeam.major != newbeam.minor):
        raise ValueError("You cannot specify a non-zero resolution "
                         "torelance if the target beam is not round")

    # read in specified FITS HDU
    oldimg = Projection.from_hdu(inhdu)
        
    tol = newbeam.major * np.array([1-res_tol, 1+res_tol])
    if ((tol[0] < oldimg.beam.major < tol[1]) and
        (tol[0] < oldimg.beam.minor < tol[1])):
        if verbose:
            print("Native resolution within tolerance - "
                  "Copying original HDU...")
        my_append_raw = False
        newimg = oldimg.copy()
    else:
        if verbose:
            print("Convolving HDU...")
        try:
            convimg = oldimg.convolve_to(newbeam,
                                         convolve=convolve_func)
            if min_coverage is not None:
                # divide the raw convolved image by the weight image
                my_append_raw = True
                wtarr = np.isfinite(inhdu.data).astype('float')
                wtimg = Projection(wtarr,
                                   wcs=oldimg.wcs, beam=oldimg.beam)
                wtimg = wtimg.convolve_to(newbeam,
                                          convolve=convolve_func)
                newimg = convimg / wtimg.hdu.data
                # mask all pixels w/ weight smaller than min_coverage
                threshold = min_coverage * u.dimensionless_unscaled
                newimg[wtimg < threshold] = np.nan
            else:
                my_append_raw = False
                newimg = convimg
        except ValueError as err:
            if suppress_error:
                return
            else:
                raise ValueError(
                    "Unsuccessful convolution: {}\nOld: {}\nNew: {}"
                    "".format(err, oldimg.beam, newbeam))
    newhdr = inhdu.header.copy(strip=True)

    # construct output HDUList
    newdata = newimg.hdu.data
    for key in ['BMAJ', 'BMIN', 'BPA']:
        newhdr[key] = newimg.header[key]
    newhdr.remove('WCSAXES', ignore_missing=True)
    newhdu = fits.PrimaryHDU(newdata, newhdr)
    if append_raw and my_append_raw:
        convhdu = fits.ImageHDU(convimg.hdu.data, newhdr)
        newhdr.remove('BUNIT')
        wthdu = fits.ImageHDU(wtimg.hdu.data, newhdr)
        return fits.HDUList([newhdu, convhdu, wthdu])
    else:
        return newhdu
