from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

from functools import partial
import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.convolution import convolve_fft
from radio_beam import Beam
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


def convolve_image(infile, newres, hdu=0, res_tol=0.0, min_coverage=0.8,
                   writefile='', append_raw=False, verbose=False):
    """
    Convolve a FITS 2D image to a specified resolution.

    Parameters
    ----------
    infile : string
        Input FITS image
    newres : astropy Quantity object
        Target resolution to convolve to
    hdu : int, optional
        Index of the HDU to read in (default: 0)
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
    writefile : string, optional
        Name of the output FITS file. If not specified, the convolved
        HDU / HDUList object itself will be returned.
    append_raw : bool, optional
        Whether to append the raw convolved image and weight image.
        Default is not to append.
    verbose : bool, optional
        Whether to print the detailed processing information in terminal.

    Returns
    -------
    If the keyword 'writefile' is specified, return its value,
    otherwise return the convolved HDU or HDUList object.
    """

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

    # create target beam
    newres_u = (newres * u.dimensionless_unscaled).unit
    if not newres_u.is_equivalent(u.deg):
        raise ValueError("`newres` must carry an angle unit")
    newbeam = Beam(major=newres, minor=newres, pa=0*u.deg)

    # read in specified FITS HDU
    hdul = fits.open(infile)
    this_hdu = hdul[hdu]
    oldimg = Projection.from_hdu(this_hdu)
    wtarr = np.isfinite(this_hdu.data).astype('float')
    wtimg = Projection(wtarr, wcs=oldimg.wcs, beam=oldimg.beam)
        
    tol = [newres * (1 - res_tol), newres * (1 + res_tol)]
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
            print("Unsuccessful convolution: ", err, sep='')
            print("Old", oldimg.beam)
            print("New", newbeam)
            hdul.close()
            return
    newhdr = this_hdu.header.copy(strip=True)
    hdul.close()

    # construct output HDUList
    newdata = newimg.hdu.data
    for key in ['BMAJ', 'BMIN', 'BPA']:
        newhdr[key] = newimg.header[key]
    newhdu = fits.PrimaryHDU(newdata, newhdr)
    if append_raw and my_append_raw:
        convhdu = fits.ImageHDU(convimg.hdu.data, newhdr)
        newhdr.remove('BUNIT')
        wthdu = fits.ImageHDU(wtimg.hdu.data, newhdr)
        hdul = fits.HDUList([newhdu, convhdu, wthdu])
    else:
        hdul = fits.HDUList([newhdu])

    if writefile:
        if verbose:
            if append_raw and my_append_raw:
                print("Writing convolved HDUs to disk...")
            else:
                print("Writing convolved HDU to disk...")
        hdul.writeto(writefile, overwrite=True)
        return writefile
    else:
        if append_raw and my_append_raw:
            if verbose:
                print("No 'writefile' specified. "
                      "Returning convolved HDUs...")
            return hdul
        else:
            if verbose:
                print("No 'writefile' specified. "
                      "Returning convolved HDU...")
            return newhdu
