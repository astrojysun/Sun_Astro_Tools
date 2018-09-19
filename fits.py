from __future__ import (division, print_function, absolute_import,
                        unicode_literals)


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
    from astropy.wcs import WCS
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
