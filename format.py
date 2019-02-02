from __future__ import (
    division, print_function, absolute_import, unicode_literals)

import numpy as np


def num2latex(value, precision=2, with_dollar=True):
    """
    Generate LaTeX code for a number (with given precision).

    Parameters
    ----------
    value : int, float
        The number to be converted to LaTeX code
    precision : positive int, optional
        Number of precision digits (default=2)
    with_dollar : bool, optional
        Whether to include the dollar signs in the output code

    Returns
    -------
    code : string
        Corresponding LaTeX (math mode) code
    """

    if not (precision > 0 and isinstance(precision, int)):
        raise ValueError("`precision` must be a positive integer")

    if not np.isfinite(value):
        code = r"\mathrm{" + str(value) + "}"
    else:
        if value == 0:
            dex = 0
        else:
            dex = int(np.floor(np.log10(abs(value))))
        if ((abs(value) >= 1e3) or (abs(value) < 1e-3) or
            (precision < np.max(dex, 0)+1)):
            s = ("{:."+str(precision-1)+"e}").format(value)
            s_nval, s_exp = s.split('e')
            code = (
                s_nval + r" \times 10^{" + str(int(s_exp)) + "}")
        else:
            code = ("{:."+str(precision-dex-1)+"f}").format(value)

    if with_dollar:
        return "$" + code + "$"
    else:
        return code
