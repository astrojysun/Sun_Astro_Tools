from __future__ import (
    division, print_function, absolute_import, unicode_literals)

import numpy as np


def num2latex(
        value, precision=2, scientific_notation='auto',
        with_dollar=True):
    """
    Generate LaTeX code for a number (with given precision).

    Parameters
    ----------
    value : int, float
        The number to be converted to LaTeX code
    precision : positive int, optional
        Number of precision digits (default=2)
    scientific_notation : {'auto', 'on', 'off'}, optional
        Whether to use scientific notation in the output code
    with_dollar : bool, optional
        Whether to include the dollar signs in the output code

    Returns
    -------
    code : string
        Corresponding LaTeX (math mode) code

    See Also
    --------
    `numpy.format_float_scientific`,
    `numpy.format_float_positional`
    """

    if not (precision > 0 and isinstance(precision, int)):
        raise ValueError("`precision` must be a positive integer")
    if scientific_notation not in ('auto', 'on', 'off'):
        raise ValueError(
            "Unrecognized value for `scientific_notation`")

    if not np.isfinite(value):
        code = r"\mathrm{" + str(value) + "}"
    elif value == 0:
        code = "0"
    else:
        dex = int(np.floor(np.log10(abs(value))))
        if scientific_notation == 'on':
            sci = True
        elif scientific_notation == 'auto' and (
                (abs(value) >= 1e3) or (0 < abs(value) < 1e-3) or
                (precision < dex + 1)):
            sci = True
        else:
            sci = False
        if sci:
            s = ("{:."+str(precision-1)+"e}").format(value)
            s_nval, s_exp = s.split('e')
            code = s_nval + r"\times10^{" + str(int(s_exp)) + "}"
        else:
            val = np.round(value/10**dex, precision-1) * 10**dex
            code = (
                "{:."+str(max(precision-dex-1, 0))+"f}").format(val)

    if with_dollar:
        return "$" + code + "$"
    else:
        return code
