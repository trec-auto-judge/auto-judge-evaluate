# This code is copied from https://github.com/eldrin/pyircore, as pyircore is not pip-installable at the moment for modern python versions.
import sys
import numpy as np
from scipy import stats
from collections.abc import Iterable


def is_numeric(obj):
    # from https://stackoverflow.com/a/500908
    attrs = ['__add__', '__sub__', '__mul__', '__truediv__', '__pow__']
    return all(hasattr(obj, attr) for attr in attrs)


def _check_types(x, arg_str):
    bad = False
    if isinstance(x, Iterable):
        x = np.array(x)
        if is_numeric(x) and x.size > 1:
            pass
        else:
            bad = True
    else:
        bad = True

    if bad:
        raise ValueError(
            '[ERROR] input {} must be a numeric vector'.format(arg_str)
        )
    return x


def check_types(x, y):
    x = _check_types(x, 'x')
    y = _check_types(y, 'y')
    if len(x) != len(y):
        raise ValueError('[ERROR] x and y must be of the same length')
    return x, y


def has_ties(x):
    return len(x) != len(set(x))


def check(x, y):
    x, y = check_types(x, y)
    if has_ties(x):
        raise ValueError('[ERROR] x contains ties')
    if has_ties(y):
        raise ValueError('[ERROR] y contains ties.')
    return x, y


def check_a(x, y):
    x, y = check_types(x, y)
    if has_ties(x):
        raise ValueError('[ERROR] x contains ties')
    return x, y


def check_b(x, y):
    return check_types(x, y)


def check_inputs(check_type='default'):
    def real_check_inputs(func):
        def wrapper(x, y, decreasing=True, *args, **kwargs):
            if check_type == 'default':
                x, y = check(x, y)
            elif check_type == 'a':
                x, y = check_a(x, y)
            elif check_type == 'b':
                x, y = check_b(x, y)

            if decreasing:
                return func(-x, -y, decreasing=False, *args, **kwargs)
            else:
                return func(x, y, *args, **kwargs)
        return wrapper
    return real_check_inputs

@check_inputs('b')
def tauap_b(x, y, decreasing=True):
    """AP-b Rank Correlation Coefficient

    Inputs:
        x (Iterable of numeric): input vector
        y (Iterable of numeric): another vector for comparison
    
    Returns:
        float: the correlation coefficient.   
    """
    return (tauap_b_ties(x, y) + tauap_b_ties(y, x)) / 2


def tauap_b_ties(x, y, decreasing=True):
    """Helper function"""
    rx = stats.rankdata(x)
    ry = stats.rankdata(y, 'ordinal')  # ties.method = 'first'
    p = stats.rankdata(y, 'min') - 1 # ties.method = 'min'
    return _tauap_b_ties(rx, ry, p)

def _tauap_b_ties(rx, ry, p):
    """Helper function for faster computation"""
    c_all = 0
    n_not_top = 0
    for i in range(len(p)):
        # ignore the first items group
        if p[i] == 0:
            continue
        n_not_top += 1

        # count concordants above the pivot's tie group
        c_above = 0
        for j in range(len(p)):
            if p[j] >= p[i]:
                continue

            sx = np.sign(rx[i] - rx[j])
            sy = np.sign(ry[i] - ry[j])

            if sx == sy:
                c_above += 1
        c_all += c_above / p[i]  # divide by p-1 instead of i-1
    if n_not_top == 0:
        # All elements are tied at the top rank - no meaningful ranking to compare
        print("Warning: tauap_b cannot compute correlation when all elements are tied (returning 0.0)", file=sys.stderr)
        return 0.0
    return 2 / n_not_top * c_all - 1