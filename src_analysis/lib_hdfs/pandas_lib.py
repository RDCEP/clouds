# _*_ coding : utf-8  _*_

import pandas as pd

def highlight_max(s, _abs=False):
    '''
    highlight the maximum in a Series yellow.
    '''
    if not _abs:
        is_max = s == s.max()
    else:
        is_max = abs(s) == abs(s).max()
    return ['background-color: red' if v else '' for v in is_max]


def highlight_min(s, _abs=False):
    '''
    highlight the maximum in a Series yellow.
    '''
    if not _abs:
        is_min = s == s.min()
    else:
        is_min = abs(s) == abs(s).min()
    return ['background-color: blue' if v else '' for v in is_min]

def highlight_zero(s, nround=1, _abs=False):
    '''
    highlight the near 0 value in a Series yellow.
    '''
    return 
    if not _abs:
        is_min = s == s.min()
    else:
        is_min = abs(s) == abs(s).min()
    return ['background-color: blue' if v else '' for v in is_min]
