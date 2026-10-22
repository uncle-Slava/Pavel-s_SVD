"""Moved to svdlab.scorecard; this name keeps old imports and commands working.

New code: `from svdlab import check, compare` and `python -m svdlab check my_svd.py`.
"""
from svdlab.scorecard import *  # noqa: F401,F403
from svdlab.scorecard import (DEFAULT_SUITE, QUICK_SUITE, COLUMNS, TOLERANCE_FACTOR, TAIL_RATIO_LIMIT,  # noqa: F401
                          builtin, check, compare, evaluate, gpu_check, main, make_case, resolve, save,
                          summary, table)

if __name__ == '__main__':
    main()
