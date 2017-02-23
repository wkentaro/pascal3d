from __future__ import print_function

import sys

import matplotlib
import numpy as np


def colorize_depth(depth, min_value, max_value):
    if np.isnan(max_value):
        print('WARNING: max_value is inf.', file=sys.stderr)
    colorized = depth.copy()
    nan_mask = np.isnan(colorized)
    colorized[nan_mask] = 0
    colorized = 1. * (colorized - min_value) / (max_value - min_value)
    colorized = matplotlib.cm.jet(colorized)[:, :, :3]
    colorized = (colorized * 255).astype(np.uint8)
    colorized[nan_mask] = (0, 0, 0)
    return colorized
