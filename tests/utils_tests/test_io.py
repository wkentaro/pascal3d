import os.path as osp

import nose.tools
import numpy as np


this_dir = osp.dirname(osp.realpath(__file__))


def test_load_pcd():
    from pascal3d.utils.io import load_pcd

    pcd_file = osp.join(this_dir, 'data/bunny.pcd')

    points = load_pcd(pcd_file)
    nose.tools.assert_equal(points.shape, (397, 3))
    nose.tools.assert_equal(points.dtype, np.float64)
    np.testing.assert_array_almost_equal(
        points.mean(axis=0),
        [-0.029081, 0.102653, 0.027302],
    )
