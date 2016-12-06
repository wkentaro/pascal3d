import tempfile

import nose.tools
import numpy as np
import requests


def test_load_pcd():
    from pascal3d.utils.io import load_pcd

    pcd_file = 'http://raw.githubusercontent.com/PointCloudLibrary/pcl/pcl-1.8.0/test/bunny.pcd'  # NOQA
    res = requests.get(pcd_file)
    assert res.status_code == 200

    tmp_file = tempfile.mktemp() + '.pcd'
    with open(tmp_file, 'w') as f:
        f.write(res.content)

    points = load_pcd(tmp_file)
    nose.tools.assert_equal(points.shape, (397, 3))
    nose.tools.assert_equal(points.dtype, np.float64)
    np.testing.assert_array_almost_equal(
        points.mean(axis=0),
        [-0.029081, 0.102653, 0.027302],
    )
