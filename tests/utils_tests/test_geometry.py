import nose
import numpy as np
import sympy as sp
import nose.tools


def test_intersect3d_ray_triangle():
    from pascal3d.utils.geometry import intersect3d_ray_triangle

    ray0 = np.array([[0, 0, 0]], dtype=np.float64)
    ray1 = np.array([[0, 0, -1]], dtype=np.float64)
    tri0 = np.array([[1, 0, -2]], dtype=np.float64)
    tri1 = np.array([[-1, -1, -2]], dtype=np.float64)
    tri2 = np.array([[-1, 1, -2]], dtype=np.float64)

    ret, p_inter = intersect3d_ray_triangle(
        ray0, ray1, tri0, tri1, tri2)
    nose.tools.assert_equal(ret.shape, (1,))
    nose.tools.assert_equal(ret[0], 1)
    nose.tools.assert_equal(p_inter.shape, (1, 3))
    np.testing.assert_allclose(p_inter[0], [0, 0, -2])
