import nose
import numpy as np
import unittest

import sympy


class TestIntersect3DRayTriangle(unittest.TestCase):

    def setUp(self):
        self.ray0 = np.random.randint(-1, 1, size=(10, 3))
        self.ray1 = np.random.randint(-10, 10, size=(10, 3))
        self.tri0 = np.random.randint(-30, 30, size=(10, 3))
        self.tri1 = np.random.randint(-30, 30, size=(10, 3))
        self.tri2 = np.random.randint(-30, 30, size=(10, 3))

    def test_intersection(self):
        for args in zip(self.ray0, self.ray1, self.tri0, self.tri1, self.tri2):
            args = [np.array([arg], dtype=np.float64) for arg in args]
            self._check_intersection(*args)

    def _check_intersection(self, ray0, ray1, tri0, tri1, tri2):
        from pascal3d.utils.geometry import intersect3d_ray_triangle
        n_rays = len(ray0)
        assert len(ray1) == n_rays
        assert len(tri0) == n_rays
        assert len(tri1) == n_rays
        assert len(tri2) == n_rays

        l1 = sympy.Line3D(sympy.Point3D(*ray0.tolist()),
                          sympy.Point3D(*ray1.tolist()))
        p1 = sympy.Plane(sympy.Point3D(*tri0.tolist()),
                         sympy.Point3D(*tri1.tolist()),
                         sympy.Point3D(*tri2.tolist()))
        t1 = sympy.Triangle(sympy.Point3D(*tri0.tolist()),
                            sympy.Point3D(*tri1.tolist()),
                            sympy.Point3D(*tri2.tolist()))
        i1 = p1.intersection(l1)
        if len(i1) == 0:
            ret1 = False
        else:
            ret1 = t1.encloses(i1[0])

        ret2, i2 = intersect3d_ray_triangle(ray0, ray1, tri0, tri1, tri2)
        nose.tools.assert_equal(ret2.shape, (1,))
        nose.tools.assert_equal(ret2[0] == 1, ret1)
        nose.tools.assert_equal(i2.shape, (1, 3))
        if ret1:
            np.testing.assert_allclose(
                i2[0], map(float, [i1[0].x, i1[0].y, i1[0].z]))
        else:
            nose.tools.assert_equal(np.isnan(i2[0]).sum(), 3)


def test_triangle_to_aabb():
    from pascal3d.utils.geometry import triangle_to_aabb

    tri0 = np.array([[1, 0, -2]], dtype=np.float64)
    tri1 = np.array([[-1, -1, -2]], dtype=np.float64)
    tri2 = np.array([[-1, 1, 1]], dtype=np.float64)

    lb, rt = triangle_to_aabb(tri0, tri1, tri2)
    nose.tools.assert_equal(lb.shape, (1, 3))
    nose.tools.assert_equal(rt.shape, (1, 3))
    np.testing.assert_allclose(lb[0], [-1, -1, -2])
    np.testing.assert_allclose(rt[0], [1, 1, 1])
