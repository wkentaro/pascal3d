import numpy as np
import sympy as sp

from .. import utils


def test_intersection_line_with_plane():
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([0.25, 0.65, 0.17])
    q0 = np.array([0.5, 0.1, 0.2])
    q1 = np.array([0.4, 0.2, 0.4])
    q2 = np.array([0.7, 0.3, 0.1])

    # using numpy
    v1 = q1 - q0
    v2 = q2 - q0
    p_co = q0
    p_no = np.cross(v1, v2)
    inter = utils.intersection_line_with_plane(
        p0[np.newaxis, :],
        p1[np.newaxis, :],
        p_co[np.newaxis, :],
        p_no[np.newaxis, :],
    )
    inter_p1 = inter[0]

    # using sympy
    p0 = sp.Point3D(*p0)
    p1 = sp.Point3D(*p1)
    q0 = sp.Point3D(*q0)
    q1 = sp.Point3D(*q1)
    q2 = sp.Point3D(*q2)
    l1 = sp.Line3D(p0, p1)
    p1 = sp.Plane(q0, q1, q2)
    # test about plane definition
    np.testing.assert_allclose(p_co.tolist(), map(float, p1.p1))
    np.testing.assert_allclose(p_no.tolist(), map(float, p1.normal_vector))
    # test about intersection
    inter = p1.intersection(l1)
    inter_p2 = map(float, inter[0])
    np.testing.assert_allclose(inter_p1, inter_p2)
