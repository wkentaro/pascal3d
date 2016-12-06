import numpy as np
import sympy as sp

from .. import utils


def test_plane_points3():
    q0 = np.array([0.5, 0.1, 0.2])
    q1 = np.array([0.4, 0.2, 0.4])
    q2 = np.array([0.7, 0.3, 0.1])
    p_co, p_no = utils.plane_points3(q0, q1, q2)

    q0_ = sp.Point3D(*q0)
    q1_ = sp.Point3D(*q1)
    q2_ = sp.Point3D(*q2)
    plane1 = sp.Plane(q0_, q1_, q2_)

    np.testing.assert_allclose(p_co.tolist(), map(float, plane1.p1))
    np.testing.assert_allclose(p_no.tolist(), map(float, plane1.normal_vector))

    Q0 = np.repeat(q0[np.newaxis, :], 3, axis=0)
    Q1 = np.repeat(q1[np.newaxis, :], 3, axis=0)
    Q2 = np.repeat(q2[np.newaxis, :], 3, axis=0)
    Q_co, Q_no = utils.plane_points3(Q0, Q1, Q2)
    for p_co, p_no in zip(Q_co, Q_no):
        np.testing.assert_allclose(
            p_co.tolist(), map(float, plane1.p1))
        np.testing.assert_allclose(
            p_no.tolist(), map(float, plane1.normal_vector))


def test_intersection_line_with_plane():
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([0.25, 0.65, 0.17])
    q0 = np.array([0.5, 0.1, 0.2])
    q1 = np.array([0.4, 0.2, 0.4])
    q2 = np.array([0.7, 0.3, 0.1])

    # using numpy
    p_co, p_no = utils.plane_points3(q0, q1, q2)
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
    line1 = sp.Line3D(p0, p1)
    plane1 = sp.Plane(q0, q1, q2)
    # test about intersection
    inter = plane1.intersection(line1)
    inter_p2 = map(float, inter[0])
    np.testing.assert_allclose(inter_p1, inter_p2)
