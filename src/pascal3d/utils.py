import math

import numpy as np
import six


def get_transformation_matrix(azimuth, elevation, distance):
    if distance == 0:
        return None

    # camera center
    C = np.zeros((3, 1))
    C[0] = distance * math.cos(elevation) * math.sin(azimuth)
    C[1] = -distance * math.cos(elevation) * math.cos(azimuth)
    C[2] = distance * math.sin(elevation)

    # rotate coordinate system by theta is equal to rotating the model by theta
    azimuth = -azimuth
    elevation = - (math.pi / 2 - elevation)

    # rotation matrix
    Rz = np.array([
        [math.cos(azimuth), -math.sin(azimuth), 0],
        [math.sin(azimuth), math.cos(azimuth), 0],
        [0, 0, 1],
    ])  # rotation by azimuth
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(elevation), -math.sin(elevation)],
        [0, math.sin(elevation), math.cos(elevation)],
    ])  # rotation by elevation
    R_rot = np.dot(Rx, Rz)
    R = np.hstack((R_rot, np.dot(-R_rot, C)))
    R = np.vstack((R, [0, 0, 0, 1]))

    return R


def transform_to_camera_frame(
        x3d,
        azimuth,
        elevation,
        distance,
        ):
    R = get_transformation_matrix(azimuth, elevation, distance)
    if R is None:
        return []

    # get points in camera frame
    x3d_ = np.hstack((x3d, np.ones((len(x3d), 1)))).T
    x3d_camframe = np.dot(R[:3, :4], x3d_).T

    return x3d_camframe


def project_vertices_3d_to_2d(
        x3d,
        azimuth,
        elevation,
        distance,
        focal,
        theta,
        principal,
        viewport,
        ):
    R = get_transformation_matrix(azimuth, elevation, distance)
    if R is None:
        return []

    # perspective project matrix
    # however, we set the viewport to 3000, which makes the camera similar to
    # an affine-camera.
    # Exploring a real perspective camera can be a future work.
    M = viewport
    P = np.array([[M * focal, 0, 0],
                  [0, M * focal, 0],
                  [0, 0, -1]]).dot(R[:3, :4])

    # project
    x3d_ = np.hstack((x3d, np.ones((len(x3d), 1)))).T
    x2d = np.dot(P, x3d_)
    x2d[0, :] = x2d[0, :] / x2d[2, :]
    x2d[1, :] = x2d[1, :] / x2d[2, :]
    x2d = x2d[0:2, :]

    # rotation matrix 2D
    R2d = np.array([[math.cos(theta), -math.sin(theta)],
                    [math.sin(theta), math.cos(theta)]])
    x2d = np.dot(R2d, x2d).T

    # transform to image coordinate
    x2d[:, 1] *= -1
    x2d = x2d + np.repeat(principal[np.newaxis, :], len(x2d), axis=0)

    return x2d


def get_camera_polygon(height, width, theta, focal, principal, viewport):
    # rotate the camera model
    R2d = np.array([[math.cos(theta), -math.sin(theta)],
                    [math.sin(theta), math.cos(theta)]])
    # projection matrix
    M = viewport
    P = np.array([
        [M * focal, 0, 0],
        [0, M * focal, 0],
        [0, 0, -1],
    ])

    x0 = np.array([0, 0, 0], dtype=np.float64)

    # rotate and project the points
    x = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height],
    ], dtype=np.float64)
    x -= principal
    x[:, 1] *= -1
    x = np.dot(np.linalg.inv(R2d), x.T).T
    x = np.hstack((x, np.ones((len(x), 1), dtype=np.float64)))
    x = np.dot(np.linalg.inv(P), x.T).T

    x = np.vstack((x0, x))

    return x


def load_pcd(pcd_file):
    """Load xyz pcd file.

    Parameters
    ----------
    pcd_file: str
        PCD filename.
    """
    points = []
    n_points = None
    with open(pcd_file, 'r') as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue

            meta_fields = [
                'VERSION',
                'FIELDS',
                'SIZE',
                'TYPE',
                'COUNT',
                'WIDTH',
                'HEIGHT',
                'VIEWPOINT',
                'POINTS',
                'DATA',
            ]
            meta = line.strip().split(' ')
            meta_header, meta_contents = meta[0], meta[1:]
            if meta_header == 'FIELDS':
                assert meta_contents == ['x', 'y', 'z']
            elif meta_header == 'POINTS':
                n_points = int(meta_contents[0])
            if meta_header in meta_fields:
                continue

            x, y, z = map(float, line.split(' '))
            points.append((x, y, z))

    points = np.array(points)

    if n_points is not None:
        assert len(points) == n_points
        assert points.shape[1] == 3

    return points


def plane_points3(p0, p1, p2):
    v0 = p1 - p0
    v1 = p2 - p0
    p_no = np.cross(v0, v1)
    p_co = p0
    return p_co, p_no


def intersection_line_with_plane(p0, p1, p_co, p_no, epsilon=1e-6):
    """Compute intersection point between line and plane.

    Parameters
    ----------
    p0, p1: Define the line.
    p_co, p_no: Define the plane.
        p_co is a point on the plane (plane coordinate).
        p_no is a normal vector defining the plane direction;
             (does not need to be normalized).
    """
    assert p0.ndim == p1.ndim == p_co.ndim == p_no.ndim == 2
    assert p0.shape[1] == 3
    assert p1.shape[1] == 3
    assert p_co.shape[1] == 3
    assert p_no.shape[1] == 3
    assert len(p0) == len(p1) == len(p_co) == len(p_no)

    def vectors_dot(a, b):
        return a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1] + a[:, 2] * b[:, 2]

    u = p1 - p0
    dot = vectors_dot(p_no, u)

    mask = np.abs(dot) < epsilon

    w = p0 - p_co
    fac = - vectors_dot(p_no, w) / dot
    u = u * fac
    p_isect = p0 + u
    p_isect[mask] = np.nan

    return p_isect
