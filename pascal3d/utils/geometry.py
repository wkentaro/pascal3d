import math

import numpy as np

from pascal3d.utils import _geometry


intersect3d_ray_triangle = _geometry.intersect3d_ray_triangle
raytrace_camera_frame_on_triangles = \
    _geometry.raytrace_camera_frame_on_triangles


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


def project_points_3d_to_2d(
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


def project_points_2d_to_3d(x2d, theta, focal, principal, viewport):
    x2d = x2d.copy()
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
    x2d -= principal
    x2d[:, 1] *= -1
    x2d = np.dot(np.linalg.inv(R2d), x2d.T).T
    x2d = np.hstack((x2d, np.ones((len(x2d), 1), dtype=np.float64)))
    x2d = np.dot(np.linalg.inv(P), x2d.T).T
    return x2d


def get_camera_polygon(height, width, theta, focal, principal, viewport):
    x0 = np.array([0, 0, 0], dtype=np.float64)

    # project the 3D points
    x = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height],
    ], dtype=np.float64)
    x = project_points_2d_to_3d(x, theta, focal, principal, viewport)

    x = np.vstack((x0, x))

    return x


def triangle_to_aabb(tri0, tri1, tri2):
    """Convert triangle to AABB.

    Parameters
    ----------
    tri0, tri1, tri2: numpy.ndarray (N, 3)
        Triangle vectors.

    Returns
    -------
    lb, rt: numpy.ndarray (N, 3)
        Min point (left-bottom: lb) and max point (right-top: rt) of
        AABB (axis-aligned bounding box).
    """
    tri = np.array(zip(tri0, tri1, tri2))
    lb = np.min(tri, axis=1)
    rt = np.max(tri, axis=1)
    return lb, rt


def intersect3d_ray_aabb(ray0, ray1, lb, rt):
    """Compute intersection between 3D ray and AABB.

    Parameters
    ----------
    ray0: numpy.ndarray (3,)
        3D point of ray's origin.
    ray1: numpy.ndarray (3,)
        3D point of ray's end.

    Returns
    -------
    flag: bool
        True if it intersects, else False.
    intersection: numpy.ndarray (3,)
        Point of intersection.
    """
    ray_dir = ray1 - ray0
    ray_dir = ray_dir / np.linalg.norm(ray_dir)

    dirfrac = 1.0 / ray_dir

    t1 = (lb[0] - ray0[0]) * dirfrac[0]
    t2 = (rt[0] - ray0[0]) * dirfrac[0]
    t3 = (lb[1] - ray0[1]) * dirfrac[1]
    t4 = (rt[1] - ray0[1]) * dirfrac[1]
    t5 = (lb[2] - ray0[2]) * dirfrac[2]
    t6 = (rt[2] - ray0[2]) * dirfrac[2]

    tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6))
    tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6))

    if tmax < 0:
        flag = False
        t = tmax
    elif tmin > tmax:
        flag = False
        t = tmax
    else:
        flag = True
        t = tmin

    intersection = ray0 + t * ray_dir
    return flag, intersection
