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


def label_colormap(n_labels):
    """Returns appropriate colormap for the specified number of labels.

    Parameters
    ----------
    n_labels: int
        Number of labels.

    Returns
    -------
    colormap: array of float, shape (n_labels, 3)
        Colormap generated for the specified number of labels.
    """

    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    colormap = np.zeros((n_labels, 3))
    for i in six.moves.range(0, n_labels):
        id = i
        r, g, b = 0, 0, 0
        for j in six.moves.range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7-j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7-j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7-j))
            id = (id >> 3)
        colormap[i, 0] = r
        colormap[i, 1] = g
        colormap[i, 2] = b
    colormap = colormap.astype(np.float32) / 255
    return colormap
