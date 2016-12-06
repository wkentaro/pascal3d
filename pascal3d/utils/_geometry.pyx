import numpy as np
cimport numpy as np


DTYPE = np.float
ctypedef np.float_t DTYPE_t


def intersect3d_ray_triangle(
        np.ndarray[DTYPE_t, ndim=2] ray0,
        np.ndarray[DTYPE_t, ndim=2] ray1,
        np.ndarray[DTYPE_t, ndim=2] tri0,
        np.ndarray[DTYPE_t, ndim=2] tri1,
        np.ndarray[DTYPE_t, ndim=2] tri2,
        ):
    """
    Parameters
    ----------
    ray0, ray1: (N, 3) Ray vectors.
    tri0, tri1, tri2: (N, 3) Triangle vectors.

    Returns
    -------
    .. TODO(wkentaro)
    """
    N = ray0.shape[0]

    flags = np.zeros((N,), dtype=np.int32)
    flags[...] = 1

    intersection = np.zeros((N, 3), dtype=np.float64)

    u = tri1 - tri0
    v = tri2 - tri0
    n = np.cross(u, v)
    invalid = (n == 0).sum(axis=-1) == 3
    flags[invalid] = -1
    intersection[invalid, ...] = np.nan

    ray_dir = ray1 - ray0
    w0 = ray0 - tri0
    a = - np.dot(n, w0.T)
    b = np.dot(n, ray_dir.T)
    invalid = (np.abs(b) < 1e-6)[:, 0]
    invalid_2 = np.bitwise_and(invalid, (a == 0)[:, 0])
    flags[invalid_2] = 2
    intersection[invalid_2, ...] = np.nan
    invalid_0 = np.bitwise_and(invalid, (a != 0)[:, 0])
    flags[invalid_0] = 0
    intersection[invalid_0, ...] = np.nan

    r = a / b
    invalid = (r < 0)[:, 0]
    flags[invalid] = 0
    intersection[invalid, ...] = np.nan

    # compute intersection point
    I = ray0 + r * ray_dir

    # compute inside/outside of plane
    uu = np.dot(u, u.T)
    uv = np.dot(u, v.T)
    vv = np.dot(v, v.T)
    w = I - tri0
    wu = np.dot(w, u.T)
    wv = np.dot(w, v.T)
    D = uv * uv - uu * vv
    s = (uv * wv - vv * wu) / D
    invalid = np.bitwise_or((s < 0)[:, 0], (s > 1)[:, 0])
    flags[invalid] = 0
    intersection[invalid] = I[invalid]
    t = (uv * wu - uu * wv) / D
    invalid = np.bitwise_or((t < 0)[:, 0], (t > 1)[:, 0])
    flags[invalid] = 0
    intersection[invalid] = I[invalid]

    # set valid points
    intersection[flags == 1] = I[flags == 1]

    return flags, intersection
