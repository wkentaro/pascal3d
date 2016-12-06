import numpy as np
cimport numpy as np


DTYPE = np.float
ctypedef np.float_t DTYPE_t


# http://geomalgorithms.com/a06-_intersect-2.html
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
    cdef unsigned int N = len(ray0)
    cdef unsigned int i
    cdef np.ndarray[DTYPE_t, ndim=2] intersection = \
        np.zeros((N, 3), dtype=DTYPE)
    cdef np.ndarray[np.int_t, ndim=1] flags = np.zeros((N,), dtype=np.int)
    cdef np.ndarray[DTYPE_t, ndim=1] u, v, n, w
    cdef np.ndarray[DTYPE_t, ndim=1] ray_dir, w0
    cdef DTYPE_t a, b, r
    cdef DTYPE_t uu, uv, vv, wu, wv, D, s, t

    for i in range(N):
        u = tri1[i] - tri0[i]
        v = tri2[i] - tri0[i]
        n = np.cross(u, v)
        if (n == 0).sum() == 3:
            flags[i] = -1
            intersection[i, ...] = np.nan
            continue

        ray_dir = ray1[i] - ray0[i]
        w0 = ray0[i] - tri0[i]
        a = - np.dot(n, w0)
        b = np.dot(n, ray_dir)
        if np.abs(b) < 1e-6:
            if a == 0:
                # ray lies in triangle plane
                flags[i] = 2
                intersection[i, ...] = np.nan
                continue
            # ray disjoint from plane
            flags[i] = 0
            intersection[i, ...] = np.nan
            continue

        r = a / b
        if r < 0:  # TODO(wkentaro): r > 1.0 flag for segment.
            # ray goes away from triangle
            flags[i] = 0
            intersection[i, ...] = np.nan
            continue

        I = ray0[i] + r * ray_dir

        uu = np.dot(u, u)
        uv = np.dot(u, v)
        vv = np.dot(v, v)
        w = I - tri0[i]
        wu = np.dot(w, u)
        wv = np.dot(w, v)
        D = uv * uv - uu * vv

        s = (uv * wv - vv * wu) / D
        if s < 0 or s > 1:
            flags[i] = 0
            intersection[i] = I
            continue
        t = (uv * wu - uu * wv) / D
        if t < 0 or (s + t) > 1:
            flags[i] = 0
            intersection[i] = I
            continue

        flags[i] = 1
        intersection[i] = I

    return flags, intersection
