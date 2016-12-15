#!/usr/bin/env python

import itertools
import time

import matplotlib.pyplot as plt
import numpy as np

import pascal3d


def aabb_to_octopoints(lb, rt):
    dim = rt - lb
    pt0 = lb
    pt1 = lb + [dim[0], 0, 0]
    pt2 = lb + [dim[0], 0, dim[2]]
    pt3 = lb + [0, 0, dim[2]]
    pt4 = pt0 + [0, dim[1], 0]
    pt5 = pt1 + [0, dim[1], 0]
    pt6 = pt2 + [0, dim[1], 0]
    pt7 = pt3 + [0, dim[1], 0]
    return np.vstack([pt0, pt1, pt2, pt3, pt4, pt5, pt6, pt7])


def main():
    ray0 = np.array([0., 0., 0.])
    ray1 = np.array([-0.2, -0.2, -1.])

    tri0 = np.array([1., 0., -2.])
    Tri1 = np.array([
        [-1., -1., -2.],
        [-1., -1., -0.5],
        [-1., -1., 1.],
    ])
    Tri2 = np.array([
        [-1., 1., -2.],
        [-1., 1., -2.],
        [-1., 1., -2.],
    ])

    Ray0 = np.repeat([ray0], 3, axis=0)
    Ray1 = np.repeat([ray1], 3, axis=0)
    Tri0 = np.repeat([tri0], 3, axis=0)

    ax = plt.figure().gca(projection='3d')

    # plot ray
    x, y, z = zip(*np.vstack(([ray0], [ray1])))
    ax.plot(x, y, z, color='b', marker='o', markersize=10)
    x, y, z = zip(*[ray0])
    ax.plot(x, y, z, color='r', marker='o', markersize=10)

    # compute intersection of ray and aabb
    lb, rt = pascal3d.utils.triangle_to_aabb(Tri0, Tri1, Tri2)
    for c, m, lb_i, rt_i in zip('ymc', '^vx', lb, rt):
        flag, intersect = pascal3d.utils.intersect3d_ray_aabb(
            ray0, ray1, lb_i, rt_i)
        assert flag

        # plot intersect point
        ax.plot(*zip(*[intersect]), color=c, marker=m, markersize=10)

        # plot 3d box
        octopoints = aabb_to_octopoints(lb_i, rt_i)
        for s, e in itertools.combinations(octopoints, 2):
            if np.sum((s - e) == 0) == 2:  # neighbor points
                ax.plot(*zip(s, e), color=c, marker=m, markersize=10)

    plt.show()


if __name__ == '__main__':
    main()
