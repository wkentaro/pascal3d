#!/usr/bin/env python

import time

import matplotlib.pyplot as plt
import numpy as np

import pascal3d


def main():
    ray0 = np.array([[0., 0., 0.]])
    ray1 = np.array([[-0.2, -0.2, -1.]])

    tri0 = np.array([[1., 0., -2.]])
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

    Ray0 = np.repeat(ray0, 3, axis=0)
    Ray1 = np.repeat(ray1, 3, axis=0)
    Tri0 = np.repeat(tri0, 3, axis=0)

    ax = plt.figure().gca(projection='3d')

    t_start = time.time()
    flags, intersects = pascal3d.utils.intersect3d_ray_triangle(
        Ray0, Ray1, Tri0, Tri1, Tri2)
    print('elapsed_time: {:.3} [s]'.format(time.time() - t_start))
    intersects = intersects[flags == 1]

    x, y, z = zip(*np.vstack((ray0, ray1, intersects)))
    ax.plot(x, y, z, color='b', marker='o')
    x, y, z = zip(*ray0)
    ax.plot(x, y, z, color='r', marker='o', markersize=10)
    x, y, z = zip(*intersects)
    ax.plot(x, y, z, color='g', marker='o', markersize=10)

    for c, tri0, tri1, tri2 in zip('ymc', Tri0, Tri1, Tri2):
        x, y, z = zip(*np.vstack((tri0, tri1, tri2, tri0)))
        ax.plot(x, y, z, color=c, marker='o')

    plt.show()


if __name__ == '__main__':
    main()
