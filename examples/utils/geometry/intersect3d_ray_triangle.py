#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

import pascal3d


def main():
    ray0 = np.array([[0., 0., 0.]])
    ray1 = np.array([[-0.2, 0.2, -1.]])

    tri0 = np.array([[1., 0., -2.]])
    tri1 = np.array([[-1., -1., -2.]])
    tri2 = np.array([[-1., 1., -2.]])

    ax = plt.figure().gca(projection='3d')

    flags, intersects = pascal3d.utils.intersect3d_ray_triangle(
        ray0, ray1, tri0, tri1, tri2)
    assert flags[0] == 1
    intersect = intersects[0]

    x, y, z = zip(*np.vstack((ray0, ray1, intersect)))
    ax.plot(x, y, z, color='k', marker='o')
    x, y, z = zip(*ray0)
    ax.plot(x, y, z, color='r', marker='o', markersize=5)
    x, y, z = zip(*intersect[np.newaxis, :])
    ax.plot(x, y, z, color='g', marker='o', markersize=5)
    x, y, z = zip(*np.vstack((tri0, tri1, tri2, tri0)))
    ax.plot(x, y, z, color='b', marker='o')

    plt.show()


if __name__ == '__main__':
    main()
