#!/usr/bin/env python

import os.path as osp

import chainer
import matplotlib.pyplot as plt
# required for 3D plot
from mpl_toolkits.mplot3d import Axes3D  # NOQA


def load_off(filename):
    with open(filename, 'r') as f:
        if 'OFF' != f.readline().strip():
            raise ValueError('Not a valid OFF header')
        n_verts, n_faces, _ = map(int, f.readline().strip().split(' '))
        vertices = []
        for i_vert in range(n_verts):
            vertex = map(float, f.readline().strip().split(' '))
            vertices.append(vertex)
        faces = []
        for i_face in range(n_faces):
            face = map(int, f.readline().strip().split(' ')[1:])
            faces.append(face)
        return vertices, faces


def main():
    dataset_dir = chainer.dataset.get_dataset_directory(
        'pascal3d/PASCAL3D+_release1.1')
    off_fname = osp.join(dataset_dir, 'Anchor/aeroplane/01.off')
    vertices, surfaces = load_off(off_fname)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x, y, z = zip(*vertices)
    ax.plot(x, y, z)
    plt.show()

if __name__ == '__main__':
    main()
