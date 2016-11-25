#!/usr/bin/env python

import math
import os
import os.path as osp

import chainer
import numpy as np
import scipy.misc
import scipy.io
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon


def project_3d(vertices, obj):
    viewpoint = obj['viewpoint']
    a = viewpoint['azimuth'][0][0][0][0] * math.pi / 180
    e = viewpoint['elevation'][0][0][0][0] * math.pi / 180
    d = viewpoint['distance'][0][0][0][0]
    f = viewpoint['focal'][0][0][0][0]
    theta = viewpoint['theta'][0][0][0][0] * math.pi / 180
    principal = np.array([viewpoint['px'][0][0][0][0],
                          viewpoint['py'][0][0][0][0]])
    viewport = viewpoint['viewport'][0][0][0][0]

    if d == 0:
        return []

    # camera center
    C = np.zeros((3, 1))
    C[0] = d * math.cos(e) * math.sin(a)
    C[1] = -d * math.cos(e) * math.cos(a)
    C[2] = d * math.sin(e)

    # rotate coordinate system by theta is equal to rotating the model by theta
    a = -a
    e = - (math.pi / 2 - e)

    # rotation matrix
    Rz = np.array([[math.cos(a), -math.sin(a), 0],
                   [math.sin(a), math.cos(a), 0],
                   [0, 0, 1]])  # rotation by a
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(e), -math.sin(e)],
                   [0, math.sin(e), math.cos(e)]])  # rotation by e
    R = np.dot(Rx, Rz)

    # perspective project matrix
    # however, we set the viewport to 3000, which makes the camera similar to
    # an affine-camera.
    # Exploring a real perspective camera can be a future work.
    M = viewport
    R_ = np.hstack((R, np.dot(-R, C)))
    P = np.array([[M * f, 0, 0],
                  [0, M * f, 0],
                  [0, 0, -1]]).dot(R_)

    # project
    vertices_ = np.hstack((vertices, np.ones((len(vertices), 1)))).T
    x = np.dot(P, vertices_)
    x[0, :] = x[0, :] / x[2, :]
    x[1, :] = x[1, :] / x[2, :]
    x = x[0:2, :]

    # rotation matrix 2D
    R2d = np.array([[math.cos(theta), -math.sin(theta)],
                    [math.sin(theta), math.cos(theta)]])
    x = np.dot(R2d, x).T

    # transform to image coordinate
    x[:, 1] *= -1
    x = x + np.repeat(principal[np.newaxis, :], len(x), axis=0)

    return x


def main():
    cls = 'car'

    dataset_dir = chainer.dataset.get_dataset_directory(
        'pascal3d/PASCAL3D+_release1.1')
    ann_dir = osp.join(dataset_dir, 'Annotations/%s_pascal' % cls)
    img_dir = osp.join(dataset_dir, 'Images/%s_pascal' % cls)

    cad_file = osp.join(dataset_dir, 'CAD/%s.mat' % cls)
    cad = scipy.io.loadmat(cad_file)[cls][0]

    for ann_fname in os.listdir(ann_dir):

        fig, ax = plt.subplots()

        ann_fname = osp.join(ann_dir, ann_fname)

        record = scipy.io.loadmat(ann_fname)['record']

        img = scipy.misc.imread(osp.join(img_dir, record['filename'][0][0][0]))
        ax.imshow(img)

        obj_indices = np.where((record['objects'][0][0]['class'] == cls)[0])[0]

        for obj_index in obj_indices:
            obj = record['objects'][0][0][0][obj_index]
            if obj['viewpoint']['distance'][0][0][0][0] == 0:
                print('No continuous viewpoint')
                continue
            cad_index = obj['cad_index'][0][0] - 1
            vertices = cad[cad_index]['vertices']
            faces = cad[cad_index]['faces']

            x2d = project_3d(vertices, obj)

            patches = []
            for face in faces:
                points = [x2d[i_vertex-1] for i_vertex in face]
                poly = Polygon(points, True)
                patches.append(poly)
            p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)
            ax.add_collection(p)

        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    main()
