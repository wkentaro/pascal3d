#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import os
import os.path as osp

import chainer
import cv2
import math
import matplotlib
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # NOQA
import numpy as np
import scipy.io
import scipy.misc
import shlex
import subprocess

import utils


class Pascal3DAnnotation(object):

    def __init__(self, ann_file):
        ann_data = scipy.io.loadmat(ann_file)

        self.img_filename = ann_data['record']['filename'][0][0][0]

        self.objects = []
        for obj in ann_data['record']['objects'][0][0][0]:
            if not obj['viewpoint']:
                continue
            elif obj['viewpoint']['distance'][0][0][0][0] == 0:
                continue

            cad_index = obj['cad_index'][0][0] - 1
            bbox = obj['bbox'][0]
            anchors = obj['anchors']

            viewpoint = obj['viewpoint']
            azimuth = viewpoint['azimuth'][0][0][0][0] * math.pi / 180
            elevation = viewpoint['elevation'][0][0][0][0] * math.pi / 180
            distance = viewpoint['distance'][0][0][0][0]
            focal = viewpoint['focal'][0][0][0][0]
            theta = viewpoint['theta'][0][0][0][0] * math.pi / 180
            principal = np.array([viewpoint['px'][0][0][0][0],
                                  viewpoint['py'][0][0][0][0]])
            viewport = viewpoint['viewport'][0][0][0][0]

            self.objects.append({
                'cad_index': cad_index,
                'bbox': bbox,
                'anchors': anchors,
                'viewpoint': {
                    'azimuth': azimuth,
                    'elevation': elevation,
                    'distance': distance,
                    'focal': focal,
                    'theta': theta,
                    'principal': principal,
                    'viewport': viewport,
                },
            })


class Pascal3DDataset(object):

    class_names = [
        'aeroplane',
        'bicycle',
        'boat',
        'bottle',
        'bus',
        'car',
        'chair',
        'diningtable',
        'motorbike',
        'sofa',
        'train',
        'tvmonitor',
    ]

    def __init__(self, data_type):
        assert data_type in ('train', 'val')
        self.dataset_dir = chainer.dataset.get_dataset_directory(
            'pascal3d/PASCAL3D+_release1.1')
        # get all data ids
        data_ids = []
        for cls in self.class_names:
            cls_ann_dir = osp.join(
                self.dataset_dir, 'Annotations/{}_pascal'.format(cls))
            for ann_file in os.listdir(cls_ann_dir):
                data_id = osp.splitext(ann_file)[0]
                data_ids.append(data_id)
        data_ids = list(set(data_ids))
        # split data to train and val
        val_size_ratio = 0.25
        if data_type == 'train':
            data_size = int(len(data_ids) * (1 - val_size_ratio))
        else:
            data_size = int(len(data_ids) * val_size_ratio)
        np.random.seed(1234)
        p = np.random.randint(0, len(data_ids), data_size)
        self.data_ids = np.array(data_ids)[p]

    def __len__(self):
        return len(self.data_ids)

    def _get_data(self, i):
        data_id = self.data_ids[i]

        img = None
        objects = []
        class_cads = {}
        for cls in self.class_names:
            ann_file = osp.join(
                self.dataset_dir,
                'Annotations/{}_pascal/{}.mat'.format(cls, data_id))
            if not osp.exists(ann_file):
                continue
            ann = Pascal3DAnnotation(ann_file)

            if cls not in class_cads:
                cad_file = osp.join(
                    self.dataset_dir,
                    'CAD/{}.mat'.format(cls))
                cad = scipy.io.loadmat(cad_file)[cls][0]
                class_cads[cls] = cad

            if img is None:
                img_file = osp.join(
                    self.dataset_dir,
                    'Images/{}_pascal'.format(cls),
                    ann.img_filename)
                img = scipy.misc.imread(img_file)

            for obj in ann.objects:
                objects.append((cls, obj))

        return img, objects, class_cads

    def draw_annotation(self, i):
        img, objects, _ = self._get_data(i)

        for cls, obj in objects:
            x1, y1, x2, y2 = obj['bbox']
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))

            if not obj['anchors']:
                continue
            anchors = obj['anchors'][0][0]
            for name in anchors.dtype.names:
                anchor = anchors[name]
                if anchor['status'] != 1:
                    continue
                x, y = anchor['location'][0][0][0]
                cv2.circle(img, (int(x), int(y)), 5, (255, 0, 0), -1)

        return img

    def show_cad(self, i, camframe=False):
        if camframe:
            return self.show_cad_camframe(i)

        img, objects, class_cads = self._get_data(i)

        for cls, obj in objects:
            cad_index = obj['cad_index']
            cad = class_cads[cls]

            vertices_3d = cad[cad_index]['vertices']

            ax = plt.figure().gca(projection='3d')
            x, y, z = zip(*vertices_3d)
            ax.plot(x, y, z, color='b')
            plt.show()

    def show_cad_camframe(self, i):
        img, objects, class_cads = self._get_data(i)

        ax1 = plt.subplot(1, 2, 1)
        ax1.imshow(img)
        plt.axis('off')

        ax2 = plt.subplot(1, 2, 2, projection='3d')
        ax2.plot([0], [0], [0], marker='o')

        max_x = min_x = 0
        max_y = min_y = 0
        max_z = min_z = 0
        for cls, obj in objects:
            cad_index = obj['cad_index']
            cad = class_cads[cls]

            vertices_3d = cad[cad_index]['vertices']

            vertices_3d_camframe = utils.transform_to_camera_frame(
                vertices_3d,
                obj['viewpoint']['azimuth'],
                obj['viewpoint']['elevation'],
                obj['viewpoint']['distance'],
            )

            # XXX: Not sure this is correct...
            delta = (obj['viewpoint']['principal'] /
                     obj['viewpoint']['viewport'])
            vertices_3d_camframe[:, 0] += delta[0] * 10
            vertices_3d_camframe[:, 1] -= delta[1] * 10

            x, y, z = zip(*vertices_3d_camframe)
            ax2.plot(x, y, z)

            max_x = max(max_x, max(x))
            max_y = max(max_y, max(y))
            max_z = max(max_z, max(z))
            min_x = min(min_x, min(x))
            min_y = min(min_y, min(y))
            min_z = min(min_z, min(z))

        # align bounding box
        max_range = max(max_x - min_x, max_y - min_y, max_z - min_z) * 0.5
        mid_x = (max_x + min_x) * 0.5
        mid_y = (max_y + min_y) * 0.5
        mid_z = (max_z + min_z) * 0.5
        ax2.set_xlim(mid_x - max_range, mid_x + max_range)
        ax2.set_ylim(mid_y - max_range, mid_y + max_range)
        ax2.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.show()

    def show_cad_overlay(self, i):
        img, objects, class_cads = self._get_data(i)

        ax1 = plt.subplot(121)
        plt.axis('off')
        ax1.imshow(img)

        ax2 = plt.subplot(122)
        plt.axis('off')
        ax2.imshow(img)

        for cls, obj in objects:
            cad_index = obj['cad_index']
            cad = class_cads[cls][cad_index]

            vertices_3d = cad['vertices']
            faces = cad['faces']

            vertices_2d = utils.project_vertices_3d_to_2d(
                vertices_3d, **obj['viewpoint'])

            patches = []
            for face in faces:
                points = [vertices_2d[i_vertex-1] for i_vertex in face]
                poly = Polygon(points, True)
                patches.append(poly)
            p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)
            ax2.add_collection(p)
        plt.show()
