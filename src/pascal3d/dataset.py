#!/usr/bin/env python

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


def get_homogenous_trans_matrix(azimuth, elevation, distance):
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

    return R


def transform_to_camera_frame(
        x3d,
        azimuth,
        elevation,
        distance,
        ):
    if distance == 0:
        return []

    R = get_homogenous_trans_matrix(azimuth, elevation, distance)

    # get points in camera frame
    x3d_ = np.hstack((x3d, np.ones((len(x3d), 1)))).T
    x3d_camframe = np.dot(R, x3d_).T

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
    if distance == 0:
        return []

    R = get_homogenous_trans_matrix(azimuth, elevation, distance)

    # perspective project matrix
    # however, we set the viewport to 3000, which makes the camera similar to
    # an affine-camera.
    # Exploring a real perspective camera can be a future work.
    M = viewport
    P = np.array([[M * focal, 0, 0],
                  [0, M * focal, 0],
                  [0, 0, -1]]).dot(R)

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

    def draw_annotation(self, i):
        data_id = self.data_ids[i]

        img = None

        for cls in self.class_names:
            ann_file = osp.join(
                self.dataset_dir,
                'Annotations/{}_pascal/{}.mat'.format(cls, data_id))
            if not osp.exists(ann_file):
                continue

            ann = Pascal3DAnnotation(ann_file)

            # img file is identical for one data_id
            if img is None:
                img_file = osp.join(
                    self.dataset_dir,
                    'Images/{}_pascal'.format(cls),
                    ann.img_filename)
                img = scipy.misc.imread(img_file)

            for obj in ann.objects:
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
        fig = plt.figure()

        ax = fig.gca(projection='3d')

        data_id = self.data_ids[i]
        for cls in self.class_names:
            ann_file = osp.join(
                self.dataset_dir,
                'Annotations/{}_pascal/{}.mat'.format(cls, data_id))
            if not osp.exists(ann_file):
                continue

            ann = Pascal3DAnnotation(ann_file)

            cad_file = osp.join(
                self.dataset_dir,
                'CAD/{}.mat'.format(cls))
            cad = scipy.io.loadmat(cad_file)[cls][0]

            for obj in ann.objects:
                cad_index = obj['cad_index']

                vertices_3d = cad[cad_index]['vertices']
                # faces = cad[cad_index]['faces']

                if camframe:
                    vertices_3d = transform_to_camera_frame(
                        vertices_3d,
                        obj['viewpoint']['azimuth'],
                        obj['viewpoint']['elevation'],
                        obj['viewpoint']['distance'],
                    )
                    ax.plot([0], [0], [0], marker='o', color='r')

                x, y, z = zip(*vertices_3d)
                ax.plot(x, y, z, color='b')
                plt.show()

    def show_cad_overlay(self, i):
        ax1 = plt.subplot(121)
        plt.axis('off')
        ax2 = plt.subplot(122)
        plt.axis('off')

        img = None

        data_id = self.data_ids[i]
        for cls in self.class_names:

            ann_file = osp.join(
                self.dataset_dir,
                'Annotations/{}_pascal/{}.mat'.format(cls, data_id))
            if not osp.exists(ann_file):
                continue

            ann = Pascal3DAnnotation(ann_file)

            # img file is identical for one data_id
            if img is None:
                img_file = osp.join(
                    self.dataset_dir,
                    'Images/{}_pascal'.format(cls),
                    ann.img_filename)
                img = scipy.misc.imread(img_file)
                ax1.imshow(img)
                ax2.imshow(img)

            cad_file = osp.join(
                self.dataset_dir,
                'CAD/{}.mat'.format(cls))
            cad = scipy.io.loadmat(cad_file)[cls][0]

            for obj in ann.objects:
                cad_index = obj['cad_index']

                vertices_3d = cad[cad_index]['vertices']
                faces = cad[cad_index]['faces']

                vertices_2d = project_vertices_3d_to_2d(
                    vertices_3d, **obj['viewpoint'])

                patches = []
                for face in faces:
                    points = [vertices_2d[i_vertex-1] for i_vertex in face]
                    poly = Polygon(points, True)
                    patches.append(poly)
                p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)
                ax2.add_collection(p)

        plt.show()
        plt.cla()

    def convert_mesh_to_pcd(self):
        for cls in self.class_names:
            cad_dir = osp.join(self.dataset_dir, 'CAD', cls)
            for off_file in os.listdir(cad_dir):
                off_file = osp.join(cad_dir, off_file)
                cad_id = osp.splitext(off_file)[0]
                ply_file = osp.join(cad_dir, cad_id + '.ply')
                pcd_file = osp.join(cad_dir, cad_id + '.pcd')
                # off file -> ply file
                cmd = 'meshlabserver -i {} -o {}'.format(off_file, ply_file)
                subprocess.call(shlex.split(cmd))
                # ply file -> pcd file
                cmd = 'pcl_mesh2pcd {} {} -no_vis_result'.format(
                    ply_file, pcd_file)
                subprocess.call(shlex.split(cmd))
