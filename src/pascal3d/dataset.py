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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D  # NOQA
import numpy as np
import PIL.Image
import scipy.io
import scipy.misc
import shlex
import skimage.color
import subprocess

import utils


class Pascal3DAnnotation(object):

    def __init__(self, ann_file):
        ann_data = scipy.io.loadmat(ann_file)

        self.img_filename = ann_data['record']['filename'][0][0][0]
        self.segmented = ann_data['record']['segmented'][0][0][0]

        self.objects = []
        for obj in ann_data['record']['objects'][0][0][0]:
            if not obj['viewpoint']:
                continue
            elif 'distance' not in obj['viewpoint'].dtype.names:
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

    voc2012_class_names = [
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'potted plant',
        'sheep',
        'sofa',
        'train',
        'tv/monitor',
    ]

    class_names = [
        'background',
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
        print('Generating index for annotations...')
        data_ids = []
        for cls in self.class_names[1:]:
            cls_ann_dir = osp.join(
                self.dataset_dir, 'Annotations/{}_pascal'.format(cls))
            for ann_file in os.listdir(cls_ann_dir):
                ann = Pascal3DAnnotation(osp.join(cls_ann_dir, ann_file))
                if not ann.segmented:
                    continue
                data_id = osp.splitext(ann_file)[0]
                data_ids.append(data_id)
        print('Done.')
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

    def get_data(self, i):
        data_id = self.data_ids[i]

        data = {
            'img': None,
            'objects': [],
            'class_cads': {},
            'label_cls': None,
        }

        for class_name in self.class_names[1:]:
            ann_file = osp.join(
                self.dataset_dir,
                'Annotations/{}_pascal/{}.mat'.format(class_name, data_id))
            if not osp.exists(ann_file):
                continue
            ann = Pascal3DAnnotation(ann_file)

            if data['label_cls'] is None:
                label_cls_file = osp.join(
                    self.dataset_dir,
                    'PASCAL/VOCdevkit/VOC2012/SegmentationClass/{}.png'
                    .format(data_id))
                label_cls = PIL.Image.open(label_cls_file)
                label_cls = np.array(label_cls)
                label_cls[label_cls == 255] = 0  # set boundary as background
                # convert label from voc2012 to pascal3D
                for voc2012_id, cls in enumerate(self.voc2012_class_names):
                    cls = cls.replace('/', '')
                    if cls in self.class_names:
                        pascal3d_id = self.class_names.index(cls)
                        label_cls[label_cls == voc2012_id] = pascal3d_id
                    else:
                        # set background class id
                        label_cls[label_cls == voc2012_id] = 0
                data['label_cls'] = label_cls

            if class_name not in data['class_cads']:
                cad_file = osp.join(
                    self.dataset_dir,
                    'CAD/{}.mat'.format(class_name))
                cad = scipy.io.loadmat(cad_file)[class_name][0]
                data['class_cads'][class_name] = cad

            if data['img'] is None:
                img_file = osp.join(
                    self.dataset_dir,
                    'Images/{}_pascal'.format(class_name),
                    ann.img_filename)
                data['img'] = scipy.misc.imread(img_file)

            for obj in ann.objects:
                obj['cad_basename'] = osp.join(
                    self.dataset_dir,
                    'CAD/{}/{:02}'.format(class_name, obj['cad_index'] + 1))
                data['objects'].append((class_name, obj))

        return data

    def show_annotation(self, i):
        data = self.get_data(i)
        img = data['img']
        objects = data['objects']
        label_cls = data['label_cls']

        ax1 = plt.subplot(121)
        plt.axis('off')

        ax2 = plt.subplot(122)
        plt.axis('off')
        label_viz = skimage.color.label2rgb(label_cls, bg_label=0)
        ax2.imshow(label_viz)

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
        ax1.imshow(img)

        plt.tight_layout()
        plt.show()

    def show_cad(self, i, camframe=False):
        if camframe:
            return self.show_cad_camframe(i)

        data = self.get_data(i)
        img = data['img']
        objects = data['objects']
        class_cads = data['class_cads']

        for cls, obj in objects:
            # show image
            ax1 = plt.subplot(1, 2, 1)
            plt.axis('off')
            ax1.imshow(img)

            ax2 = plt.subplot(1, 2, 2, projection='3d')

            cad_index = obj['cad_index']
            cad = class_cads[cls]

            # show camera model
            height, width = img.shape[:2]
            x = utils.get_camera_polygon(
                height=height,
                width=width,
                theta=obj['viewpoint']['theta'],
                focal=obj['viewpoint']['focal'],
                principal=obj['viewpoint']['principal'],
                viewport=obj['viewpoint']['viewport'],
            )
            R = utils.get_transformation_matrix(
                obj['viewpoint']['azimuth'],
                obj['viewpoint']['elevation'],
                obj['viewpoint']['distance'],
            )
            x = np.hstack((x, np.ones((len(x), 1), dtype=np.float64)))
            x = np.dot(np.linalg.inv(R)[:3, :4], x.T).T
            x0, x1, x2, x3, x4 = x
            verts = [
                [x0, x1, x2],
                [x0, x2, x3],
                [x0, x3, x4],
                [x0, x4, x1],
                [x1, x2, x3, x4],
            ]
            ax2.add_collection3d(
                Poly3DCollection([verts[0]], facecolors='r', linewidths=1))
            ax2.add_collection3d(
                Poly3DCollection(verts[1:], facecolors='w',
                                 linewidths=1, alpha=0.5))
            x, y, z = zip(*x)
            ax2.plot(x, y, z)  # to show the camera model in the range

            max_x = max(x)
            max_y = max(y)
            max_z = max(z)
            min_x = min(x)
            min_y = min(y)
            min_z = min(z)

            # display the cad model
            vertices_3d = cad[cad_index]['vertices']
            x, y, z = zip(*vertices_3d)
            ax2.plot(x, y, z, color='b')

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

            plt.tight_layout()
            plt.show()

    def show_cad_camframe(self, i):
        data = self.get_data(i)
        img = data['img']
        objects = data['objects']
        class_cads = data['class_cads']

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

        plt.tight_layout()
        plt.show()

    def show_cad_overlay(self, i):
        data = self.get_data(i)
        img = data['img']
        objects = data['objects']
        class_cads = data['class_cads']

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

        plt.tight_layout()
        plt.show()

    def show_pcd_overlay(self, i):
        data = self.get_data(i)
        img = data['img']
        objects = data['objects']

        ax1 = plt.subplot(121)
        plt.axis('off')
        ax1.imshow(img)

        ax2 = plt.subplot(122)
        plt.axis('off')

        n_classes = len(self.class_names)
        colormap = plt.cm.Spectral(
            np.linspace(0, 1, n_classes-1))[:, :3]   # w/o background color
        colormap = np.vstack(([0, 0, 0], colormap))  # w/ background color
        for cls, obj in objects:
            cls_id = self.class_names.index(cls)
            pcd_file = obj['cad_basename'] + '.pcd'
            points_3d = utils.load_pcd(pcd_file)
            points_2d = utils.project_vertices_3d_to_2d(
                points_3d, **obj['viewpoint'])
            img = img.astype(np.float64)
            height, width = img.shape[:2]
            for x, y in points_2d:
                if x > width or x < 0 or y > height or y < 0:
                    continue
                img[y, x] = colormap[cls_id] * 255
            img = img.astype(np.uint8)

        ax2.imshow(img)
        plt.tight_layout()
        plt.show()

    def show_depth(self, i):
        data = self.get_data(i)
        img = data['img']
        objects = data['objects']

        ax1 = plt.subplot(131)
        plt.axis('off')
        ax1.imshow(img)

        height, width = img.shape[:2]
        depth = np.zeros((height, width), dtype=np.float64)
        depth[...] = np.nan
        max_depth = depth.copy()
        for cls, obj in objects:
            cls_id = self.class_names.index(cls)
            pcd_file = obj['cad_basename'] + '.pcd'
            points_3d = utils.load_pcd(pcd_file)
            points_3d_camframe = utils.transform_to_camera_frame(
                points_3d,
                obj['viewpoint']['azimuth'],
                obj['viewpoint']['elevation'],
                obj['viewpoint']['distance'],
            )
            points_2d = utils.project_vertices_3d_to_2d(
                points_3d, **obj['viewpoint'])
            for (x, y), (_, _, z) in zip(points_2d, points_3d_camframe):
                x, y = int(x), int(y)
                if x >= width or x < 0 or y >= height or y < 0:
                    continue
                if np.isnan(depth[y, x]):
                    assert np.isnan(max_depth[y, x])
                    depth[y, x] = max_depth[y, x] = abs(z)
                else:
                    depth[y, x] = min(depth[y, x], abs(z))
                    max_depth[y, x] = max(max_depth[y, x], abs(z))

        obj_depth = max_depth - depth

        ax2 = plt.subplot(132)
        plt.axis('off')
        ax2.imshow(depth)

        ax2 = plt.subplot(133)
        plt.axis('off')
        ax2.imshow(obj_depth)

        plt.tight_layout()
        plt.show()

    def convert_mesh_to_pcd(self, dry_run=False):
        # scrape off files
        off_files = []
        for cls in self.class_names[1:]:
            cad_dir = osp.join(self.dataset_dir, 'CAD', cls)
            for off_file in os.listdir(cad_dir):
                off_file = osp.join(cad_dir, off_file)
                if osp.splitext(off_file)[-1] == '.off':
                    off_files.append(off_file)
        # using pcl_mesh2pcd
        for off_file in off_files:
            cad_dir = osp.dirname(off_file)
            cad_id = osp.splitext(osp.basename(off_file))[0]
            obj_file = osp.join(cad_dir, cad_id + '.obj')
            pcd_file = osp.join(cad_dir, cad_id + '.pcd')
            if osp.exists(pcd_file):
                if not dry_run:
                    print('PCD file exists, so skipping: {}'
                          .format(pcd_file))
                continue
            # off file -> obj file
            cmd = 'meshlabserver -i {} -o {}'.format(off_file, obj_file)
            if dry_run:
                print(cmd)
            else:
                subprocess.call(shlex.split(cmd))
            # obj file -> pcd file
            cmd = 'pcl_mesh2pcd {} {} -no_vis_result'.format(
                obj_file, pcd_file)
            if dry_run:
                print(cmd)
            else:
                subprocess.call(shlex.split(cmd))
        # using pcl_mesh_sampling
        # FIXME: sometimes pcl_mesh2pcd segfaults
        for off_file in off_files:
            cad_dir = osp.dirname(off_file)
            cad_id = osp.splitext(osp.basename(off_file))[0]
            obj_file = osp.join(cad_dir, cad_id + '.obj')
            pcd_file = osp.join(cad_dir, cad_id + '.pcd')
            if osp.exists(pcd_file):
                if not dry_run:
                    print('PCD file exists, so skipping: {}'
                          .format(pcd_file))
                continue
            # ply file -> pcd file
            cmd = 'pcl_mesh_sampling {} {} -no_vis_result'.format(
                obj_file, pcd_file)
            if dry_run:
                print(cmd)
            else:
                subprocess.call(shlex.split(cmd))
