#!/usr/bin/env python

import os
import os.path as osp

import cv2
import chainer
import matplotlib.pyplot as plt
import scipy.io
import scipy.misc


def main():
    cls = 'car'

    dataset_dir = chainer.dataset.get_dataset_directory(
        'pascal3d/PASCAL3D+_release1.1')
    path_image = osp.join(dataset_dir, 'Images/%s_pascal' % cls)
    path_ann = osp.join(dataset_dir, 'Annotations/%s_pascal' % cls)

    img_fnames = os.listdir(path_image)
    for img_fname in img_fnames:
        data_id = osp.splitext(osp.basename(img_fname))[0]

        img_fname = osp.join(path_image, img_fname)
        img = scipy.misc.imread(img_fname)

        ann_fname = osp.join(path_ann, '%s.mat' % data_id)
        ann = scipy.io.loadmat(ann_fname)
        record = ann['record']
        title = ''

        for obj in record['objects'][0][0][0]:
            if obj['class'] != cls:
                continue

            bbox = obj['bbox'][0]
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))

            anchors = obj['anchors']
            if len(anchors):
                names = anchors.dtype.names
                for name in names:
                    anchor = anchors[name][0][0]
                    if anchor['status'] != 1:
                        continue
                    x, y = anchor['location'][0][0][0]
                    x, y = int(x), int(y)
                    cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
            viewpoint = obj['viewpoint']
            if obj['viewpoint']['distance'] == 0:
                title += 'ac=%.2f, ec=%.2f, '\
                    % (viewpoint['azimuth_coarse'],
                       viewpoint['elevation_coarse'])
            else:
                title += 'a=%.2f, e=%.2f, d=%.2f, '\
                    % (viewpoint['azimuth'],
                       viewpoint['elevation'],
                       viewpoint['distance'])

        plt.imshow(img)
        plt.title(title)
        plt.show()


if __name__ == '__main__':
    main()
