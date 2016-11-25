#!/usr/bin/env python

import os.path as osp

import chainer
import cv2
import matplotlib.pyplot as plt
import scipy.io
import scipy.misc


def main():
    data_id = 'car_pascal/2008_005158'

    dataset_dir = chainer.dataset.get_dataset_directory(
        'pascal3d/PASCAL3D+_release1.1')

    img_file = osp.join(dataset_dir, 'Images/%s.jpg' % data_id)
    ann_file = osp.join(dataset_dir, 'Annotations/%s.mat' % data_id)

    ann = scipy.io.loadmat(ann_file)

    img = scipy.misc.imread(img_file)

    anchors = ann['record'][0][0][8]['anchors'][0][0]

    for name in anchors.dtype.names:
        anchor = anchors[name][0][0][0][0][0]
        if not len(anchor):
            continue
        anchor = anchor[0]
        anchor = tuple(anchor.astype(int).tolist())
        cv2.circle(img, anchor, 3, (255, 0, 0), -1)
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    main()
