#!/usr/bin/env python

import os
import os.path as osp

import numpy as np
import skimage.color
import skimage.io

import pascal3d


def generate_depth_dataset(data_type, out):
    dataset = pascal3d.dataset.Pascal3DDataset(data_type)
    for i in xrange(len(dataset)):
        out_sub = osp.join(out, '%06d' % i)
        if not osp.exists(out_sub):
            os.makedirs(out_sub)

        data = dataset.get_data(i)
        img = data['img']
        lbl_cls = data['label_cls']
        skimage.io.imsave(osp.join(out_sub, 'image.jpg'), img)

        # visualize label
        label_viz = skimage.color.label2rgb(lbl_cls, bg_label=0)
        skimage.io.imsave(osp.join(out_sub, 'label_viz.jpg'), label_viz)

        min_depth, max_depth = dataset.get_depth(i)
        np.save(osp.join(out_sub, 'min_depth.npy'), min_depth)
        np.save(osp.join(out_sub, 'max_depth.npy'), max_depth)

        # visualize depth
        min_value = np.nanmin(min_depth)
        max_value = np.nanmax(max_depth)
        min_depth_viz = pascal3d.utils.colorize_depth(
            min_depth, min_value, max_value)
        max_depth_viz = pascal3d.utils.colorize_depth(
            max_depth, min_value, max_value)
        skimage.io.imsave(
            osp.join(out_sub, 'min_depth_viz.jpg'), min_depth_viz)
        skimage.io.imsave(
            osp.join(out_sub, 'max_depth_viz.jpg'), max_depth_viz)

        print('wrote result to: %s' % out_sub)


def main():
    for data_type in ['train', 'val']:
        out = 'out_%s' % data_type
        generate_depth_dataset(data_type, out)


if __name__ == '__main__':
    main()
