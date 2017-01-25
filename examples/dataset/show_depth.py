#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import skimage.color

import pascal3d


def main():
    data_type = 'val'
    dataset = pascal3d.dataset.Pascal3DDataset(data_type)
    for i in xrange(len(dataset)):
        print('[{dtype}:{id}] showing depth'.format(dtype=data_type, id=i))
        data = dataset.get_data(i)
        img = data['img']
        lbl_cls = data['label_cls']
        label_viz = skimage.color.label2rgb(lbl_cls, bg_label=0)

        min_depth, max_depth = dataset.get_depth(i)
        if np.isnan(min_depth).sum() == min_depth.size:
            continue
        min_value = min_depth[~np.isnan(min_depth)].min()
        max_value = max_depth[~np.isnan(max_depth)].max()
        plt.subplot(221)
        plt.imshow(img)
        plt.subplot(222)
        plt.imshow(label_viz)
        plt.subplot(223)
        min_depth_viz = pascal3d.utils.colorize_depth(
            min_depth, min_value, max_value)
        plt.imshow(min_depth_viz)
        plt.subplot(224)
        max_depth_viz = pascal3d.utils.colorize_depth(
            max_depth, min_value, max_value)
        plt.imshow(max_depth_viz)
        plt.show()


if __name__ == '__main__':
    main()
