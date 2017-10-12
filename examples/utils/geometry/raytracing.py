#!/usr/bin/env python

import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np

import pascal3d


def main():
    dataset = pascal3d.dataset.Pascal3DDataset('val')

    for i in xrange(len(dataset)):
        depth, backdepth = dataset.get_depth(i)

        data = dataset.get_data(i)
        img = data['img']

        min_max_values = [(0, 30), (np.nanmin(depth), np.nanmax(backdepth))]
        for i, (min_imvalue, max_imvalue) in enumerate(min_max_values):
            depth_viz = pascal3d.utils.colorize_depth(
                depth, min_imvalue, max_imvalue)
            backdepth_viz = pascal3d.utils.colorize_depth(
                backdepth, min_imvalue, max_imvalue)

            plt.subplot(2, 3, i * 3 + 1)
            plt.imshow(img)
            plt.subplot(2, 3, i * 3 + 2)
            plt.imshow(depth_viz)
            plt.subplot(2, 3, i * 3 + 3)
            plt.imshow(backdepth_viz)
        plt.show()


if __name__ == '__main__':
    main()
