#!/usr/bin/env python

import time

import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import PIL.ImageDraw

import pascal3d


def main():
    dataset = pascal3d.dataset.Pascal3DDataset('val')

    for i in xrange(len(dataset)):
        for depth, backdepth in dataset.get_depth(i):
            data = dataset.get_data(i)
            img = data['img']
            gray = img.mean(axis=-1)
            gray = np.repeat(np.atleast_3d(gray), 3, axis=-1)

            min_imvalue = 0
            max_imvalue = 30
            # min_imvalue = depth[~np.isnan(depth)].min()
            # max_imvalue = backdepth[~np.isnan(backdepth)].max()

            # static scaling
            depth[np.isnan(depth)] = 0.0
            depth = (depth - min_imvalue) / (max_imvalue - min_imvalue)
            depth[depth > 1.0] = 1.0
            depth[depth < 0.0] = 0.0
            depth_viz = matplotlib.cm.jet(depth)[:, :, :3]
            depth_viz = (depth_viz * 255).astype(np.uint8)
            depth_viz[depth == 0] = [0, 0, 0]
            plt.subplot(132)
            plt.imshow(depth_viz)

            backdepth[np.isnan(backdepth)] = 0.0
            backdepth = (backdepth - min_imvalue) / (max_imvalue - min_imvalue)
            backdepth[backdepth > 1.0] = 1.0
            backdepth[backdepth < 0.0] = 0.0
            backdepth_viz = matplotlib.cm.jet(backdepth)[:, :, :3]
            backdepth_viz = (backdepth_viz * 255).astype(np.uint8)
            backdepth_viz[backdepth == 0] = [0, 0, 0]

            plt.subplot(131)
            img_viz = gray * 0.7 + depth_viz * 0.3
            img_viz = img_viz.astype(np.uint8)
            plt.imshow(img_viz)
            plt.subplot(132)
            plt.imshow(depth_viz)
            plt.subplot(133)
            plt.imshow(backdepth_viz)
            plt.show()


if __name__ == '__main__':
    main()
