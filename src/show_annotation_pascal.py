#!/usr/bin/env python

import matplotlib.pyplot as plt

import pascal3d


def main():
    dataset = pascal3d.dataset.Pascal3DDataset('val')
    for i in xrange(len(dataset)):
        img_viz = dataset.draw_annotation(i)
        plt.imshow(img_viz)
        plt.show()


if __name__ == '__main__':
    main()
