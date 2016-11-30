#!/usr/bin/env python

import os.path as osp

import chainer
import matplotlib.pyplot as plt
# required for 3D plot
from mpl_toolkits.mplot3d import Axes3D  # NOQA

import pascal3d


def main():
    dataset = pascal3d.dataset.Pascal3DDataset('val')
    for i in xrange(len(dataset)):
        dataset.show_cad(i)


if __name__ == '__main__':
    main()
