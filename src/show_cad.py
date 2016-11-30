#!/usr/bin/env python

import argparse

import pascal3d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camframe', action='store_true')
    args = parser.parse_args()

    camframe = args.camframe

    dataset = pascal3d.dataset.Pascal3DDataset('val')
    for i in xrange(len(dataset)):
        dataset.show_cad(i, camframe=camframe)


if __name__ == '__main__':
    main()
