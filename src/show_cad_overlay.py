#!/usr/bin/env python

import pascal3d


def main():
    dataset = pascal3d.dataset.Pascal3DDataset('val')
    for i in xrange(len(dataset)):
        dataset.overlay_cad(i)


if __name__ == '__main__':
    main()
