#!/usr/bin/env python

import pascal3d


def main():
    data_type = 'val'
    dataset = pascal3d.dataset.Pascal3DDataset(data_type)
    for i in xrange(len(dataset)):
        print('[{dtype}:{id}] showing pcd overlay'
              .format(dtype=data_type, id=i))
        dataset.show_pcd_overlay(i)


if __name__ == '__main__':
    main()
