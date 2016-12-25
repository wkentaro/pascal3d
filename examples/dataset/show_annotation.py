#!/usr/bin/env python

import pascal3d


def main():
    data_type = 'val'
    dataset = pascal3d.dataset.Pascal3DDataset(data_type)
    for i in xrange(len(dataset)):
        print('[{dtype}:{id}] showing annotation'
              .format(dtype=data_type, id=i))
        dataset.show_annotation(i)


if __name__ == '__main__':
    main()
