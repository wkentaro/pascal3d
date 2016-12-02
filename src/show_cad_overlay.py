#!/usr/bin/env python

import pascal3d


def main():
    data_type = 'val'
    dataset = pascal3d.dataset.Pascal3DDataset(data_type)
    for i in xrange(len(dataset)):
        if data_type == 'val' and i == 0:
            print('Skipping invalid data: id={}, data_type={}'
                  .format(i, data_type))
            continue
        print('Showing cad overlay: id={}, data_type={}'
              .format(i, data_type))
        dataset.show_cad_overlay(i)


if __name__ == '__main__':
    main()
