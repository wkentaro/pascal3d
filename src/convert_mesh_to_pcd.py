#!/usr/bin/env python

import pascal3d


def main():
    for data_type in ['train', 'val']:
        dataset = pascal3d.dataset.Pascal3DDataset(data_type)
        dataset.convert_mesh_to_pcd()


if __name__ == '__main__':
    main()
