#!/usr/bin/env python

import argparse

import pascal3d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--dry-run', action='store_true')
    parser.add_argument('-r', '--replace', action='store_true')
    args = parser.parse_args()

    dry_run = args.dry_run
    replace = args.replace

    for data_type in ['train', 'val']:
        dataset = pascal3d.dataset.Pascal3DDataset(data_type)
        dataset.convert_mesh_to_pcd(dry_run, replace)


if __name__ == '__main__':
    main()
