#!/usr/bin/env python

import os.path as osp

import chainer
import fcn


def main():
    dataset_dir = osp.expanduser('~/data/datasets/Pascal3D')
    path = osp.join(dataset_dir, 'PASCAL3D+_release1.1.zip')
    fcn.data.cached_download(
        url='https://drive.google.com/uc?id=0B9P1L--7Wd2vcndfcElOT1d2a0E',
        path=path,
        md5='28c2d7dac539cf5f2592cbc06299f895',
    )
    fcn.data.extract_file(path, dataset_dir)


if __name__ == '__main__':
    main()
