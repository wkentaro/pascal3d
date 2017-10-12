#!/usr/bin/env python

import distutils.extension
from setuptools import find_packages
from setuptools import setup

import numpy as np
from skimage._build import cython


__version__ = '0.1'

cython(['_geometry.pyx'], working_path='pascal3d/utils')
ext_modules = [
    distutils.extension.Extension(
        'pascal3d.utils._geometry',
        sources=['pascal3d/utils/_geometry.c'],
        include_dirs=[np.get_include()],
    ),
]

setup(
    name='pascal3d',
    version=__version__,
    packages=find_packages(),
    author='Kentaro Wada',
    author_email='www.kentaro.wada@gmail.com',
    license='MIT',
    install_requires=open('requirements.txt').readlines(),
    tests_require=['pytest'],
    ext_modules=ext_modules,
)
