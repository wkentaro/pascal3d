#!/usr/bin/env python

import subprocess
import sys

import distutils.extension
from setuptools import find_packages
from setuptools import setup

import numpy as np
from skimage._build import cython


__version__ = '0.1.2'


if sys.argv[-1] == 'release':
    commands = [
        'python setup.py sdist upload',
        'git tag v{0}'.format(__version__),
        'git push origin master --tag',
    ]
    for cmd in commands:
        subprocess.call(cmd, shell=True)
    sys.exit(0)


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
