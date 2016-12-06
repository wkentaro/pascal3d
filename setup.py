#!/usr/bin/env python

from setuptools import find_packages
from setuptools import setup

from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np


__version__ = '0.1'

ext_modules = [
    Extension(
        'pascal3d.utils._geometry',
        ['pascal3d/utils/_geometry.pyx'],
        include_dirs=[np.get_include()]
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
    tests_require=['nose'],
    ext_modules=cythonize(ext_modules),
)
