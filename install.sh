#!/bin/bash

set -x

set +x
if [ ! -e $HOME/.anaconda2/bin/activate ]; then
  echo 'Please install Anaconda to ~/.anaconda2'
  exit 1
fi
unset PYTHONPATH
source $HOME/.anaconda2/bin/activate
conda --version
set -x

if [ ! -e $CONDA_PREFIX/envs/pascal3d ]; then
  conda create -q -y --name=pascal3d python=2.7
fi
set +x
source activate pascal3d
set -x

conda info -e

conda install -y opencv -c menpo

pip install cython
pip install numpy
pip install scikit-image

pip install -e .

set +x
