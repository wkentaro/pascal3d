FROM ubuntu:trusty

RUN set -x && \
  sudo apt-get update -qq && \
  sudo apt-get upgrade -qq -y && \
  sudo apt-get install -qq -y \
    git \
    wget \
    python-dev python-setuptools python-tk \
    liblapack-dev \
    libopencv-dev python-opencv

RUN set -x && \
  sudo easy_install -U pip && \
  sudo pip install -U pip setuptools
