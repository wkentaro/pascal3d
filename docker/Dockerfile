FROM ubuntu:trusty

RUN set -x && \
  sudo apt-get update -qq && \
  sudo apt-get upgrade -qq -y && \
  sudo apt-get install -qq -y \
    git \
    wget \
    python python-virtualenv \
    liblapack-dev \
    libopencv-dev python-opencv
