#!/bin/bash

source ~/.bashrc

make clean

if [ "$1" = "debug" ];
then
  make debug=-Ddebug OPT=$2
  ./sobel_orig
else
  make OPT=$1
  for value in {1..12};
  do
    ./sobel_orig >> metrices.txt
  done
fi
