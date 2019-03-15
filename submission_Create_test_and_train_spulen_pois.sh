#!/bin/sh
#BSUB -q mpi
#BSUB -W 48:00
#BSUB -o out/Create_test_train_spulen_pois%J
#BSUB -n 24
#BSUB -x
#BSUB -R "intel=4"
#BSUB -R scratch2

$HOME/build/Python-3.6.7/python $HOME/Create_test_and_train_spulen_pois.py &> output/Create_test_train_spulen_pois.txt&
wait
