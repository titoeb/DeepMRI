#!/bin/sh
#BSUB -q mpi
#BSUB -W 48:00
#BSUB -o out.%J
#BSUB -n 24
#BSUB -x
#BSUB -R "intel=4"
#BSUB -R scratch2

$HOME/build/Python-3.6.7/python $HOME/Change_Arrays.py &> output_change_array.txt&
wait
