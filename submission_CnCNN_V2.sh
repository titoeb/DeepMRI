#!/bin/sh
#BSUB -q gpu
#BSUB -W 48:00
#BSUB -o out/out.%J
#BSUB -n 24
#BSUB -R "ngpus=2"
#BSUB -R "rusage[ngpus_shared=24]"
#BSUB -x
#BSUB -R scratch2

module load cuda90/blas
module load cuda90/fft
module load cuda90/nsight
module load cuda90/profiler
module load cuda90/toolkit
module load cudnn/90v7.0.4

$HOME/build/Python-3.6.7/python $HOME/DnCNN_V2.py &> output/output_DnCNN_V2_Run_2&
wait
