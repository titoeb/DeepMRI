#!/bin/sh
#BSUB -q gpu
#BSUB -W 48:00
#BSUB -o out/Auto_Encoder_spulen_V3.%J
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

$HOME/build/Python-3.6.7/python $HOME/Auto_Encoder_spulen_V3.py &> output/Auto_Encoder_spulen_V3_Run_1&
wait
