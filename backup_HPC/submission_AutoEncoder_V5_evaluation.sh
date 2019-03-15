#!/bin/sh
#BSUB -q gpu
#BSUB -W 48:00
#BSUB -o out/AutoEncoder_V5_eval.%J
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

$HOME/build/Python-3.6.7/python $HOME/DnCNN_V5_evaluation.py  &> output/output_AutoEncoder_V5_eval_Run_1&
wait
