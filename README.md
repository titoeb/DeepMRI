# DeepMRI
This repository contains the scripts from the DeepMRI project. 

## Create Data 
The 'Create Data' folder contains shell and python scripts to creating undersample 2d images from three dimensional input data using different unersampling schemes. These can then be converted to numpy arrays.

## network architectures
The 'network architectures' folders contains a variety of convolutional networks in tensorflow that we can be trained to recover the undersamples images.

## Evaluation
Here, the evaluation scripts are stored that compute the performance of the networks based on the average L1, L2 and SSIM difference between the original and the reconstructed images.
