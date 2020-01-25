#!/usr/bin/env python
# coding: utf-8

# Quality Detector
# In this script I will create toy CNN that is supposed to detect whether the input image is of a good quality or has artefacts.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# ## Data Preprocessing

# I will start by loading two of the images in. Then I will select from the originals each only one. Aftwards, I will select the 500 images in good and bad quality from the two image and create the classification label for it.
P3_X = np.load('/home/cloud/MRT_Data/unziped/out/P3_X.npy')
P3_Y = np.load('/home/cloud/MRT_Data/unziped/out/P3_Y.npy')
P14_X = np.load('/home/cloud/MRT_Data/unziped/out/P14_X.npy')
P14_Y = np.load('/home/cloud/MRT_Data/unziped/out/P14_Y.npy')

# Select images from Y only once
# To match the number of distorted images in X, the images in Y are each repeated five times subsequently. To get unique images I will only select the first once.

P3_Y = P3_Y[[i for i in range(0,P3_Y.shape[0],5)],:,:,:]
P14_Y = P14_Y[[i for i in range(0,P14_Y.shape[0],5)],:,:,:]

# For the distorted pictures saved in P3_X and P14_X I will select on of the five distorted images randomly.
P3_X = P3_X[[i + np.random.randint(0,4) for i in range(0,P3_X.shape[0],5)], :, :,:]
P14_X = P14_X[[i + np.random.randint(0,4) for i in range(0,P14_X.shape[0],5)], :, :,:]
shape_x = (4 * P3_X.shape[0],) + P3_X.shape[1:4]

X = np.empty(shape_x,dtype='uint8')
X[0:811,:,:,:] = P3_Y
X[811:1622,:,:,:] = P14_Y
X[1622:2433,:,:,:] = P3_X
X[2433:3244,:,:,:] = P14_X

Y = np.array([1] * 1622 + [0] * 1622 , dtype='uint8')
shuffle_index = np.random.choice(range(0,Y.shape[0]), size = Y.shape[0], replace = False)
X = X[shuffle_index,:,:,:]
Y = Y[shuffle_index,]

# Create train and test and validation set
X_train = X[1:2270, :,:,:]
X_eval = X[2270:3244,:,:,:]

Y_train = Y[1:2270,]
Y_eval = Y[2270:3244,]

# Create and train CNN
# Create the CNN model
def cnn_model_fn(features, labels, mode):
    print(features['x'].shape)
    print(labels.shape)
   
# Load train and test data
train_data = X_train
train_labels = Y_train
eval_data = X_eval
eval_labels = Y_eval

# Create the estimator
noise_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/noise_classifier")

# Set Up a Logging Hook
# Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)

# Train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)

noise_classifier.train(
    input_fn=train_input_fn,
    steps=100)