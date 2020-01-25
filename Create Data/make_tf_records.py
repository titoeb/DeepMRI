#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

def _float_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value = value))

def convert_to_records_all(input_path = "/home/cloud/MRT_Data/unziped/out/P",
                           output_path =  "/home/cloud/data/MRI.tfrecordsk",
                           n = 1000):
    with tf.python_io.TFRecordWriter(output_path) as writer: 
        for i in range(1,21):
            print("Starting to write image " + str(i) + " out!")
            
            # Create the path of the current input nd-array
            path_X = input_path + str(i) + "_X.npy"
            path_y = input_path + str(i) + "_Y.npy"

            # Load X,Y
            X = np.load(path_X)
            Y = np.load(path_y)
            
            # Select only n images per input!
            select = np.random.choice(np.arange(0, 4055), size = n)
            X = X[select, :, :]
            Y = Y[select, :, :]
        
            # Transform them to tf.records
            for i in range(X.shape[0]):
                example = tf.train.Example(features = tf.train.Features(
                                feature = 
                                {
                                    'image':_float_feature(X[i].tostring()),
                                    'label':_float_feature(Y[i].tostring())
                                }
                                                                    )
                                      )
                # given the 'tupel', serialize this example
                writer.write(example.SerializeToString())
                if i%500==0:
                    print('writing {}th image'.format(i))
            print("Image " + str(i) + " was written out!")

convert_to_records_all()

