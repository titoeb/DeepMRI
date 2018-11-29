# Load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
import datetime
import os
import tensorflow as tf

# Import Data 

# Import X
X_1 = np.load('/scratch2/ttoebro/data/P10_X.npy')
X_2 = np.load('/scratch2/ttoebro/data/P6_X.npy')
X_3 = np.load('/scratch2/ttoebro/data/P3_X.npy')
X_4 = np.load('/scratch2/ttoebro/data/P1_X.npy')
X = np.concatenate(seq = (X_1, X_2, X_3, X_4), axis=0)
print(X.shape)
del X_1, X_2, X_3, X_4
X = X.reshape([X.shape[0], 256, 256, 1])
gc.collect()

#Import Y
Y_1 = np.load('/scratch2/ttoebro/data/P10_Y.npy')
Y_2 = np.load('/scratch2/ttoebro/data/P6_Y.npy')
Y_3 = np.load('/scratch2/ttoebro/data/P3_Y.npy')
Y_4 = np.load('/scratch2/ttoebro/data/P1_Y.npy')
Y = np.concatenate(seq = (Y_1, Y_2, Y_3, Y_4), axis=0)
print(Y.shape)
del Y_1, Y_2, Y_3, Y_4
Y = Y.reshape([Y.shape[0], 256, 256, 1])
gc.collect()

# Create test and validation set
train_frac = 0.8
train_index = int(train_frac * Y.shape[0])
X_train = X[0:train_index,:,:,:]
X_eval = X[train_index:X.shape[0],:,:,:]
Y_train = Y[0:train_index,:,:]
Y_eval = Y[train_index:X.shape[0],:,:,:]

# Definition of the network
def cnn_model_fn(features, labels, mode):
    
    # Input Layer
    input_layer = features['x']
    
    # Convolutional layer #1     
    conv1 = tf.layers.conv2d(
        inputs = input_layer,
        filters = 64,
        strides = 2,
        kernel_size = 5,
        padding = "same",
        activation= tf.nn.relu,
        name = "Conv_1")
    
    # Convolutional layer #2
    conv2 = tf.layers.conv2d(
        inputs = conv1,
        filters = 128,
        strides = [2, 2],
        kernel_size = [5, 5],
        padding = "same",
        activation= tf.nn.relu,
        name = "Conv_2")
    
    conv2_bn = tf.layers.batch_normalization(conv2,
                                             name = "Conv_2_bn",
                                             center=True, 
                                             scale=True, 
                                             training=(mode == tf.estimator.ModeKeys.TRAIN))

        
    # Convolutional layer #3
    conv3 = tf.layers.conv2d(
        inputs = conv2_bn,
        filters = 256,
        strides = [2, 2],
        kernel_size = [5, 5],
        padding = "same",
        activation = tf.nn.relu,
        name = "Conv_3")
    
    conv3_bn = tf.layers.batch_normalization(conv3,
                                             name = "Conv_3_bn",
                                             center=True, 
                                             scale=True, 
                                             training=(mode == tf.estimator.ModeKeys.TRAIN))
    
    
    # Deconvolutional layer #1
    deconv1 = tf.layers.conv2d_transpose(
        inputs = conv3_bn,
        filters = 256,
        strides = [2, 2],
        kernel_size = [5, 5],
        padding = "same",
        activation= tf.nn.relu,
        name = "Deconv_1")
    
    deconv1_bn = tf.layers.batch_normalization(deconv1,
                                             name = "deconv_1_bn",
                                             center=True, 
                                             scale=True, 
                                             training=(mode == tf.estimator.ModeKeys.TRAIN))

    
    # Deconvolutional layer #2
    deconv2 = tf.layers.conv2d_transpose(
        inputs = deconv1_bn,
        filters = 128,
        strides = [2, 2],
        kernel_size = [5, 5],
        activation= tf.nn.relu,
        padding = "same",
        name = "Deconv_2")
    
    deconv2_bn = tf.layers.batch_normalization(deconv2,
                                             name = "deconv_2_bn",
                                             center=True, 
                                             scale=True, 
                                            training=(mode == tf.estimator.ModeKeys.TRAIN))

    
    # Deconvolutional layer #3
    deconv3 = tf.layers.conv2d_transpose(
        inputs = deconv2_bn,
        filters = 64,
        strides = [2, 2],
        kernel_size = [5, 5],
        padding = "same", 
        activation= tf.nn.relu,
        name = "Deconv_3")
    
    deconv3_bn = tf.layers.batch_normalization(deconv3,
                                             name = "deconv_3_bn",
                                             center=True, 
                                             scale=True, 
                                             training=(mode == tf.estimator.ModeKeys.TRAIN))
    
    # final covolution to get to 3 layers
    conv4 = tf.layers.conv2d(
        inputs = deconv3_bn,
        filters = 1,
        kernel_size = [1, 1],
        padding = "same",
        activation = tf.nn.relu,
        name = "Conv_4") + input_layer

    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions=conv4)
    
    # Calculate Loss (for both Train and EVAL modes)
    loss = tf.losses.absolute_difference(labels = labels , predictions = conv4)
    tf.summary.scalar("Value_Loss_Function", loss)

    for var in tf.trainable_variables():
        name = var.name
        name = name.replace(':', '_')
        tf.summary.histogram(name, var)
    merged_summary = tf.summary.merge_all()
        
    # Configure the Training OP (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
        train_op = optimizer.minimize(loss = loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        
    # Add evaluation metrics
    eval_metric_ops = {
        "accuracy": tf.metrics.mean_absolute_error(
            labels=labels, predictions=conv4)}
    return tf.estimator.EstimatorSpec(
          mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)    

# rename test and train
train_data = X_train
train_labels = Y_train
eval_data = X_eval
eval_labels = Y_eval

runconf = tf.estimator.RunConfig(save_summary_steps=5, log_step_count_steps = 10)
save_dir = "/scratch2/ttoebro/models/" + str(datetime.datetime.now())[0:19].replace("-", "_").replace(" ", "_").replace(":", "_").replace(".", "_")

ImpNet = tf.estimator.Estimator(config=runconf,
    model_fn=cnn_model_fn, model_dir= save_dir)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=Y_train,
    batch_size=32,
    num_epochs=10,
    shuffle=True)

# run model
ImpNet.train(
    input_fn=train_input_fn,
    steps=10)
