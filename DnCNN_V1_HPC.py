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
def conv_layer(tensor_in, name_layer, is_training):
    x = tf.layers.conv2d(
        inputs = tensor_in,
        filters = 64,
        kernel_size = [3, 3],
        padding = "same",
        activation= None,
        name = name_layer)
    
    x = tf.layers.batch_normalization(x, name = name_layer + "_bn",
                                             center=True, 
                                             scale=True, 
                                             training=is_training)
    
    return tf.nn.relu(x, name = name_layer + "_relu")
	
def cnn_model_fn(features, labels, mode):
    
    ## Hyper paramters ##
    eps_start = 0.05 #learning rate in the beginning
    eps_end = eps_start / 100 #final learning rate
    tau = 20000 # number of iterations afterwards is the learning rate constant
    #####################
    
    # Input Layer
    input_layer = features['x']
    
    # Convolutional layer #1     
    conv1 = tf.layers.conv2d(
        inputs = input_layer,
        filters = 64,
        kernel_size = 3,
        padding = "same",
        activation= tf.nn.relu,
        name = "Conv_1")
    is_training_mode = (mode == tf.estimator.ModeKeys.TRAIN)
    
     # 18 of the middle layers with Convolution, batch normalization and afterwards ReLu
    conv2 = conv_layer(conv1, "conv2", is_training = is_training_mode)
    conv3 = conv_layer(conv2, "conv3", is_training = is_training_mode)
    conv4 = conv_layer(conv3, "conv4", is_training = is_training_mode)
    conv5 = conv_layer(conv4, "conv5", is_training = is_training_mode)
    conv6 = conv_layer(conv5, "conv6", is_training = is_training_mode)
    conv7 = conv_layer(conv6, "conv7", is_training = is_training_mode)
    conv8 = conv_layer(conv7, "conv8", is_training = is_training_mode)
    conv9 = conv_layer(conv8, "conv9", is_training = is_training_mode)
    conv10 = conv_layer(conv9, "conv10", is_training = is_training_mode)
    conv11 = conv_layer(conv10, "conv11", is_training = is_training_mode)
    conv12 = conv_layer(conv11, "conv12", is_training = is_training_mode)
    conv13 = conv_layer(conv12, "conv13", is_training = is_training_mode)
    conv14 = conv_layer(conv13, "conv14", is_training = is_training_mode)
    conv15 = conv_layer(conv14, "conv15", is_training = is_training_mode)
    conv16 = conv_layer(conv15, "conv16", is_training = is_training_mode)
    conv17 = conv_layer(conv16, "conv17", is_training = is_training_mode)
    conv18 = conv_layer(conv17, "conv18", is_training = is_training_mode)
    conv19 = conv_layer(conv18, "conv19", is_training = is_training_mode)

    # final 
    final_layer = tf.layers.conv2d(
        inputs = conv19,
        filters = 1,
        kernel_size = [1, 1],
        padding = "same",
        activation = None,
        name = "final_layer") + input_layer
    
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions=final_layer)
    
    # Calculate Loss (for both Train and EVAL modes)
    # See that the residual learning is implemented here.
    loss = tf.losses.mean_squared_error(labels = labels , predictions = final_layer)
    tf.summary.scalar("Value_Loss_Function", loss)
        
    # Configure the Training OP (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # calculate current learning rate:
        alpha = tf.train.get_global_step() / tau
        cur_learning_rate = tf.maximum(tf.constant(0.0, dtype ='float64'),(1-alpha)) * eps_start + tf.minimum(tf.constant(1.0, dtype ='float64') , alpha) * eps_end
        tf.summary.scalar("Learning_rate", cur_learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate = cur_learning_rate)
        train_op = optimizer.minimize(loss = loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        
    # Output all learnable variables for tensorboard
    for var in tf.trainable_variables():
        name = var.name
        name = name.replace(':', '_')
        tf.summary.histogram(name, var)
    merged_summary = tf.summary.merge_all()
    
    # Add evaluation metrics
    eval_metric_ops = {
        "accuracy": tf.metrics.mean_squared_error(
            labels=labels, predictions=final_layer)}
    return tf.estimator.EstimatorSpec(
          mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)    


# rename test and train
train_data = X_train
train_labels = Y_train
eval_data = X_eval
eval_labels = Y_eval

runconf = tf.estimator.RunConfig(save_summary_steps=20, log_step_count_steps = 20)
save_dir = "/scratch2/ttoebro/models/" + str(datetime.datetime.now())[0:19].replace("-", "_").replace(" ", "_").replace(":", "_").replace(".", "_")

ImpNet = tf.estimator.Estimator(config=runconf,
    model_fn=cnn_model_fn, model_dir= save_dir)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=Y_train,
    batch_size=12,
    num_epochs=None,
    shuffle=True)

# run model
ImpNet.train(
    input_fn=train_input_fn,
    steps=100000)
