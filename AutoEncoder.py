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
X_train = np.load('/scratch2/ttoebro/data/X_train.npy')

#Import Y
Y_train = np.load('/scratch2/ttoebro/data/Y_train.npy')
gc.collect()

# Layers for the NN #
def conv_2(tensor_in, name_layer, n_filter):
    x = tf.layers.conv2d(
        inputs = tensor_in,
        filters = n_filter,
        kernel_size = [3, 3],
        padding = "same",
        activation= tf.nn.relu,
        name = name_layer + "_conv_1")
    x = tf.layers.conv2d(
        inputs = x,
        filters = n_filter,
        kernel_size = [3, 3],
        padding = "same",
        activation= tf.nn.relu,
        name = name_layer + "_conv_2")
    
    return x

def level_up(tensor_in, insert_layer, name_layer, n_filter):
    #print("Shape before level up: " + str(tensor_in.shape))
    x = tf.layers.conv2d_transpose(
            tensor_in,
            filters=n_filter,
            kernel_size=2,
            strides=2,
            padding = 'same',
            name=name_layer + "_upconv")
    #print("Shape after level up: " + str(x.shape))
    
    x = tf.concat([insert_layer, x], axis=-1, name=name_layer + "_insert")
    #print("Shape after putting in other vector: " + str(x.shape))
    
    x = tf.layers.conv2d(
        inputs = x,
        filters = n_filter,
        kernel_size = [3, 3],
        padding = "same",
        activation= tf.nn.relu,
        name = name_layer + "_conv_1")
    #print("Shape after first conv in level up: " + str(x.shape))
    
    x = tf.layers.conv2d(
        inputs = x,
        filters = n_filter,
        kernel_size = [3, 3],
        padding = "same",
        activation= tf.nn.relu,
        name = name_layer + "_conv_2")
    #print("Shape after second conv in level up: " + str(x.shape))
    
    return x

# Definition of the NN #
def AutoEncoder_model(features, labels, mode):
    
    ## Hyper paramters ##
    eps_start = 0.001 #learning rate in the beginning
    eps_end = 0.0005 #final learning rate
    tau = 40000 # number of iterations afterwards is the learning rate constant
    #####################
    
    is_training_mode = (mode == tf.estimator.ModeKeys.TRAIN)
    
    # Input Tensor
    input_tensor = features['x']
    
    # Level 0
    level_0 = conv_2(input_tensor, "level_0", n_filter = 64)
    level_0_pool = tf.layers.max_pooling2d(level_0, (2, 2), strides=(2, 2), name="level_0_pooling")
    
    # Level 1
    level_1 = conv_2(level_0_pool, "level_1", n_filter = 128)
    level_1_pool = tf.layers.max_pooling2d(level_1, (2, 2), strides=(2, 2), name="level_1_pooling")
    
    # Level 2
    level_2 = conv_2(level_1_pool, "level_2", n_filter = 256)
    level_2_pool = tf.layers.max_pooling2d(level_2, (2, 2), strides=(2, 2), name="level_2_pooling")
    
    # Level 3
    level_3 = conv_2(level_2_pool, "level_3", n_filter = 512)
    level_3_pool = tf.layers.max_pooling2d(level_3, (2, 2), strides=(2, 2), name="level_3_pooling")
    
    # Level 4
    level_4 = conv_2(level_3_pool, "level_4", n_filter = 1024)
    
    # Level 3
    level_3_up = level_up(level_4,level_3,"level_3_up" , n_filter = 512)
    
    # Level 2
    level_2_up = level_up(level_3_up,level_2, "level_2_up" , n_filter = 256)
    
    # Level 1
    level_1_up = level_up(level_2_up,level_1, "level_1_up" , n_filter = 128)
    
    # Level 0
    level_0_up = level_up(level_1_up,level_0,"level_0_up"  , n_filter = 64)
    
        # final 
    final_layer = tf.layers.conv2d(
        inputs = level_0_up,
        filters = 1,
        kernel_size = [1, 1],
        padding = "same",
        activation = None,
        name = "final_layer")
    
    # Give output in prediction mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions=final_layer)
    
    # Output all learnable variables for tensorboard
    for var in tf.trainable_variables():
        name = var.name
        name = name.replace(':', '_')
        tf.summary.histogram(name, var)
    merged_summary = tf.summary.merge_all()
    
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
        optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
        train_op = optimizer.minimize(loss = loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

runconf = tf.estimator.RunConfig(save_summary_steps=5, log_step_count_steps = 10)
save_dir = "/scratch2/ttoebro/models/AutoEncoder_" + str(datetime.datetime.now())[0:19].replace("-", "_").replace(" ", "_").replace(":", "_").replace(".", "_")
AutoEncoder = tf.estimator.Estimator(config=runconf,
    model_fn=AutoEncoder_model, model_dir=save_dir)
train = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=Y_train,
    batch_size=12,
    num_epochs=None,
    shuffle=True)
AutoEncoder.train(
    input_fn=train,
    steps=1000000)

