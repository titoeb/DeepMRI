# Load packages
import numpy as np
# import pandas as pd
import datetime
import os
import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import _fspecial_gauss
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes

# Import Data
# Import X
X_train = np.load('/scratch2/truhkop/knee1/data/X_train.npy')

#Import Y
Y_train = np.load('/scratch2/truhkop/knee1/data/Y_train.npy')

lossflavour = ['MAE', 'MSE', 'SSIM', 'MS-SSIM', 'MS-SSIM-GL1'][1]
def conv_2(tensor_in, name_layer, n_filter, mode, is_start = False):

        
    x = tf.layers.conv2d(
        inputs = tensor_in,
        filters = n_filter,
        kernel_size = [3, 3],
        padding = "same",
        activation= None,
        name = name_layer + "_conv_1")
    
    x = tf.layers.batch_normalization(x,
                                      axis = -1,
                                      training = (mode == tf.estimator.ModeKeys.TRAIN),
                                                  name = name_layer + "_bn_1")
    x = tf.nn.relu(x, name = name_layer + "relu_1")
    
    x = tf.layers.conv2d(
        inputs = x,
        filters = n_filter,
        kernel_size = [3, 3],
        padding = "same",
        activation= None,
        name = name_layer + "_conv_2")
    
    x = tf.layers.batch_normalization(x,
                                      axis = -1,
                                      training = (mode == tf.estimator.ModeKeys.TRAIN),
                                                  name = name_layer + "_bn_2")
    x = tf.nn.relu(x, name = name_layer + "relu_2")
    
    return x

def level_up(tensor_in, insert_layer, name_layer, n_filter, mode):
    #print("Shape before level up: " + str(tensor_in.shape))

    x = tf.layers.conv2d_transpose(
            tensor_in,
            filters=n_filter,
            kernel_size=2,
            strides=2,
            padding = 'same',
            activation = None,
            name=name_layer + "_upconv")
   # print("Shape after level up: " + str(x.shape))
    
    x = tf.layers.batch_normalization(x,
                                      axis = -1,
                                      training = (mode == tf.estimator.ModeKeys.TRAIN),
                                                  name = name_layer + "_bn_1")
    x = tf.nn.relu(x, name_layer + "relu_1")
    
    #print("x has dim " + str(x.shape) + " and stuff to insert has dim " + str(insert_layer.shape))
    x = tf.concat([insert_layer, x], axis=-1, name=name_layer + "_insert")
    #print("Shape after putting in other vector: " + str(x.shape))
    

    x = tf.layers.conv2d(
        inputs = x,
        filters = n_filter,
        kernel_size = [3, 3],
        padding = "same",
        activation= None,
        name = name_layer + "_conv_1")
    
    x = tf.layers.batch_normalization(x,
                                      axis = -1,
                                      training = (mode == tf.estimator.ModeKeys.TRAIN),
                                                  name = name_layer + "_bn_2")
    
    x = tf.nn.relu(x, name = name_layer + "relu_2")
    #print("Shape after first conv in level up: " + str(x.shape))

    x = tf.layers.conv2d(
        inputs = x,
        filters = n_filter,
        kernel_size = [3, 3],
        padding = "same",
        activation= None,
        name = name_layer + "_conv_2")
    
    x = tf.layers.batch_normalization(x,
                                      axis = -1,
                                      training = (mode == tf.estimator.ModeKeys.TRAIN),
                                                  name = name_layer + "_bn_3")
    
    x = tf.nn.relu(x, name = name_layer + "relu_3")
    #print("Shape after second conv in level up: " + str(x.shape))
    
    return x

# Definition of the NN #
def AutoEncoder_model(features, labels, mode):

    # Input Tensor
    input_tensor = features
    
    # Level 0
    level_0 = conv_2(input_tensor, "level_0", n_filter = 64, mode = mode, is_start = True)
    level_0_pool = tf.layers.max_pooling2d(level_0, (2, 2), strides=(2, 2), name="level_0_pooling")
    
    # Level 1
    level_1 = conv_2(level_0_pool, "level_1", n_filter = 128, mode = mode, is_start = False)
    level_1_pool = tf.layers.max_pooling2d(level_1, (2, 2), strides=(2, 2), name="level_1_pooling")
    
    # Level 2
    level_2 = conv_2(level_1_pool, "level_2", n_filter = 256, mode = mode, is_start = False)
    level_2_pool = tf.layers.max_pooling2d(level_2, (2, 2), strides=(2, 2), name="level_2_pooling")
    
    # Level 3
    level_3 = conv_2(level_2_pool, "level_3", n_filter = 512, mode = mode, is_start = False)
    level_3_pool = tf.layers.max_pooling2d(level_3, (2, 2), strides=(2, 2), name="level_3_pooling")
    
    # Level 4
    level_4 = conv_2(level_3_pool, "level_4", n_filter = 1024, mode = mode, is_start = False)
    level_4_pool = tf.layers.max_pooling2d(level_4, (2, 2), strides=(2, 2), name="level_4_pooling")
    
    # level 5
    level_5 = conv_2(level_4_pool, "level_5", n_filter = 1024, mode = mode, is_start = False)
    level_5_pool = tf.layers.max_pooling2d(level_5, (2, 2), strides=(2, 2), name="level_5_pooling")
    
    # level 6
    level_6 = conv_2(level_5_pool, "level_6", n_filter = 1024, mode = mode, is_start = False)
    
    # level 5
    level_5_up = level_up(level_6,level_5,"level_5_up" , n_filter = 1024, mode = mode)
    
    # level 4
    level_4_up = level_up(level_5_up,level_4,"level_4_up" , n_filter = 1024, mode = mode)
    
    # Level 3
    level_3_up = level_up(level_4_up,level_3,"level_3_up" , n_filter = 512, mode = mode)
    
    # Level 2
    level_2_up = level_up(level_3_up,level_2, "level_2_up" , n_filter = 256, mode = mode)
    
    # Level 1
    level_1_up = level_up(level_2_up,level_1, "level_1_up" , n_filter = 128, mode = mode)
    
    # Level 0
    level_0_up = level_up(level_1_up,level_0,"level_0_up"  , n_filter = 64, mode = mode)
    
    # final 
    final_layer = tf.layers.conv2d(
        inputs = level_0_up,
        filters = 1,
        kernel_size = [1, 1],
        padding = "same",
        activation = None,
        name = "final_layer")

    
    if not (mode == tf.estimator.ModeKeys.PREDICT):
        # Output all learnable variables for tensorboard
        for var in tf.trainable_variables():
            name = var.name
            name = name.replace(':', '_')
            tf.summary.histogram(name, var)
        merged_summary = tf.summary.merge_all()

        if mode == tf.estimator.ModeKeys.PREDICT:
            tf.summary.image("Input_Image", input_tensor, max_outputs = 1)
            tf.summary.image("Output_Image", final_layer, max_outputs = 1)
            tf.summary.image("True_Image", labels,  max_outputs = 1)
            tf.summary.histogram("Summary_final_layer", final_layer)
            tf.summary.histogram("Summary_labels", labels)
        
    # Give output in prediction mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions=final_layer)
    


    # (losses) -----------------------------------------------------------------
    # Calculate Loss (for both Train and EVAL modes)
    def l1 (prediction, labels):
        return tf.losses.absolute_difference(
            labels=labels,
            predictions=prediction)

    def mse (prediction, labels):
        return tf.losses.mean_squared_error(
            labels=labels,
            predictions=prediction)


    def ssim (prediction, labels):
        # ssim returns a tensor containing ssim value for each image in batch: reduce!
        return 1 - tf.reduce_mean(
            tf.image.ssim(
                prediction,
                labels,
                max_val=1))

    def ssim_multiscale(prediction, label):
        return 1- tf.reduce_mean(
            tf.image.ssim_multiscale(
                    img1=label,
                    img2=prediction,
                    max_val=1)
            )

    def loss_ssim_multiscale_gl1 (prediction, label, alpha=0.84):
        ''' Loss function, calculating alpha * Loss_msssim + (1-alpha) gaussiankernel * L1_loss
        according to 'Loss Functions for Image Restoration with Neural Networks' [Zhao]
        :alpha: default value accoording to paper'''

        # stride according to MS-SSIM source
        kernel_on_l1 = tf.nn.conv2d(
            input=tf.subtract(label, prediction),
            filter=gaussiankernel,
            strides=[1, 1, 1, 1],
            padding='VALID')

        # total no. of pixels: number of patches * number of pixels per patch
        img_patch_norm = tf.to_float(kernel_on_l1.shape[1] * filter_size ** 2)
        gl1 = tf.reduce_sum(kernel_on_l1) / img_patch_norm

        # ssim_multiscale already calculates the dyalidic pyramid (with as replacment avg.pooling)
        msssim = tf.reduce_sum(
            tf.image.ssim_multiscale(
                img1=label,
                img2=prediction,
                max_val=1)
        )
        return alpha * (1 - msssim) + (1 - alpha) * gl1

        # Discrete Gaussian Kernel (required only in MS-SSIM-GL1 case)
        # not in MS-SSIM-GL1 function, as it is executed only once
        # values according to MS-SSIM source code

    filter_size = constant_op.constant(11, dtype=dtypes.int32)
    filter_sigma = constant_op.constant(1.5, dtype=features.dtype)
    gaussiankernel = _fspecial_gauss(
        size=filter_size,
        sigma=filter_sigma
    )

    # for TRAIN & EVAL
    loss = {
        'MAE': l1,
        'MSE': mse,
        'SSIM': ssim,
        'MS-SSIM':ssim_multiscale,
        'MS-SSIM-GL1': loss_ssim_multiscale_gl1
    }[lossflavour](final_layer, labels)

    tf.summary.scalar("Value_Loss_Function", loss)
        
    # Configure Learning when training.
    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            original_optimizer = tf.train.AdamOptimizer(learning_rate =  0.005)
            optimizer = tf.contrib.estimator.clip_gradients_by_norm(original_optimizer, clip_norm=5.0)
            train_op = optimizer.minimize(loss = loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)



AutoEncoder = tf.estimator.Estimator(
    config= tf.estimator.RunConfig(save_summary_steps=2, log_step_count_steps = 10),
    model_fn=AutoEncoder_model,
    model_dir="/scratch2/truhkop/model/AutoEncoder_{}".format(lossflavour))

train = tf.estimator.inputs.numpy_input_fn(
    x=X_train,
    y=Y_train,
    batch_size=2,
    num_epochs=None,
    shuffle=True)

AutoEncoder.train(
    input_fn=train,
    steps=100)

