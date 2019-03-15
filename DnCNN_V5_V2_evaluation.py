#!/usr/bin/env python
# coding: utf-8

# Loading some packages

# In[1]:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import gc
import datetime
from PIL import Image
from SSIM_PIL import compare_ssim


# Helper functions for the network

# In[2]:


def conv_layer(tensor_in, name_layer, is_training):
    x = tf.layers.conv2d(
        inputs = tensor_in,
        filters = 64,
        kernel_size = [3, 3],
        padding = "same",
        kernel_initializer=tf.truncated_normal_initializer(stddev=(2 / (9.0 * 64)) ** 0.5),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-6),
        activation= None,
        name = name_layer,
        use_bias=False)
    
    x = tf.layers.batch_normalization(x, name = name_layer + "_bn",
                                             center=True, 
                                             scale=True, 
                                             training=is_training)
    
    return tf.nn.relu(x, name = name_layer + "_relu")
# Definition of the NN

# In[4]:


def AutoEncoder_model(features, labels, mode):
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
    
    
    if not (mode == tf.estimator.ModeKeys.PREDICT):
        # Output all learnable variables for tensorboard
        for var in tf.trainable_variables():
            name = var.name
            name = name.replace(':', '_')
            tf.summary.histogram(name, var)
        merged_summary = tf.summary.merge_all()

        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.summary.image("Input_Image", input_layer, max_outputs = 1)
            tf.summary.image("Output_Image", final_layer, max_outputs = 1)
            tf.summary.image("True_Image", labels,  max_outputs = 1)
            tf.summary.histogram("Summary_final_layer", final_layer)
            tf.summary.histogram("Summary_labels", labels)
            
    # Give output in prediction mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions=final_layer)
    

    # Calculate Loss (for both Train and EVAL modes)
    # See that the residual learning is implemented here.
    loss = tf.losses.mean_squared_error(labels = labels , predictions = final_layer)
    tf.summary.scalar("Value_Loss_Function", loss)
        
    # Configure Learning when training.
    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            original_optimizer = tf.train.AdamOptimizer(learning_rate =  0.003)
            optimizer = tf.contrib.estimator.clip_gradients_by_norm(original_optimizer, clip_norm=5.0)
            train_op = optimizer.minimize(loss = loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)



# Running Specification

# In[4]:


runconf = tf.estimator.RunConfig(save_summary_steps=5, log_step_count_steps = 10)

AutoEncoder = tf.estimator.Estimator(config=runconf,
    model_fn=AutoEncoder_model, model_dir="/scratch2/ttoebro/models/DnCNN_V5_V2")


# Evalutation

# Evaluate on rad_41

# Load Data

# In[5]:


X = np.load('/scratch2/ttoebro/data/X_test_rad41.npy')
Y = np.load('/scratch2/ttoebro/data/Y_test_rad41.npy')


# In[6]:


# Evaluate the model and print results
predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X[:,:,:,:]},
    y=Y[:,:,:,:],
    batch_size = 1,
    shuffle=False)
predict_results = AutoEncoder.predict(input_fn=predict_input_fn)


# In[7]:


sum_mae_rad41 = 0
sum_mse_rad41 = 0
sum_ssim_rad41 = 0


# In[8]:


for im_num in range(0, Y.shape[0]):
    prediction = next(predict_results)
    true_image = Y[im_num,:,:,:]
    sum_mae_rad41 += np.mean(np.abs(prediction - true_image))
    sum_mse_rad41 += np.mean(np.power((prediction - true_image), 2))
    sum_ssim_rad41 += compare_ssim(Image.fromarray((prediction[:,:,0] * 255).astype('uint8'), 'L'),
             Image.fromarray((true_image[:,:,0] * 255).astype('uint8'), 'L'))
    if(im_num % 10000 == 0):
        print("Current mae is " + str(sum_mae_rad41) + " and mse is " + str(sum_mse_rad41) + " and ssim is " + str(sum_ssim_rad41) + " at picture " + str(im_num))


# In[ ]:


sum_mae_rad41 /= X.shape[0]
sum_mse_rad41 /= X.shape[0]
sum_ssim_rad41 /= X.shape[0]


# Evaluate on rad_15

# In[ ]:


X = np.load('/scratch2/ttoebro/data/X_test_rad_15.npy')
Y = np.load('/scratch2/ttoebro/data/Y_test_rad_15.npy')


# In[ ]:


# Evaluate the model and print results
predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X[:,:,:,:]},
    y=Y[:,:,:,:],
    batch_size = 1,
    shuffle=False)
predict_results = AutoEncoder.predict(input_fn=predict_input_fn)


# In[ ]:


sum_mae_rad15 = 0
sum_mse_rad15 = 0
sum_ssim_rad15 = 0


# In[ ]:


for im_num in range(0, Y.shape[0]):
    prediction = next(predict_results)
    true_image = Y[im_num,:,:,:]
    sum_mae_rad15 += np.mean(np.abs(prediction - true_image))
    sum_mse_rad15 += np.mean(np.power((prediction - true_image), 2))
    sum_ssim_rad15 += compare_ssim(Image.fromarray((prediction[:,:,0] * 255).astype('uint8'), 'L'),
             Image.fromarray((true_image[:,:,0] * 255).astype('uint8'), 'L'))
    if(im_num % 10000 == 0):
        print("Current mae is " + str(sum_mae_rad15) + " and mse is " + str(sum_mse_rad15) + " and ssim is " + str(sum_ssim_rad15) + " at picture " + str(im_num))


# In[ ]:


sum_mae_rad15 /= X.shape[0]
sum_mse_rad15 /= X.shape[0]
sum_ssim_rad15 /= X.shape[0]


# Evaluate on Pois

# In[ ]:


X = np.load('/scratch2/ttoebro/data/X_test_pois_1_9.npy')
Y = np.load('/scratch2/ttoebro/data/Y_test_pois_1_9.npy')


# In[ ]:


# Evaluate the model and print results
predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X[:,:,:,:]},
    y=Y[:,:,:,:],
    batch_size = 1,
    shuffle=False)
predict_results = AutoEncoder.predict(input_fn=predict_input_fn)


# In[ ]:


sum_mae_pois_1_9 = 0
sum_mse_pois_1_9 = 0
sum_ssim_pois_1_9= 0


# In[ ]:


for im_num in range(0, Y.shape[0]):
    prediction = next(predict_results)
    true_image = Y[im_num,:,:,:]
    sum_mae_pois_1_9 += np.mean(np.abs(prediction - true_image))
    sum_mse_pois_1_9 += np.mean(np.power((prediction - true_image), 2))
    sum_ssim_pois_1_9 += compare_ssim(Image.fromarray((prediction[:,:,0] * 255).astype('uint8'), 'L'),
             Image.fromarray((true_image[:,:,0] * 255).astype('uint8'), 'L'))
    if(im_num % 10000 == 0):
        print("Current mae is " + str(sum_mae_pois_1_9) + " and mse is " + str(sum_mse_pois_1_9) + " and ssim is " + str(sum_ssim_pois_1_9) + " at picture " + str(im_num))


# In[ ]:


sum_mae_pois_1_9 /= X.shape[0]
sum_mse_pois_1_9 /= X.shape[0]
sum_ssim_pois_1_9 /= X.shape[0]


# In[ ]:


dic_data_frame = {'MAE' : [sum_mae_rad41, sum_mae_rad15, sum_mae_pois_1_9],
                 'MSE' : [sum_mse_rad41, sum_mse_rad15, sum_mse_pois_1_9],
                 'SSIM' : [sum_ssim_rad41, sum_ssim_rad15, sum_ssim_pois_1_9]}


# In[21]:
result = pd.DataFrame(dic_data_frame, index=['Rad_41', 'Rad_15', 'pois_1_9'])


# In[23]:


result.to_csv('/scratch2/ttoebro/results/DnCNN_V5_V2')