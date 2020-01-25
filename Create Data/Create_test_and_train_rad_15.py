#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import gc

# Load the whole image
Y_name = '/home/cloud/Create_Data/' + "out/" + "Y.npy"
Y = np.load(Y_name) 

# Determine how many unique images are in their. Here, we have 3 noisy images per image
rep_image = 3
num_pix = int(Y.shape[0] / rep_image)

# I will shuffle the elements so that I can select the first 80 % to train then test with having random images.
all_pics = np.arange(0, num_pix)
np.random.seed(1993)
np.random.shuffle(all_pics)

# Configure how many images are test train to create
train_frac = 0.6
test_frac = 0.2
validate_frac = 1 - train_frac - test_frac

# Now I will seperate the shuffled images into test, train, ect.
pics_train = all_pics[0:int(all_pics.shape[0] * train_frac)]
pics_test = all_pics[int(all_pics.shape[0] * train_frac):int(all_pics.shape[0] * (train_frac + test_frac))]
pics_validate = all_pics[int(all_pics.shape[0] * (train_frac + test_frac)):]

# So far we a talking about the real underlying images. But in the numpy array they are repeated. This means that if we want to select the true image 2 we really have to select image number 3, 4, 5 from both arrays. In the following I will do exaclyt that.
pics_train = np.repeat(pics_train, rep_image) + np.tile(np.arange(0, rep_image), pics_train.shape[0])
pics_test = np.repeat(pics_test, rep_image) + np.tile(np.arange(0, rep_image), pics_test.shape[0])
pics_validate = np.repeat(pics_validate, rep_image) + np.tile(np.arange(0, rep_image), pics_validate.shape[0])

# Now select train, test and validate and save them.
Y_train = Y[pics_train, :, :,:]
Y_name = '/home/cloud/Create_Data/' + "out/" + "Y_train.npy"
np.save(Y_name, Y_train) 
del(Y_train)
gc.collect()

Y_test = Y[pics_test, :, :,:]
Y_name = '/home/cloud/Create_Data/' + "out/" + "Y_test.npy"
np.save(Y_name, Y_test) 
del(Y_test)
gc.collect()

Y_validate = Y[pics_validate, :, :,:]
Y_name = '/home/cloud/Create_Data/' + "out/" + "Y_validate.npy"
np.save(Y_name, Y_validate) 
del(Y_validate)
gc.collect()

del Y
gc.collect()

X_name = '/home/cloud/Create_Data/' + "out/" + "X.npy"
X = np.load(X_name) 

X_train = X[pics_train, :, :,:]
X_name = '/home/cloud/Create_Data/' + "out/" + "X_train.npy"
np.save(X_name, X_train) 
del(X_train)
gc.collect()

X_test = X[pics_test, :, :,:]
X_name = '/home/cloud/Create_Data/' + "out/" + "X_test.npy"
np.save(X_name, X_test) 
del(X_test)
gc.collect()

X_validate = X[pics_validate, :, :,:]
X_name = '/home/cloud/Create_Data/' + "out/" + "X_validate.npy"
np.save(X_name, X_validate) 
del(X_validate)
gc.collect()

