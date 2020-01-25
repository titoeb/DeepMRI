#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import gc

# Load the whole image
Y_name = "/scratch2/ttoebro/data/Y_spulen_pois.npy"
Y = np.load(Y_name) 

# Determine how many unique images are in their. Here, we have 3 noisy images per image
Y.shape


# We have a total of 13900 unique images. But for the spulen data I was only able to create every third image because of hard drive constraints (otherwise the data would we about 120 GB).
num_pix = 13900


# I will shuffle the elements so that I can select the first 80 % to train then test with having random images.
all_pics = np.arange(0, num_pix)
np.random.seed(1993)
np.random.shuffle(all_pics)


# Configure how many images are test train to create
train_frac = 0.6
test_frac = 0.2
validate_frac = 1 - train_frac - test_frac


# Now I will seperate the shuffled images into test, train, ect.
pics_train_tmp = all_pics[0:int(all_pics.shape[0] * train_frac)]
pics_test_tmp = all_pics[int(all_pics.shape[0] * train_frac):int(all_pics.shape[0] * (train_frac + test_frac))]
pics_validate_tmp = all_pics[int(all_pics.shape[0] * (train_frac + test_frac)):]

# So far we a talking about the real underlying images. But for the spulen data due to memory / hard drive constraints I could only use every third image of the slices. I used for each image only slice 0, 3, 6 ...
x_start = 15
x_end = 300
y_start = 50
y_end = 280
z_start = 40
z_end = 220

# Initialize lists # 
names_y = list()
names_x = list()
count_original = 0
count_spulen = 0
num_original = list()
num_spulen = list()

for pic in range(1, 21):
    loc_counter = 0
    for x in range(x_start, x_end, 1):
        if(loc_counter % 3 == 0):
            num_spulen.append(count_spulen)
            count_spulen += 1
        else:
            num_spulen.append(-1)
        num_original.append(count_original)
        count_original += 1
        loc_counter += 1
    loc_counter = 0
    for y in range(y_start, y_end, 1): 
        if(loc_counter % 3 == 0):
            num_spulen.append(count_spulen)
            count_spulen += 1
        else:
            num_spulen.append(-1)
        num_original.append(count_original)
        count_original += 1
        loc_counter += 1
    loc_counter = 0
    for z in range(z_start, z_end, 1):
        if(loc_counter % 3 == 0):
            num_spulen.append(count_spulen)
            count_spulen += 1
        else:
            num_spulen.append(-1)
        num_original.append(count_original)
        count_original += 1
        loc_counter += 1

conv_dict = dict(zip(num_original, num_spulen))

# The dic above gives you for each image in the original dataset the corresponding element in the spulen dataset or -1 if that picture was not used. From that I will convert the pics_train ect to the true images.
pics_train = list()
pics_test = list()
pics_validate = list()
for pic in pics_train_tmp:
    if(conv_dict[pic] != -1):
        pics_train.append(conv_dict[pic])
for pic in pics_test_tmp:
    if(conv_dict[pic] != -1):
        pics_test.append(conv_dict[pic])
for pic in pics_validate_tmp:
    if(conv_dict[pic] != -1):
        pics_validate.append(conv_dict[pic])
        
pics_train = np.array(pics_train)
pics_test = np.array(pics_test)
pics_validate = np.array(pics_validate)

# Now select train, test and validate and save them.
Y_train = Y[pics_train, :, :,:]
Y_name = "/scratch2/ttoebro/data/"  + "Y_train_spulen_pois.npy"
np.save(Y_name, Y_train) 
del(Y_train)
gc.collect()

# In[10]:
Y_test = Y[pics_test, :, :,:]
Y_name = "/scratch2/ttoebro/data/" + "Y_test_spulen_pois.npy"
np.save(Y_name, Y_test) 
del(Y_test)
gc.collect()

Y_validate = Y[pics_validate, :, :,:]
Y_name = "/scratch2/ttoebro/data/"  + "Y_validate_spulen_pois.npy"
np.save(Y_name, Y_validate) 
del(Y_validate)
gc.collect()

del Y
gc.collect()

X_name = "/scratch2/ttoebro/data/X_spulen_pois.npy"
X = np.load(X_name) 

X_train = X[pics_train, :, :,:]
X_name = "/scratch2/ttoebro/data/" + "X_train_spulen_pois.npy"
np.save(X_name, X_train) 
del(X_train)
gc.collect()

X_test = X[pics_test, :, :,:]
X_name = "/scratch2/ttoebro/data/"  + "X_test_spulen_pois.npy"
np.save(X_name, X_test) 
del(X_test)
gc.collect()

X_validate = X[pics_validate, :, :,:]
X_name = "/scratch2/ttoebro/data/"  + "X_validate_spulen_pois.npy"
np.save(X_name, X_validate) 
del(X_validate)
gc.collect()
