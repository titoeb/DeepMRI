#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# In this script I will load the dataset, make them from RGB to grayscale (remove the last dimension) and store them by overwritting the old files.

# First create a list containing all files to loop around.
name_list = []
for i in range(1,21):
    name_list.append("P" + str(i) + "_X.npy")
    name_list.append("P" + str(i) + "_Y.npy")


# Now iterate over the list containing all names pics, load them, select only the first layer of the RGB colors (all have the same values anyway), normalize the pic, make it float32 and write it out again!
for cur_name in name_list:
    Cur_pic = np.load(cur_name)
    Cur_pic = Cur_pic[:,:,:,0]
    Cur_pic.reshape([4055, 256, 256, 1])
    Cur_pic = Cur_pic / 255
    Cur_pic = Cur_pic.astype('float32')
    np.save(cur_name, Cur_pic)

