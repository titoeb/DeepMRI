import numpy as np
import gc
# Import Data 

# Import X
X_1 = np.load('/scratch2/ttoebro/data/P1_X.npy')
X_2 = np.load('/scratch2/ttoebro/data/P2_X.npy')
X_3 = np.load('/scratch2/ttoebro/data/P3_X.npy')
X_4 = np.load('/scratch2/ttoebro/data/P4_X.npy')
X_5 = np.load('/scratch2/ttoebro/data/P5_X.npy')
X_6 = np.load('/scratch2/ttoebro/data/P6_X.npy')
X_7 = np.load('/scratch2/ttoebro/data/P7_X.npy')
X_8 = np.load('/scratch2/ttoebro/data/P8_X.npy')
X_9 = np.load('/scratch2/ttoebro/data/P9_X.npy')
X_10 = np.load('/scratch2/ttoebro/data/P10_X.npy')
X_11 = np.load('/scratch2/ttoebro/data/P11_X.npy')
X_12 = np.load('/scratch2/ttoebro/data/P12_X.npy')
X_13 = np.load('/scratch2/ttoebro/data/P13_X.npy')
X_14 = np.load('/scratch2/ttoebro/data/P14_X.npy')
X_15 = np.load('/scratch2/ttoebro/data/P15_X.npy')
X_16 = np.load('/scratch2/ttoebro/data/P16_X.npy')
X_17 = np.load('/scratch2/ttoebro/data/P17_X.npy')
X_18 = np.load('/scratch2/ttoebro/data/P18_X.npy')
X_19 = np.load('/scratch2/ttoebro/data/P19_X.npy')
X_20 = np.load('/scratch2/ttoebro/data/P20_X.npy')
X = np.concatenate(seq = (X_1, X_2, X_3, X_4, X_5, X_6,X_7,X_8,X_9,X_10, X_11, X_12,X_13, X_14, X_15, X_16, X_17, X_18, X_19, X_20), axis=0)
del X_1, X_2, X_3, X_4, X_5, X_6,X_7,X_8,X_9,X_10, X_11, X_12,X_13, X_14, X_15, X_16, X_17, X_18, X_19, X_20
X = X.reshape([X.shape[0], 256, 256, 1])
gc.collect()

#Import Y
Y_1 = np.load('/scratch2/ttoebro/data/P1_Y.npy')
Y_2 = np.load('/scratch2/ttoebro/data/P2_Y.npy')
Y_3 = np.load('/scratch2/ttoebro/data/P3_Y.npy')
Y_4 = np.load('/scratch2/ttoebro/data/P4_Y.npy')
Y_5 = np.load('/scratch2/ttoebro/data/P5_Y.npy')
Y_6 = np.load('/scratch2/ttoebro/data/P6_Y.npy')
Y_7 = np.load('/scratch2/ttoebro/data/P7_Y.npy')
Y_8 = np.load('/scratch2/ttoebro/data/P8_Y.npy')
Y_9 = np.load('/scratch2/ttoebro/data/P9_Y.npy')
Y_10 = np.load('/scratch2/ttoebro/data/P10_Y.npy')
Y_11 = np.load('/scratch2/ttoebro/data/P11_Y.npy')
Y_12 = np.load('/scratch2/ttoebro/data/P12_Y.npy')
Y_13 = np.load('/scratch2/ttoebro/data/P13_Y.npy')
Y_14 = np.load('/scratch2/ttoebro/data/P14_Y.npy')
Y_15 = np.load('/scratch2/ttoebro/data/P15_Y.npy')
Y_16 = np.load('/scratch2/ttoebro/data/P16_Y.npy')
Y_17 = np.load('/scratch2/ttoebro/data/P17_Y.npy')
Y_18 = np.load('/scratch2/ttoebro/data/P18_Y.npy')
Y_19 = np.load('/scratch2/ttoebro/data/P19_Y.npy')
Y_20 = np.load('/scratch2/ttoebro/data/P20_Y.npy')
Y = np.concatenate(seq = (Y_1, Y_2, Y_3, Y_4, Y_5, Y_6,Y_7,Y_8,Y_9,Y_10, Y_11, Y_12,Y_13, Y_14, Y_15, Y_16, Y_17, Y_18, Y_19, Y_20), axis=0)
del Y_1, Y_2, Y_3, Y_4, Y_5, Y_6,Y_7,Y_8,Y_9,Y_10, Y_11, Y_12,Y_13, Y_14, Y_15, Y_16, Y_17, Y_18, Y_19, Y_20
Y = Y.reshape([Y.shape[0], 256, 256, 1])


# Shuffle the arrays!
shuff_index = np.arange(0,Y.shape[0])
np.random.shuffle(shuff_index)
X = X[shuff_index,:,:,:]
Y = Y[shuff_index,:,:,:]

# Create Test, Train and Validation set
train_frac = 0.6
train_valid = 0.2
train_index = int(train_frac * Y.shape[0])
valid_index = int((train_valid + train_frac)* Y.shape[0])
X_train = X[0:train_index,:,:,:]
X_validation = X[train_index:valid_index,:,:,:]
X_test = X[valid_index:X.shape[0],:,:,:]

Y_train = Y[0:train_index,:,:,:]
Y_validation = Y[train_index:valid_index,:,:,:]
Y_test = Y[valid_index:X.shape[0],:,:,:]

# Write the arrays out
np.save('/scratch2/ttoebro/data/X_train.npy', X_train)
np.save('/scratch2/ttoebro/data/X_validation.npy', X_validation)
np.save('/scratch2/ttoebro/data/X_test.npy', X_test)

np.save('/scratch2/ttoebro/data/Y_train.npy', Y_train)
np.save('/scratch2/ttoebro/data/Y_validation.npy', Y_validation)
np.save('/scratch2/ttoebro/data/Y_test.npy', Y_test)
