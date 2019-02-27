import datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers

# (MODEL) ----------------------------------------------------------------------
def DnCNN_model_fn (features, labels, mode):
    """ Beyond a Gaussian Denoiser: Residual learning
        of Deep CNN for Image Denoising.
        Residual learning (originally with X = True + Noisy, Y = noisy)
        :depth: total number of conv-layers in network
        :filters: number of filters for every convolution
        :kernelsize:
    """

    # LAYERS -----------------------------
    # workaround, due to tf 1.11.0 rigid argument checking in Estimator
    depth = 10
    filters = 20
    kernelsize = 5

    # (0) Input layer. (batch)-1*256*256*1(channels)
    input_layer = features  # ['X']

    # (i) Conv + Relu. Filters: 64, 3x3x1, same padding
    conv_first = tf.contrib.layers.conv2d(
        inputs=input_layer,
        num_outputs=filters,
        kernel_size=kernelsize,
        padding='SAME',
        stride=1,
        activation_fn=tf.nn.relu,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=initializers.xavier_initializer(),
        weights_regularizer=None,
        biases_initializer=tf.zeros_initializer(),
        biases_regularizer=None,
        trainable=True,
        scope=None
    )

    # (ii) Conv + BN + Relu. Filters: 64, 3x3x64,  same padding
    # 17 or 20 of these layers
    conv_bn = tf.contrib.layers.repeat(
        inputs=conv_first, repetitions=depth - 2,
        layer=tf.contrib.layers.conv2d,
        num_outputs=filters, padding='SAME', stride=1,
        kernel_size=[kernelsize, kernelsize],
        activation_fn=tf.nn.relu,
        normalizer_fn=tf.layers.batch_normalization,
        normalizer_params={'momentum': 0.99, 'epsilon': 0.001,
                           'trainable': False,
                           'training': mode == tf.estimator.ModeKeys.TRAIN},
        scope='conv2'
    )

    # (iii) Conv. 3x3x64, same padding
    conv_last = tf.contrib.layers.conv2d(
        inputs=conv_bn,
        num_outputs=1,
        kernel_size=kernelsize,
        padding='SAME',
        stride=1,
        activation_fn=tf.nn.relu,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=initializers.xavier_initializer(),
        weights_regularizer=None,
        biases_initializer=tf.zeros_initializer(),
        biases_regularizer=None,
        trainable=True,
        scope=None
    )

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=conv_last + input_layer)

    # LOSSES -----------------------------

    # (MSE Version)
    loss = tf.losses.mean_squared_error(
        labels=labels,
        predictions=conv_last + input_layer)  # learning difference only

    # (L1 Version)
    # loss = tf.losses.absolute_difference(
    #    labels=labels,
    #    predictions=conv_last + input_layer)

    # (SSIM Version)
    # loss = tf.image.ssim(conv_last + input_layer, labels, max_val = 1)

    # (MS-SSIM Version)
    # loss = tf.image.ssim_multiscale(
    #    img1=labels,
    #    img2=conv_last + input_layer,
    #    max_val=1)

    tf.summary.scalar("Value_Loss_Function", loss)

    # For both TRAIN & EVAL:
    # TENSORBOARD ------------------------
    for var in tf.trainable_variables():
        # write out all variables
        name = var.name
        name = name.replace(':', '_')
        tf.summary.histogram(name, var)
    merged_summary = tf.summary.merge_all()

    # RUN SPECIFICATIONS -----------------
    if mode == tf.estimator.ModeKeys.TRAIN:
        # BATCHNROM 'memoize'
        # look at tf.contrib.layers.batch_norm doc.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            original_optimizer = tf.train.AdamOptimizer(
                learning_rate=0.035)
            optimizer = tf.contrib.estimator.clip_gradients_by_norm(
                original_optimizer,
                clip_norm=5.0)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            eval_metric_ops={
                "accuracy": tf.metrics.mean_absolute_error(
                    labels=labels,
                    predictions=conv_last + input_layer)
            }
        )

# (ESTIMATOR) ------------------------------------------------------------------
root = '/scratch2/ttoebro/'
d = datetime.datetime.now()

DnCNN = tf.estimator.Estimator(
    model_fn=DnCNN_model_fn,
    # Estimators automatically save and restore variables in the model_dir.
    model_dir=root + 'model/' +
              "DnCNN_MSE_{}_{}_{}".format(d.month, d.day, d.hour),
    config=tf.estimator.RunConfig(save_summary_steps=10,
                                  log_step_count_steps=10) #  frequency,
)


# (DATA) -----------------------------------------------------------------------

train_data = np.load('/scratch2/ttoebro/data/X_train.npy')
train_label = np.load('/scratch2/ttoebro/data/Y_train.npy')

# (TRAINING) -------------------------------------------------------------------
# learning with mini batch (128 images), 50 epochs
# rewrite with tf.placeholder, session.run
# https://stackoverflow.com/questions/49743838/predict-single-image-after-training-model-in-tensorflow
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x=train_data,
    y=train_label,
    batch_size=1,
    num_epochs=50,
    shuffle=True)

DnCNN.train(input_fn=train_input_fn)
