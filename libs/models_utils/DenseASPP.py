import tensorflow as tf
from tensorflow.contrib import slim
from libs.builders import frontend_builder


def Upsampling(inputs,scale):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*scale,  tf.shape(inputs)[2]*scale])


def DilatedConvBlock(inputs, n_filters, rate=1, kernel_size=[3, 3]):
    """
    Basic dilated conv block 
    Apply successivly BatchNormalization, ReLU nonlinearity, dilated convolution 
    """
    net = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    net = slim.conv2d(net, n_filters, kernel_size, rate=rate, activation_fn=None, normalizer_fn=None)
    return net


def build_dense_aspp(inputs, num_classes, preset_model='DenseASPP', frontend="ResNet101", load_pre_train=False,
                    pre_trained_dir=None, is_training=True):
    """
       Builds the Dense ASPP model.

       Arguments:
         inputs: The input tensor
         num_classes: Number of classes

         preset_model:
         frontend: Which frontend model you want to use. Select which ResNet model to use for feature extraction
         load_pre_train: Indicate whether to load pre-trained models
         pre_trained_dir: Path to directory where pre-trained model is to be loaded from.
         is_training: Training Flag for batch normalization

       Returns:
         Dense ASPP model
       """

    logits, end_points, frontend_scope, init_fn, stable_params = \
        frontend_builder.build_frontend(inputs, frontend, pre_trained_dir=pre_trained_dir, is_training=is_training,
                                        load_pre_train=load_pre_train, requested_stages=[3])

    init_features = end_points['stage3']

    # First block, rate = 3
    with tf.variable_scope('Atrous_conv_d3'):
        d_3_features = DilatedConvBlock(init_features, n_filters=256, kernel_size=[1, 1])
        d_3 = DilatedConvBlock(d_3_features, n_filters=64, rate=3, kernel_size=[3, 3])

    with tf.variable_scope('concat_1'):
        d_4 = tf.concat([init_features, d_3], axis=-1)
    # Second block, rate = 6
    with tf.variable_scope('Atrous_conv_d6'):
        d_4 = DilatedConvBlock(d_4, n_filters=256, kernel_size=[1, 1])
        d_4 = DilatedConvBlock(d_4, n_filters=64, rate=6, kernel_size=[3, 3])

    with tf.variable_scope('concat_2'):
        d_5 = tf.concat([init_features, d_3, d_4], axis=-1)
    # Third block, rate = 12
    with tf.variable_scope('Atrous_conv_d12'):
        d_5 = DilatedConvBlock(d_5, n_filters=256, kernel_size=[1, 1])
        d_5 = DilatedConvBlock(d_5, n_filters=64, rate=12, kernel_size=[3, 3])

    with tf.variable_scope('concat_3'):
        d_6 = tf.concat([init_features, d_3, d_4, d_5], axis=-1)
    # Fourth block, rate = 18
    with tf.variable_scope('Atrous_conv_d18'):
        d_6 = DilatedConvBlock(d_6, n_filters=256, kernel_size=[1, 1])
        d_6 = DilatedConvBlock(d_6, n_filters=64, rate=18, kernel_size=[3, 3])

    with tf.variable_scope('concat_4'):
        d_7 = tf.concat([init_features, d_3, d_4, d_5, d_6], axis=-1)
    # Fifth block, rate = 24
    with tf.variable_scope('Atrous_conv_d24'):
        d_7 = DilatedConvBlock(d_7, n_filters=256, kernel_size=[1, 1])
        d_7 = DilatedConvBlock(d_7, n_filters=64, rate=24, kernel_size=[3, 3])

    full_block = tf.concat([init_features, d_3, d_4, d_5, d_6, d_7], axis=-1)
    
    net = slim.conv2d(full_block, num_classes, [1, 1], activation_fn=None, scope='logits')

    with tf.variable_scope('Upsampling'):
        net = Upsampling(net, scale=8)

    return net, init_fn, stable_params
