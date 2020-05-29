# coding=utf-8

import tensorflow as tf
from tensorflow.contrib import slim
from libs.builders import frontend_builder
from libs.builders.frontend_builder import FRONT_END_MODEL_PATH


def Upsampling(inputs,scale):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*scale,  tf.shape(inputs)[2]*scale])


def ConvUpscaleBlock(inputs, n_filters, kernel_size=[3, 3], scale=2):
    """
    Basic conv transpose block for Encoder-Decoder upsampling
    Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
    """
    net = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    net = slim.conv2d_transpose(net, n_filters, kernel_size=[3, 3], stride=[scale, scale], activation_fn=None)
    return net


def ConvBlock(inputs, n_filters, kernel_size=[3, 3], strides=1):
    """
    Basic conv block for Encoder-Decoder
    Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
    """
    net = slim.conv2d(inputs, n_filters, kernel_size, stride=[strides, strides], activation_fn=None, normalizer_fn=None)
    net = tf.nn.relu(slim.batch_norm(net, fused=True))
    return net


def AttentionRefinementModule(inputs, n_filters):

    # Global average pooling
    net = tf.reduce_mean(inputs, [1, 2], keepdims=True)
    n_filters = inputs.shape[-1]
    net = slim.conv2d(net, n_filters, kernel_size=[1, 1])
    net = slim.batch_norm(net, fused=True)
    net = tf.sigmoid(net)

    net = tf.multiply(inputs, net)

    return net


def FeatureFusionModule(input_1, input_2, n_filters):
    inputs = tf.concat([input_1, input_2], axis=-1)
    inputs = ConvBlock(inputs, n_filters=n_filters, kernel_size=[3, 3])

    # Global average pooling
    net = tf.reduce_mean(inputs, [1, 2], keepdims=True)
    
    net = slim.conv2d(net, n_filters, kernel_size=[1, 1])
    net = tf.nn.relu(net)
    net = slim.conv2d(net, n_filters, kernel_size=[1, 1])
    net = tf.sigmoid(net)

    net = tf.multiply(inputs, net)

    net = tf.add(inputs, net)

    return net


def build_bisenet(inputs, num_classes=2, preset_model='BiSeNet', frontend="ResNet101", load_pre_train=False,
                    pre_trained_dir=None, is_training=True):
    """
    Builds the BiSeNet model. 

    Arguments:
      inputs: The input tensor
      num_classes: Number of classes
      preset_model:
      frontend: Which frontend model you want to use. Select which ResNet model to use for feature extraction
      load_pre_train: Indicate whether to load pre-trained models
      pre_trained_dir: Path to directory where pre-trained model is to be loaded from.
      is_training: Training Flag for batch normalization

    Returns:
      BiSeNet model
    """

    # The spatial path
    # The number of feature maps for each convolution is not specified in the paper
    # It was chosen here to be equal to the number of feature maps of a classification
    # model at each corresponding stage
    with tf.variable_scope('Spatial_path'):
        spatial_net = ConvBlock(inputs, n_filters=64, kernel_size=[3, 3], strides=2)
        spatial_net = ConvBlock(spatial_net, n_filters=128, kernel_size=[3, 3], strides=2)
        spatial_net = ConvBlock(spatial_net, n_filters=256, kernel_size=[3, 3], strides=2)

    # Context path
    logits, end_points, frontend_scope, init_fn, stable_params = \
        frontend_builder.build_frontend(inputs, frontend, pre_trained_dir=pre_trained_dir, is_training=is_training,
                                        load_pre_train=load_pre_train, requested_stages=[4, 7])

    with tf.variable_scope('Context_path'):
        net_4 = AttentionRefinementModule(end_points['stage4'], n_filters=512)

        net_5 = AttentionRefinementModule(end_points['stage7'], n_filters=2048)

        global_channels = tf.reduce_mean(net_5, [1, 2], keepdims=True)
        net_5_scaled = tf.multiply(global_channels, net_5)

    # Combining the paths
    with tf.variable_scope('Upsampling_1'):
        net_4 = Upsampling(net_4, scale=2)
    with tf.variable_scope('Upsampling_2'):
        net_5_scaled = Upsampling(net_5_scaled, scale=4)

    context_net = tf.concat([net_4, net_5_scaled], axis=-1)

    with tf.variable_scope('FeatureFusionModule'):
        net = FeatureFusionModule(input_1=spatial_net, input_2=context_net, n_filters=num_classes)

    # Final upscaling and finish
    with tf.variable_scope('Upsampling_3'):
        net = Upsampling(net, scale=8)
    
    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')

    return net, init_fn, stable_params

