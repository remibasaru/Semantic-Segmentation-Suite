import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
from libs.builders import frontend_builder
from libs.builders.frontend_builder import FRONT_END_MODEL_PATH


def Upsampling(inputs, feature_map_shape):
    return tf.image.resize_bilinear(inputs, size=feature_map_shape)


def ConvUpscaleBlock(inputs, n_filters, kernel_size=[3, 3], scale=2):
    """
    Basic conv transpose block for Encoder-Decoder upsampling
    Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
    """
    net = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    net = slim.conv2d_transpose(net, n_filters, kernel_size=[3, 3], stride=[scale, scale], activation_fn=None)
    return net


def ConvBlock(inputs, n_filters, kernel_size=[3, 3]):
    """
    Basic conv block for Encoder-Decoder
    Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
    """
    net = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    net = slim.conv2d(net, n_filters, kernel_size, activation_fn=None, normalizer_fn=None)
    return net


def InterpBlock(net, level, feature_map_shape, pooling_type):
    
    # Compute the kernel and stride sizes according to how large the final feature map will be
    # When the kernel size and strides are equal, then we can compute the final feature map size
    # by simply dividing the current size by the kernel or stride size
    # The final feature map sizes are 1x1, 2x2, 3x3, and 6x6. We round to the closest integer
    kernel_size = [int(np.round(float(feature_map_shape[0]) / float(level))), int(np.round(float(feature_map_shape[1]) / float(level)))]
    stride_size = kernel_size

    net = slim.pool(net, kernel_size, stride=stride_size, pooling_type='MAX')
    net = slim.conv2d(net, 512, [1, 1], activation_fn=None)
    net = slim.batch_norm(net, fused=True)
    net = tf.nn.relu(net)
    net = Upsampling(net, feature_map_shape)
    return net


def PyramidPoolingModule(inputs, feature_map_shape, pooling_type):
    """
    Build the Pyramid Pooling Module.
    """

    interp_block1 = InterpBlock(inputs, 1, feature_map_shape, pooling_type)
    interp_block2 = InterpBlock(inputs, 2, feature_map_shape, pooling_type)
    interp_block3 = InterpBlock(inputs, 3, feature_map_shape, pooling_type)
    interp_block6 = InterpBlock(inputs, 6, feature_map_shape, pooling_type)

    res = tf.concat([inputs, interp_block6, interp_block3, interp_block2, interp_block1], axis=-1)
    return res


def build_pspnet(inputs, num_classes, label_size, upscaling_method="conv", pooling_type="MAX", preset_model=None,
                 frontend="ResNet101", load_pre_train=False, pre_trained_dir=None, is_training=True):
    """
    Builds the PSPNet model. 

    Arguments:
      inputs: The input tensor
      num_classes: Number of classes

      label_size: Size of the final label tensor. We need to know this for proper upscaling
      upscaling_method:
      pooling_type: Max or Average pooling

      preset_model:
      frontend: Which frontend model you want to use. Select which ResNet model to use for feature extraction
      load_pre_train: Indicate whether to load pre-trained models
      pre_trained_dir: Path to directory where pre-trained model is to be loaded from.
      is_training: Training Flag for batch normalization

    Returns:
      PSPNet model
    """

    logits, end_points, frontend_scope, init_fn, stable_params = \
        frontend_builder.build_frontend(inputs, frontend, pre_trained_dir=pre_trained_dir, is_training=is_training,
                                        load_pre_train=load_pre_train, requested_stages=[3])

    feature_map_shape = [int(x / 8.0) for x in label_size]
    print(feature_map_shape)
    with tf.variable_scope('PyramidPoolingModule'):
        psp = PyramidPoolingModule(end_points['stage3'], feature_map_shape=feature_map_shape, pooling_type=pooling_type)

    with tf.variable_scope('Conv'):
        net = slim.conv2d(psp, 512, [3, 3], activation_fn=None)
        net = slim.batch_norm(net, fused=True)
        net = tf.nn.relu(net)
    with tf.variable_scope('Upscaling'):
        if upscaling_method.lower() == "conv":
            with tf.variable_scope('ConvUpscaleBlock_1'):
                net = ConvUpscaleBlock(net, 256, kernel_size=[3, 3], scale=2)
            net = ConvBlock(net, 256)
            with tf.variable_scope('ConvUpscaleBlock_2'):
                net = ConvUpscaleBlock(net, 128, kernel_size=[3, 3], scale=2)
            net = ConvBlock(net, 128)
            with tf.variable_scope('ConvUpscaleBlock_3'):
                net = ConvUpscaleBlock(net, 64, kernel_size=[3, 3], scale=2)
            net = ConvBlock(net, 64)
        elif upscaling_method.lower() == "bilinear":
            net = Upsampling(net, label_size)
    
    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')

    return net, init_fn, stable_params


def mean_image_subtraction(inputs, means=[123.68, 116.78, 103.94]):
    inputs = tf.to_float(inputs)
    num_channels = inputs.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=inputs)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)