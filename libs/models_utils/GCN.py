import tensorflow as tf
from tensorflow.contrib import slim
from libs.builders import frontend_builder
from libs.builders.frontend_builder import FRONT_END_MODEL_PATH


def Upsampling(inputs,scale):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*scale,  tf.shape(inputs)[2]*scale])


def ConvUpscaleBlock(inputs, n_filters, kernel_size=[3, 3], scale=2):
    """
    Basic deconv block for GCN
    Apply Transposed Convolution for feature map upscaling
    """
    net = slim.conv2d_transpose(inputs, n_filters, kernel_size=[3, 3], stride=[2, 2], activation_fn=None)
    return net


def BoundaryRefinementBlock(inputs, n_filters, kernel_size=[3, 3]):
    """
    Boundary Refinement Block for GCN
    """
    net = slim.conv2d(inputs, n_filters, kernel_size, activation_fn=None, normalizer_fn=None)
    net = tf.nn.relu(net)
    net = slim.conv2d(net, n_filters, kernel_size, activation_fn=None, normalizer_fn=None)
    net = tf.add(inputs, net)
    return net


def GlobalConvBlock(inputs, n_filters=21, size=3):
    """
    Global Conv Block for GCN
    """

    net_1 = slim.conv2d(inputs, n_filters, [size, 1], activation_fn=None, normalizer_fn=None)
    net_1 = slim.conv2d(net_1, n_filters, [1, size], activation_fn=None, normalizer_fn=None)

    net_2 = slim.conv2d(inputs, n_filters, [1, size], activation_fn=None, normalizer_fn=None)
    net_2 = slim.conv2d(net_2, n_filters, [size, 1], activation_fn=None, normalizer_fn=None)

    net = tf.add(net_1, net_2)

    return net


def build_gcn(inputs, num_classes, preset_model='GCN', frontend="ResNet101", load_pre_train=False,
                    pre_trained_dir=None, is_training=True):
    """
    Builds the GCN model. 

    Arguments:
      inputs: The input tensor
      num_classes: Number of classes

      preset_model:
      frontend: Which frontend model you want to use. Select which ResNet model to use for feature extraction
      load_pre_train: Indicate whether to load pre-trained models
      pre_trained_dir: Path to directory where pre-trained model is to be loaded from.
      is_training: Training Flag for batch normalization

    Returns:
      GCN model
    """

    logits, end_points, frontend_scope, init_fn, stable_params = \
        frontend_builder.build_frontend(inputs, frontend, pre_trained_dir=pre_trained_dir, is_training=is_training,
                                        load_pre_train=load_pre_train, requested_stages=[2, 3, 4, 7])

    res = [end_points['stage7'], end_points['stage4'], end_points['stage3'], end_points['stage2']]

    with tf.variable_scope('GCN_1'):
        down_5 = GlobalConvBlock(res[0], n_filters=21, size=3)
    with tf.variable_scope('BR_1'):
        down_5 = BoundaryRefinementBlock(down_5, n_filters=21, kernel_size=[3, 3])
    with tf.variable_scope('Deconv_1'):
        down_5 = ConvUpscaleBlock(down_5, n_filters=21, kernel_size=[3, 3], scale=2)

    with tf.variable_scope('GCN_2'):
        down_4 = GlobalConvBlock(res[1], n_filters=21, size=3)
    with tf.variable_scope('BR2'):
        down_4 = BoundaryRefinementBlock(down_4, n_filters=21, kernel_size=[3, 3])
    with tf.variable_scope('Add_1'):
        down_4 = tf.add(down_4, down_5)
    with tf.variable_scope('BR2_1'):
        down_4 = BoundaryRefinementBlock(down_4, n_filters=21, kernel_size=[3, 3])
    down_4 = ConvUpscaleBlock(down_4, n_filters=21, kernel_size=[3, 3], scale=2)

    with tf.variable_scope('GCN_3'):
        down_3 = GlobalConvBlock(res[2], n_filters=21, size=3)
    with tf.variable_scope('BR_3'):
        down_3 = BoundaryRefinementBlock(down_3, n_filters=21, kernel_size=[3, 3])
    with tf.variable_scope('Add_2'):
        down_3 = tf.add(down_3, down_4)
    with tf.variable_scope('BR3_1'):
        down_3 = BoundaryRefinementBlock(down_3, n_filters=21, kernel_size=[3, 3])
    down_3 = ConvUpscaleBlock(down_3, n_filters=21, kernel_size=[3, 3], scale=2)

    with tf.variable_scope('GCN_4'):
        down_2 = GlobalConvBlock(res[3], n_filters=21, size=3)
    with tf.variable_scope('BR_4'):
        down_2 = BoundaryRefinementBlock(down_2, n_filters=21, kernel_size=[3, 3])
    with tf.variable_scope('Add_3'):
        down_2 = tf.add(down_2, down_3)
    with tf.variable_scope('BR4_1'):
        down_2 = BoundaryRefinementBlock(down_2, n_filters=21, kernel_size=[3, 3])
    down_2 = ConvUpscaleBlock(down_2, n_filters=21, kernel_size=[3, 3], scale=2)

    net = BoundaryRefinementBlock(down_2, n_filters=21, kernel_size=[3, 3])
    net = ConvUpscaleBlock(net, n_filters=21, kernel_size=[3, 3], scale=2)
    net = BoundaryRefinementBlock(net, n_filters=21, kernel_size=[3, 3])

    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')

    return net, init_fn, stable_params


def mean_image_subtraction(inputs, means=[123.68, 116.78, 103.94]):
    inputs=tf.to_float(inputs)
    num_channels = inputs.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=inputs)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)
