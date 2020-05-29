import tensorflow as tf
from tensorflow.contrib import slim


def Upsampling(inputs, scale, spatial_shape=None):
    if spatial_shape is not None:
        upsampled_image = tf.image.resize_nearest_neighbor(inputs,
                                                           size=[spatial_shape[0], spatial_shape[1]])
    else:
        upsampled_image = tf.image.resize_nearest_neighbor(inputs,
                                                           size=[tf.shape(inputs)[1]*scale,  tf.shape(inputs)[2]*scale])

    return upsampled_image


def Unpooling(inputs, scale, spatial_shape=None):
    with tf.variable_scope('Unpooling'):
        if spatial_shape is None:
            return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*scale,  tf.shape(inputs)[2]*scale])
        else:
            return tf.image.resize_bilinear(inputs, size=[spatial_shape[0], spatial_shape[1]])


def ResidualUnit(inputs, n_filters=48, filter_size=3):
    """
    A local residual unit

    Arguments:
      inputs: The input tensor
      n_filters: Number of output feature maps for each conv
      filter_size: Size of convolution kernel

    Returns:
      Output of local residual block
    """

    net = slim.conv2d(inputs, n_filters, filter_size, activation_fn=None)
    net = slim.batch_norm(net, fused=True)
    net = tf.nn.relu(net)
    net = slim.conv2d(net, n_filters, filter_size, activation_fn=None)
    net = slim.batch_norm(net, fused=True)
    # net = tf.add(net, inputs)
    return net


def FullResolutionResidualUnit(pool_stream, res_stream, n_filters_3, n_filters_1, pool_scale):
    """
    A full resolution residual unit

    Arguments:
      pool_stream: The inputs from the pooling stream
      res_stream: The inputs from the residual stream
      n_filters_3: Number of output feature maps for each 3x3 conv
      n_filters_1: Number of output feature maps for each 1x1 conv
      pool_scale: scale of the pooling layer i.e window size and stride

    Returns:
      Output of full resolution residual block
    """
    pool = slim.pool(res_stream, [pool_scale, pool_scale], stride=[pool_scale, pool_scale], pooling_type='MAX')
    # pool = tf.image.resize_image_with_crop_or_pad(pool, tf.shape(pool_stream)[1], tf.shape(pool_stream)[2])
    G = tf.concat([pool_stream, pool], axis=-1)
    # First convolution layer
    net = slim.conv2d(G, n_filters_3, kernel_size=3, activation_fn=None)
    net = slim.batch_norm(net, fused=True)
    net = tf.nn.relu(net)

    # Second convolution layer
    net = slim.conv2d(net, n_filters_3, kernel_size=3, activation_fn=None)
    net = slim.batch_norm(net, fused=True)
    pool_stream_out = tf.nn.relu(net)

    # Conv1x1 and Upsample
    net = slim.conv2d(pool_stream_out, n_filters_1, kernel_size=1, activation_fn=None)
    res_stream_shape = tf.shape(res_stream)
    res_stream_out = Upsampling(net, scale=pool_scale, spatial_shape=(res_stream_shape[1], res_stream_shape[2]))

    # # Add to form next residual stream
    # res_stream_out = tf.add(res_stream, net)

    return pool_stream_out, res_stream_out


def build_frrn(inputs, train_flag, num_classes, preset_model='FRRN-A'):
    """
    Builds the Full Resolution Residual Networks model. 

    Arguments:
      inputs: The input tensor
      preset_model: Which model you want to use. Select FRRN-A or FRRN-B
      num_classes: Number of classes

    Returns:
      FRRN model
    """

    if preset_model == 'FRRN-A':

        filters_per_down_layer = [96, 192, 384, 384]  # Number of filters in conv layer during unpooling
        num_frru_per_down_layer = [3, 4, 2, 2]

        filters_per_up_layer = [192, 192, 96]  # Number of filters in conv layer during max pooling
        num_frru_per_up_layer = [2, 2, 2]

    elif preset_model == 'FRRN-B':

        filters_per_down_layer = [96, 192, 384, 384, 384]  # Number of filters in conv layer during unpooling
        num_frru_per_down_layer = [3, 4, 2, 2, 2]

        filters_per_up_layer = [192, 192, 192, 96]  # Number of filters in conv layer during max pooling
        num_frru_per_up_layer = [2, 2, 2, 2]
    else:
        raise ValueError("Unsupported FRRN model '%s'. This function only supports FRRN-A and FRRN-B" % (preset_model))

    # filters_per_up_layer = [int(x / 2) for x in filters_per_up_layer]
    # filters_per_down_layer = [int(x / 2) for x in filters_per_down_layer]
    #####################
    # Initial Stage
    #####################
    net = slim.conv2d(inputs, 48, kernel_size=5, activation_fn=None)
    net = slim.batch_norm(net, fused=True)
    net = tf.nn.relu(net)
    with tf.variable_scope('ResidualUnit_Bgn'):
        with tf.variable_scope('ResidualUnit_1'):
            net = ResidualUnit(net, n_filters=48, filter_size=3)
        with tf.variable_scope('ResidualUnit_2'):
            net = ResidualUnit(net, n_filters=48, filter_size=3)
        with tf.variable_scope('ResidualUnit_3'):
            net = ResidualUnit(net, n_filters=48, filter_size=3)

    #####################
    # Downsampling Path
    #####################

    pool_scale = 1
    set_id = 0
    for j in range(len(filters_per_down_layer)):
        set_id = set_id + 1
        pool_scale = pool_scale * 2
        num_filters = filters_per_down_layer[j]
        num_frru = num_frru_per_down_layer[j]
        with tf.variable_scope('FRRU_Up_' + str(set_id)):
            if j == 0:
                pool_stream = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX', scope='max_pool')
                res_stream = slim.conv2d(net, 32, kernel_size=1, activation_fn=None, scope='conv_1x1')
            else:
                pool_stream = slim.pool(pool_stream, [2, 2], stride=[2, 2], pooling_type='MAX', scope='max_pool')
            for i in range(num_frru):
                idx = i + 1
                with tf.variable_scope('FRRU_' + str(idx)):
                    pool_stream, res_stream_out = FullResolutionResidualUnit(pool_stream=pool_stream,
                                                                             res_stream=res_stream,
                                                                             n_filters_3=num_filters,
                                                                             n_filters_1=32,
                                                                             pool_scale=pool_scale)
                # Add to form next residual stream
                res_stream = tf.add(res_stream, res_stream_out, name='add_' + str(idx))

    #####################
    # Upsampling Path
    #####################
    for j in range(len(filters_per_up_layer)):
        set_id = set_id + 1
        pool_scale = pool_scale / 2
        num_filters = filters_per_up_layer[j]
        num_frru = num_frru_per_up_layer[j]
        with tf.variable_scope('FRRU_Down_' + str(set_id)):

            pool_stream = Unpooling(pool_stream, 2)
            for i in range(num_frru):
                idx = i + 1
                with tf.variable_scope('FRRU_' + str(idx)):
                    pool_stream, res_stream_out = FullResolutionResidualUnit(pool_stream=pool_stream,
                                                                             res_stream=res_stream,
                                                                             n_filters_3=num_filters,
                                                                             n_filters_1=32,
                                                                             pool_scale=pool_scale)
                # Add to form next residual stream
                res_stream = tf.add(res_stream, res_stream_out, name='add_' + str(idx))

    pool_stream = Unpooling(pool_stream, 2)
    with tf.variable_scope('resize'):
        pool_stream = tf.image.resize_bilinear(pool_stream, size=[tf.shape(res_stream)[1], tf.shape(res_stream)[2]])

    #####################
    # Final Stage
    #####################
    net = tf.concat([pool_stream, res_stream], axis=-1)
    with tf.variable_scope('ResidualUnit_End'):
        with tf.variable_scope('ResidualUnit_1'):
            net = ResidualUnit(net, n_filters=48, filter_size=3)
        with tf.variable_scope('ResidualUnit_2'):
            net = ResidualUnit(net, n_filters=48, filter_size=3)
        with tf.variable_scope('ResidualUnit_3'):
            net = ResidualUnit(net, n_filters=48, filter_size=3)

    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')
    return net
