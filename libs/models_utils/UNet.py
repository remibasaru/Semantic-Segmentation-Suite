import os
from functools import partial

import tensorflow as tf


def get_filt_size(filter_size=1):
    fs_c = 2 * filter_size + 1
    fs_dc = 3
    return fs_c, fs_dc


def build_unet(input_img, train_mode, num_classes=2):
    fs_c, fs_dc = get_filt_size()
    growth_fact = 2

    # First layer (Convoluted)
    num_filters = 64
    conv1_1 = conv_layer(input_img, fs_c, num_filters, "conv1_1", train_mode, in_channels=3)
    conv1_2 = conv_layer(conv1_1, fs_c, num_filters, "conv1_2", train_mode)
    pool1 = max_pool(conv1_2, 'pool1')

    # Second layer (Convoluted)
    num_filters = num_filters * growth_fact
    conv2_1 = conv_layer(pool1, fs_c, num_filters, "conv2_1", train_mode)
    conv2_2 = conv_layer(conv2_1, fs_c, num_filters, "conv2_2", train_mode)
    pool2 = max_pool(conv2_2, 'pool2')

    # Third layer (Convoluted)
    num_filters = num_filters * growth_fact
    convl1_1, biasl1_1, bnorml1_1, conv3_1 = conv_layer(pool2, fs_c, num_filters, "conv3_1", train_mode, debug=True)
    conv3_2 = conv_layer(conv3_1, fs_c, num_filters, "conv3_2", train_mode)
    conv3_3 = conv_layer(conv3_2, fs_c, num_filters, "conv3_3", train_mode)
    pool3 = max_pool(conv3_3, 'pool3')

    # Fourth layer (Convoluted)
    num_filters = num_filters * growth_fact
    conv4_1 = conv_layer(pool3, fs_c, num_filters, "conv4_1", train_mode)
    conv4_2 = conv_layer(conv4_1, fs_c, num_filters, "conv4_2", train_mode)
    conv4_3 = conv_layer(conv4_2, fs_c, num_filters, "conv4_3", train_mode)
    pool4 = max_pool(conv4_3, 'pool4')

    # Fifth layer (Convoluted)
    conv5_1 = conv_layer(pool4, fs_c, num_filters, "conv5_1", train_mode)
    conv5_2 = conv_layer(conv5_1, fs_c, num_filters, "conv5_2", train_mode)
    conv5_3 = conv_layer(conv5_2, fs_c, num_filters, "conv5_3", train_mode)
    pool5 = max_pool(conv5_3, 'pool5')

    # Sixth layer (Convoluted)
    conv6 = conv_layer(pool5, fs_c, num_filters, "conv6", train_mode)

    # =====================***********************  Up-scaling  ************************========================== #

    # Seventh layer (Convoluted)
    conv7 = conv_layer(conv6, 1, num_filters, "conv7", train_mode)

    # Eight layer (Deconvoluted)
    conv8Tc = deconv_layer(conv7, fs_dc, num_filters, conv5_3, "conv8Tc")
    conv8_1 = conv_layer(conv8Tc, fs_c, num_filters, "conv8_1", train_mode)
    conv8_2 = conv_layer(conv8_1, fs_c, num_filters, "conv8_2", train_mode)
    conv8_3 = conv_layer(conv8_2, fs_c, num_filters, "conv8_3", train_mode)

    # Ninth layer (Deconvoluted)
    conv9Tc = deconv_layer(conv8_3, fs_dc, num_filters, conv4_3, "conv9Tc")
    conv9_1 = conv_layer(conv9Tc, fs_c, num_filters, "conv9_1", train_mode)
    conv9_2 = conv_layer(conv9_1, fs_c, num_filters, "conv9_2", train_mode)
    conv9_3 = conv_layer(conv9_2, fs_c, num_filters, "conv9_3", train_mode)

    # Tenth layer (Deconvoluted)
    num_filters = int(num_filters/growth_fact)
    conv10Tc = deconv_layer(conv9_3, fs_dc, num_filters, conv3_3, "conv10Tc")
    conv10_1 = conv_layer(conv10Tc, fs_c, num_filters, "conv10_1", train_mode)
    conv10_2 = conv_layer(conv10_1, fs_c, num_filters, "conv10_2", train_mode)
    conv10_3 = conv_layer(conv10_2, fs_c, num_filters, "conv10_3", train_mode)

    # Eleventh layer (Deconvoluted)
    num_filters = int(num_filters/growth_fact)
    conv11Tc = deconv_layer(conv10_3, fs_dc, num_filters, conv2_2, "conv11Tc")
    conv11_1 = conv_layer(conv11Tc, fs_c, num_filters, "conv11_1", train_mode)
    conv11_2 = conv_layer(conv11_1, fs_c, num_filters, "conv11_2", train_mode)

    # Twelve layer (Deconvoluted)
    num_filters = int(num_filters/growth_fact)
    conv12Tc = deconv_layer(conv11_2, fs_dc, num_filters, conv1_2, "conv12Tc")
    conv12_1 = conv_layer(conv12Tc, fs_c, num_filters, "conv12_1", train_mode)
    conv12_2 = conv_layer(conv12_1, fs_c, num_classes, "conv12_2", train_mode, batch_n=False, relu_n=False)

    # prob = tf.nn.softmax(conv12_2, name="prob")

    return conv12_2, partial(load_pre_trained, variables=tf.trainable_variables()), \
           load_pre_trained(None, tf.global_variables(), update=False)


def get_tensor_handles(filters=['Conv2D', 'Relu']):
    probe = dict()

    def run_filter(tensor_name):
        for filter_name in filters:
            if filter_name in tensor_name:
                return True
        return False

    for node in tf.get_default_graph().get_operations():
        if len(node.outputs) == 0:
            continue
        node_id = node.outputs[0].name.replace('/', '_').replace(':', '_')
        if filters is not None:
            if run_filter(node.name):
                probe[node_id] = node.outputs[0]
        else:
            probe[node_id] = node.outputs[0]
    return probe


def build_small_unet(input_img, train_mode, num_classes=2, win_siz=1):

    fs_c, fs_dc = get_filt_size(win_siz)
    growth_fact = 2

    # First layer (Convoluted)
    num_filters = 64
    conv1_1 = conv_layer(input_img, fs_c, num_filters, "conv1_1", train_mode, in_channels=3)
    conv1_2 = conv_layer(conv1_1, fs_c, num_filters, "conv1_2", train_mode)
    pool1 = max_pool(conv1_2, 'pool1')

    # Second layer (Convoluted)
    num_filters = num_filters * growth_fact
    conv2_1 = conv_layer(pool1, fs_c, num_filters, "conv2_1", train_mode)
    conv2_2 = conv_layer(conv2_1, fs_c, num_filters, "conv2_2", train_mode)
    pool2 = max_pool(conv2_2, 'pool2')

    # Third layer (Convoluted)
    conv3_1 = conv_layer(pool2, fs_c, num_filters, "conv3_1", train_mode)
    conv3_2 = conv_layer(conv3_1, fs_c, num_filters, "conv3_2", train_mode)
    pool3 = max_pool(conv3_2, 'pool3')

    num_filters = num_filters * growth_fact
    # Fourth layer (Convoluted)
    conv4 = conv_layer(pool3, fs_c, num_filters, "conv4", train_mode)
    conv4_1 = conv_layer(conv4, fs_c, num_filters, "conv4_1", train_mode)
    pool4 = max_pool(conv4_1, 'pool4')

    conv5 = conv_layer(pool4, fs_c, num_filters, "conv5", train_mode)

    # =====================***********************  Up-scaling  ************************========================== #

    # Fifth layer (Convoluted)
    conv6 = conv_layer(conv5, 1, num_filters, "conv6", train_mode)

    conv7Tc = deconv_layer(conv6, fs_dc, num_filters, conv4_1, "conv7Tc")
    conv7_1 = conv_layer(conv7Tc, fs_c, num_filters, "conv7_1", train_mode)
    conv7_2 = conv_layer(conv7_1, fs_c, num_filters, "conv7_2", train_mode)

    num_filters = int(num_filters / growth_fact)
    # Sixth layer (Deconvoluted)
    conv8Tc = deconv_layer(conv7_2, fs_dc, num_filters, conv3_2, "conv8Tc")
    conv8_1 = conv_layer(conv8Tc, fs_c, num_filters, "conv8_1", train_mode)
    conv8_2 = conv_layer(conv8_1, fs_c, num_filters, "conv8_2", train_mode)

    # Seventh layer (Deconvoluted)
    conv9Tc = deconv_layer(conv8_2, fs_dc, num_filters, conv2_2, "conv9Tc")
    conv9_1 = conv_layer(conv9Tc, fs_c, num_filters, "conv9_1", train_mode)
    conv9_2 = conv_layer(conv9_1, fs_c, num_filters, "conv9_2", train_mode)

    # Eight layer (Deconvoluted)
    num_filters = int(num_filters/growth_fact)
    conv10Tc = deconv_layer(conv9_2, fs_dc, num_filters, conv1_2, "conv10Tc")
    conv10_1 = conv_layer(conv10Tc, fs_c, num_filters, "conv10_1", train_mode)
    conv10_2 = conv_layer(conv10_1, fs_c, num_classes, "conv10_2", train_mode, batch_n=False, relu_n=False)

    # prob = tf.nn.softmax(conv12_2, name="prob")

    return conv10_2, None, [], get_tensor_handles()


def get_conv_var(filter_size, in_channels, out_channels, name, is_bias=True):
    initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
    filters = get_var(initial_value, name + "_filters")
    if is_bias:
        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = get_var(initial_value, name + "_biases")
    else:
        biases = None

    return filters, biases


def get_convT_var(filter_size, out_channels, in_channels, name):
    initial_value = tf.truncated_normal([filter_size, filter_size, out_channels, in_channels], 0.0, 0.001)
    filters = get_var(initial_value, name + "_filters")

    initial_value = tf.truncated_normal([out_channels], .0,  0.001)
    biases = get_var(initial_value, name + "_biases")

    return filters, biases


def max_pool(bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def batch_norm(bottom, name):
    with tf.variable_scope(name):

        batch_mean, batch_var = tf.nn.moments(bottom, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(tf.constant(True),
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))

        n_out = bottom.shape[3]
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
        normed = tf.nn.batch_normalization(bottom, mean, var, offset=beta,  scale=gamma, variance_epsilon=1e-5)

        return normed


def trainable_batch_norm(bottom, name, momentum=0.0009, epsilon=1e-5, train=True):
    with tf.variable_scope(name):
        output = tf.layers.batch_normalization(bottom,
                                               momentum=momentum,
                                               epsilon=epsilon,
                                               scale=True,
                                               beta_initializer=tf.zeros_initializer(),
                                               gamma_initializer=tf.ones_initializer(),
                                               moving_mean_initializer=tf.zeros_initializer(),
                                               moving_variance_initializer=tf.ones_initializer(),
                                               training=train,
                                               name=name)
    return output


def conv_layer(bottom, filter_size, out_channels, name, train_mode, batch_n=True, relu_n=True, debug=False,
               in_channels=None):
    with tf.variable_scope(name):
        if in_channels is None:
            in_channels = int(bottom.shape[3])
        filt, conv_biases = get_conv_var(filter_size, in_channels, out_channels, name, is_bias=not batch_n)
        # Not need to add bias if batch normalization is to be added
        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME', data_format="NHWC")
        # bias = tf.nn.bias_add(conv, conv_biases)
        if batch_n:
            bias = conv
            # bnorm = batch_norm(bias, name)
            bnorm = trainable_batch_norm(bias, name, train=train_mode)
        else:
            bias = tf.nn.bias_add(conv, conv_biases)
            bnorm = bias
        if relu_n:
            relu = tf.nn.relu(bnorm)
        else:
            relu = bnorm

        if debug:
            return conv, bias, bnorm, relu
        else:
            return relu


def deconv_layer(bottom_l, filter_size, in_channels,  bottom_r, name, out_channels=None):
    with tf.variable_scope(name):
        if out_channels is None:
            out_channels = int(bottom_l.shape[3])
        filt, conv_biases = get_convT_var(filter_size, in_channels, out_channels,   name)
        deconv = tf.nn.conv2d_transpose(bottom_l, filt, output_shape=tf.shape(bottom_r),
                                        strides=[1, 2, 2, 1], padding='SAME')
        bias = tf.nn.bias_add(deconv, conv_biases)

        # cropped = tf.slice(bias, crop, tf.shape(bias) - crop)

        conct = tf.concat([bottom_r, bias], axis=3)

        return conct


def get_var(initial_value, var_name):
        value = initial_value
        var = tf.Variable(value, name=var_name)

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var


stable_params = [
        'conv1_1/conv1_1_filters:0',
        'conv1_1/conv1_1/conv1_1/beta:0',
        'conv1_1/conv1_1/conv1_1/gamma:0',
        'conv1_2/conv1_2_filters:0',
        'conv1_2/conv1_2/conv1_2/beta:0',
        'conv1_2/conv1_2/conv1_2/gamma:0',

        'conv2_1/conv2_1_filters:0',
        'conv2_1/conv2_1/conv2_1/beta:0',
        'conv2_1/conv2_1/conv2_1/gamma:0',
        'conv2_2/conv2_2_filters:0',
        'conv2_2/conv2_2/conv2_2/beta:0',
        'conv2_2/conv2_2/conv2_2/gamma:0',

        'conv3_1/conv3_1_filters:0',
        'conv3_1/conv3_1/conv3_1/beta:0',
        'conv3_1/conv3_1/conv3_1/gamma:0',
        'conv3_2/conv3_2_filters:0',
        'conv3_2/conv3_2/conv3_2/beta:0',
        'conv3_2/conv3_2/conv3_2/gamma:0',
        'conv3_3/conv3_3_filters:0',
        'conv3_3/conv3_3/conv3_3/beta:0',
        'conv3_3/conv3_3/conv3_3/gamma:0',

        'conv4_1/conv4_1_filters:0',
        'conv4_1/conv4_1/conv4_1/beta:0',
        'conv4_1/conv4_1/conv4_1/gamma:0',
        'conv4_2/conv4_2_filters:0',
        'conv4_2/conv4_2/conv4_2/beta:0',
        'conv4_2/conv4_2/conv4_2/gamma:0',
        'conv4_3/conv4_3_filters:0',
        'conv4_3/conv4_3/conv4_3/beta:0',
        'conv4_3/conv4_3/conv4_3/gamma:0',

        'conv5_1/conv5_1_filters:0',
        'conv5_1/conv5_1/conv5_1/beta:0',
        'conv5_1/conv5_1/conv5_1/gamma:0',
        'conv5_2/conv5_2_filters:0',
        'conv5_2/conv5_2/conv5_2/beta:0',
        'conv5_2/conv5_2/conv5_2/gamma:0',
        'conv5_3/conv5_3_filters:0',
        'conv5_3/conv5_3/conv5_3/beta:0'
        'conv5_3/conv5_3/conv5_3/gamma:0']


def load_pre_trained(sess, variables, update=True):
    pre_trained_model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'models', 'vgg16', 'vgg_16.ckpt')
    model = tf.train.NewCheckpointReader(pre_trained_model_path)

    name_map = {
        'conv1_1/conv1_1_filters:0': 'vgg_16/conv1/conv1_1/weights',
        'conv1_1/conv1_1/conv1_1/beta:0': 'vgg_16/conv1/conv1_1/biases',
        'conv1_2/conv1_2_filters:0': 'vgg_16/conv1/conv1_2/weights',
        'conv1_2/conv1_2/conv1_2/beta:0': 'vgg_16/conv1/conv1_2/biases',

        'conv2_1/conv2_1_filters:0': 'vgg_16/conv2/conv2_1/weights',
        'conv2_1/conv2_1/conv2_1/beta:0': 'vgg_16/conv2/conv2_1/biases',
        'conv2_2/conv2_2_filters:0': 'vgg_16/conv2/conv2_2/weights',
        'conv2_2/conv2_2/conv2_2/beta:0': 'vgg_16/conv2/conv2_2/biases',

        'conv3_1/conv3_1_filters:0': 'vgg_16/conv3/conv3_1/weights',
        'conv3_1/conv3_1/conv3_1/beta:0': 'vgg_16/conv3/conv3_1/biases',
        'conv3_2/conv3_2_filters:0': 'vgg_16/conv3/conv3_2/weights',
        'conv3_2/conv3_2/conv3_2/beta:0': 'vgg_16/conv3/conv3_2/biases',
        'conv3_3/conv3_3_filters:0': 'vgg_16/conv3/conv3_3/weights',
        'conv3_3/conv3_3/conv3_3/beta:0': 'vgg_16/conv3/conv3_3/biases',

        'conv4_1/conv4_1_filters:0': 'vgg_16/conv4/conv4_1/weights',
        'conv4_1/conv4_1/conv4_1/beta:0': 'vgg_16/conv4/conv4_1/biases',
        'conv4_2/conv4_2_filters:0': 'vgg_16/conv4/conv4_2/weights',
        'conv4_2/conv4_2/conv4_2/beta:0': 'vgg_16/conv4/conv4_2/biases',
        'conv4_3/conv4_3_filters:0': 'vgg_16/conv4/conv4_3/weights',
        'conv4_3/conv4_3/conv4_3/beta:0': 'vgg_16/conv4/conv4_3/biases',

        'conv5_1/conv5_1_filters:0': 'vgg_16/conv5/conv5_1/weights',
        'conv5_1/conv5_1/conv5_1/beta:0': 'vgg_16/conv5/conv5_1/biases',
        'conv5_2/conv5_2_filters:0': 'vgg_16/conv5/conv5_2/weights',
        'conv5_2/conv5_2/conv5_2/beta:0': 'vgg_16/conv5/conv5_2/biases',
        'conv5_3/conv5_3_filters:0': 'vgg_16/conv5/conv5_3/weights',
        'conv5_3/conv5_3/conv5_3/beta:0': 'vgg_16/conv5/conv5_3/biases'
    }
    assign_count = 0
    assigned_tensors = []
    for var in variables:
        # print(var.name, ' ', str(var.shape).replace(' ', ''))
        if var.name in name_map:
            # print(var.name)
            src_name = name_map[var.name]
            assigned_tensors.append(var)
            if update:
                try:
                    param = model.get_tensor(src_name)

                    sess.run(tf.assign(var, param))
                    assign_count += 1
                except ValueError:
                    print('Incompatible load, ignoring: ', var.name, ' with shape: ', var.shape)
        else:
            # print(var.name)
            pass
    if update:
        print(str(assign_count) + " parameters assigned")
    else:
        print(str(len(assigned_tensors)) + " assignable parameters")

    return assigned_tensors


if __name__ == "__main__":
    pre_trained_model_path = os.path.join('..', 'models', 'vgg16', 'vgg_16.ckpt')
    model = tf.train.NewCheckpointReader(pre_trained_model_path)
    # b = model.get_variable_to_shape_map()
    name_map = {
        'vgg_16/conv1/conv1_1/weights': 'conv1_1/conv1_1_filters:0',
        'vgg_16/conv1/conv1_1/biases': 'conv1_1/conv1_1/conv1_1/beta:0',
        'vgg_16/conv1/conv1_2/weights': 'conv1_2/conv1_2_filters:0',
        'vgg_16/conv1/conv1_2/biases': 'conv1_2/conv1_2/conv1_2/beta:0',

        'vgg_16/conv2/conv2_1/weights': 'conv2_1/conv2_1_filters:0',
        'vgg_16/conv2/conv2_1/biases': 'conv2_1/conv2_1/conv2_1/beta:0',
        'vgg_16/conv2/conv2_2/weights': 'conv2_2/conv2_2_filters:0',
        'vgg_16/conv2/conv2_2/biases': 'conv2_2/conv2_2/conv2_2/beta:0',

        'vgg_16/conv3/conv3_1/weights': 'conv3_1/conv3_1_filters:0',
        'vgg_16/conv3/conv3_1/biases': 'conv3_1/conv3_1/conv3_1/beta:0',
        'vgg_16/conv3/conv3_2/weights': 'conv3_2/conv3_2_filters:0',
        'vgg_16/conv3/conv3_2/biases': 'conv3_2/conv3_2/conv3_2/beta:0',
        'vgg_16/conv3/conv3_3/weights': 'conv3_3/conv3_3_filters:0',
        'vgg_16/conv3/conv3_3/biases': 'conv3_3/conv3_3/conv3_3/beta:0',

        'vgg_16/conv4/conv4_1/weights': 'conv4_1/conv4_1_filters:0',
        'vgg_16/conv4/conv4_1/biases': 'conv4_1/conv4_1/conv4_1/beta:0',
        'vgg_16/conv4/conv4_2/weights': 'conv4_2/conv4_2_filters:0',
        'vgg_16/conv4/conv4_2/biases': 'conv4_2/conv4_2/conv4_2/beta:0',
        'vgg_16/conv4/conv4_3/weights': 'conv4_3/conv4_3_filters:0',
        'vgg_16/conv4/conv4_3/biases': 'conv4_3/conv4_3/conv4_3/beta:0',

        'vgg_16/conv5/conv5_1/weights': 'conv5_1/conv5_1_filters:0',
        'vgg_16/conv5/conv5_1/biases': 'conv5_1/conv5_1/conv5_1/beta:0',
        'vgg_16/conv5/conv5_2/weights': 'conv5_2/conv5_2_filters:0',
        'vgg_16/conv5/conv5_2/biases': 'conv5_2/conv5_2/conv5_2/beta:0',
        'vgg_16/conv5/conv5_3/weights': 'conv5_3/conv5_3_filters:0',
        'vgg_16/conv5/conv5_3/biases': 'conv5_3/conv5_3/conv5_3/beta:0'
    }

















