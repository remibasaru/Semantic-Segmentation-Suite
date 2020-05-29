
import tensorflow as tf

from libs.frontends.frontend_util import batch_norm, relu6, _BATCH_NORM_DECAY


def conv2d(input_, output_dim, k_h, k_w, d_h, d_w, stddev=0.02, name='conv2d', bias=False):
	with tf.variable_scope(name):
		w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
							regularizer=tf.contrib.layers.l2_regularizer(_BATCH_NORM_DECAY),
							initializer=tf.truncated_normal_initializer(stddev=stddev))
		conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
		if bias:
			biases = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
			conv = tf.nn.bias_add(conv, biases)

		return conv


def conv2d_block(input, out_dim, k, s, is_train, name):
	with tf.name_scope(name), tf.variable_scope(name):
		net = conv2d(input, out_dim, k, k, s, s, name='conv2d')
		net = batch_norm(net, training=is_train)
		net = relu6(net)
		return net


def conv_1x1(input, output_dim, name, bias=False):
	with tf.name_scope(name):
		return conv2d(input, output_dim, 1, 1, 1, 1, stddev=0.02, name=name, bias=bias)


def pwise_block(input, output_dim, is_train, name, bias=False, scale=False):
	with tf.name_scope(name), tf.variable_scope(name):
		out = conv_1x1(input, output_dim, bias=bias, name='pwb')
		out = batch_norm(out, training=is_train, scale=scale)
		out = relu6(out)
		return out


def dwise_conv(input, k_h=3, k_w=3, channel_multiplier=1, strides=None, padding='SAME', stddev=0.02, name='dwise_conv', bias=False):
	if strides is None:
		strides = [1, 1, 1, 1]
	with tf.variable_scope(name):
		in_channel=input.get_shape().as_list()[-1]
		w = tf.get_variable('w', [k_h, k_w, in_channel, channel_multiplier],
							regularizer=tf.contrib.layers.l2_regularizer(_BATCH_NORM_DECAY),
							initializer=tf.truncated_normal_initializer(stddev=stddev))
		conv = tf.nn.depthwise_conv2d(input, w, strides, padding, rate=None, name=None,data_format=None)
		if bias:
			biases = tf.get_variable('bias', [in_channel*channel_multiplier], initializer=tf.constant_initializer(0.0))
			conv = tf.nn.bias_add(conv, biases)

		return conv


def res_block(inputs, expansion_ratio, output_dim, stride, is_train, name, bias=False, shortcut=True, is_pw=True):
	with tf.name_scope(name), tf.variable_scope(name):
		if is_pw:
			# pw
			bottleneck_dim = round(expansion_ratio * inputs.get_shape().as_list()[-1])
			net = conv_1x1(inputs, bottleneck_dim, name='pw', bias=bias)
			net = batch_norm(net, training=is_train, scale=True, name='batch_normalization_pw')
			net = relu6(net)
		else:
			net = inputs
		# dw
		net = dwise_conv(net, strides=[1, stride, stride, 1], name='dw', bias=bias)
		net = batch_norm(net, training=is_train, scale=True, name='batch_normalization_dw')
		net = relu6(net)
		# pw & linear
		net = conv_1x1(net, output_dim, name='pw_linear', bias=bias)
		net = batch_norm(net, training=is_train, scale=True, name='batch_normalization_pw_linear')

		# element wise add, only for stride==1
		if shortcut and stride == 1:
			in_dim = int(inputs.get_shape().as_list()[-1])
			if in_dim != output_dim:
				ins = conv_1x1(inputs, output_dim, name='ex_dim')
				net = ins + net
			else:
				net = inputs + net

		return net


def separable_conv(inputs, k_size, output_dim, stride, pad='SAME', channel_multiplier=1, name='sep_conv', bias=False):
	with tf.name_scope(name), tf.variable_scope(name):
		in_channel = inputs.get_shape().as_list()[-1]
		dwise_filter = tf.get_variable('dw', [k_size, k_size, in_channel, channel_multiplier],
									   regularizer=tf.contrib.layers.l2_regularizer(_BATCH_NORM_DECAY),
									   initializer=tf.truncated_normal_initializer(stddev=0.02))

		pwise_filter = tf.get_variable('pw', [1, 1, in_channel*channel_multiplier, output_dim],
									   regularizer=tf.contrib.layers.l2_regularizer(_BATCH_NORM_DECAY),
									   initializer=tf.truncated_normal_initializer(stddev=0.02))
		strides = [1, stride, stride, 1]

		conv=tf.nn.separable_conv2d(inputs, dwise_filter, pwise_filter, strides, padding=pad, name=name)
		if bias:
			biases = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
			conv = tf.nn.bias_add(conv, biases)
		return conv


def global_avg(x):
	with tf.name_scope('global_avg'):
		net=tf.layers.average_pooling2d(x, x.get_shape()[1:-1], 1)
		return net


def flatten(x):
	return tf.contrib.layers.flatten(x)


def pad2d(inputs, pad=(0, 0), mode='CONSTANT'):
	paddings = [[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]]
	net = tf.pad(inputs, paddings, mode=mode)
	return net
