import os,time,cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


def ConvBlock(inputs, n_filters, kernel_size=[3, 3]):
	"""
	Builds the conv block for MobileNets
	Apply successivly a 2D convolution, BatchNormalization relu
	"""
	# Skip pointwise by setting num_outputs=Non
	net = slim.conv2d(inputs, n_filters, kernel_size=[1, 1], activation_fn=None)
	net = slim.batch_norm(net, fused=True)
	net = tf.nn.relu(net)
	return net


def DepthwiseSeparableConvBlock(inputs, n_filters, kernel_size=[3, 3]):
	"""
	Builds the Depthwise Separable conv block for MobileNets
	Apply successivly a 2D separable convolution, BatchNormalization relu, conv, BatchNormalization, relu
	"""
	# Skip pointwise by setting num_outputs=None
	net = slim.separable_convolution2d(inputs, num_outputs=None, depth_multiplier=1, kernel_size=[3, 3], activation_fn=None)

	net = slim.batch_norm(net, fused=True)
	net = tf.nn.relu(net)
	net = slim.conv2d(net, n_filters, kernel_size=[1, 1], activation_fn=None)
	net = slim.batch_norm(net, fused=True)
	net = tf.nn.relu(net)
	return net


def conv_transpose_block(inputs, n_filters, kernel_size=[3, 3]):
	"""
	Basic conv transpose block for Encoder-Decoder upsampling
	Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
	"""
	net = slim.conv2d_transpose(inputs, n_filters, kernel_size=[3, 3], stride=[2, 2], activation_fn=None)
	net = tf.nn.relu(slim.batch_norm(net))
	return net


def resize_and_add(t_a, t_b):

	t_a = tf.image.resize_bilinear(t_a, size=[tf.shape(t_b)[1], tf.shape(t_b)[2]])
	return tf.add(t_a, t_b)


def build_mobile_unet(inputs, preset_model, num_classes):

	has_skip = False
	if preset_model == "MobileUNet":
		has_skip = False
	elif preset_model == "MobileUNet-Skip":
		has_skip = True
	else:
		raise ValueError("Unsupported MobileUNet model '%s'. This function only supports MobileUNet and MobileUNet-Skip" % (preset_model))


	#####################
	# Downsampling path #
	#####################

	filters_per_unit = [64, 128, 256, 512, 512]
	layer_per_unit = [2, 2, 3, 3, 3]
	prev_skip = []
	net = inputs
	layer_id = 1
	for i, num_filters in enumerate(filters_per_unit):

		num_layers = layer_per_unit[i]
		sub_layer_id = 1
		for j in range(num_layers):
			name_id = str(layer_id) + '_' + str(sub_layer_id)
			if i == 0 and j == 0:
				with tf.variable_scope('conv_' + name_id):
					net = ConvBlock(net, num_filters)
			else:
				with tf.variable_scope('DSconv_' + name_id):
					net = DepthwiseSeparableConvBlock(net, num_filters)
			sub_layer_id = sub_layer_id + 1
		with tf.variable_scope('max_pool' + '_' + str(layer_id)):
			net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX', scope='pool' + '_' + str(layer_id))
			if has_skip:
				prev_skip.append(net)
		layer_id = layer_id + 1

	#####################
	# Upsampling path #
	#####################

	filters_per_unit.reverse()
	layer_per_unit.reverse()

	for i, num_filters in enumerate(filters_per_unit):
		num_layers = layer_per_unit[i]
		sub_layer_id = 1
		if has_skip:
			if i != 0:
				with tf.variable_scope('add' + '_' + str(layer_id)):
					net = resize_and_add(net, prev_skip.pop())
			else:
				prev_skip.pop()
		with tf.variable_scope('Tconv' + '_' + str(layer_id)):
			net = conv_transpose_block(net, num_filters)
		end_num_filter = filters_per_unit[min(i + 1, len(filters_per_unit) - 1)]
		for j in range(num_layers - 1):
			name_id = str(layer_id) + '_' + str(sub_layer_id)
			with tf.variable_scope('DSconv_' + name_id):
				net = DepthwiseSeparableConvBlock(net, num_filters)
			sub_layer_id = sub_layer_id + 1
		name_id = str(layer_id) + '_' + str(sub_layer_id)
		with tf.variable_scope('DSconv_' + name_id):
			net = DepthwiseSeparableConvBlock(net, end_num_filter)
		layer_id = layer_id + 1

	#####################
	#      Softmax      #
	#####################
	with tf.variable_scope('logits'):
		net = tf.image.resize_bilinear(net, size=[tf.shape(inputs)[1], tf.shape(inputs)[2]])
		net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')

	return net