from __future__ import division
import os,time,cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


def conv_block(inputs, n_filters, kernel_size=[3, 3], dropout_p=0.0):
	"""
	Basic conv block for Encoder-Decoder
	Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
	Dropout (if dropout_p > 0) on the inputs
	"""
	conv = slim.conv2d(inputs, n_filters, kernel_size, activation_fn=None, normalizer_fn=None)
	out = tf.nn.relu(slim.batch_norm(conv, fused=True))
	if dropout_p != 0.0:
		out = slim.dropout(out, keep_prob=(1.0-dropout_p))
	return out


def conv_transpose_block(inputs, n_filters, kernel_size=[3, 3], dropout_p=0.0):
	"""
	Basic conv transpose block for Encoder-Decoder upsampling
	Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
	Dropout (if dropout_p > 0) on the inputs
	"""
	conv = slim.conv2d_transpose(inputs, n_filters, kernel_size=[3, 3], stride=[2, 2], activation_fn=None)
	out = tf.nn.relu(slim.batch_norm(conv))
	if dropout_p != 0.0:
		out = slim.dropout(out, keep_prob=(1.0-dropout_p))
	return out


def resize_and_add(t_a, t_b):

	t_a = tf.image.resize_bilinear(t_a, size=[tf.shape(t_b)[1], tf.shape(t_b)[2]])
	return tf.add(t_a, t_b)


def build_encoder_decoder(inputs, num_classes, preset_model="Encoder-Decoder", dropout_p=0.5, scope=None):
	"""
	Builds the Encoder-Decoder model. Inspired by SegNet with some modifications
	Optionally includes skip connections
	
	Arguments:
	  inputs: the input tensor
	  n_classes: number of classes
	  dropout_p: dropout rate applied after each convolution (0. for not using)
	
	Returns:
	  Encoder-Decoder model
	"""

	if preset_model == "Encoder-Decoder":
		has_skip = False
	elif preset_model == "Encoder-Decoder-Skip":
		has_skip = True
	else:
		raise ValueError("Unsupported Encoder-Decoder model '%s'. This function only supports Encoder-Decoder and Encoder-Decoder-Skip" % (preset_model))

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
			with tf.variable_scope('conv_' + name_id):
				net = conv_block(net, num_filters)
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
			with tf.variable_scope('conv_' + name_id):
				net = conv_block(net, num_filters)
			sub_layer_id = sub_layer_id + 1
		name_id = str(layer_id) + '_' + str(sub_layer_id)
		with tf.variable_scope('conv_' + name_id):
			net = conv_block(net, end_num_filter)
		layer_id = layer_id + 1

	#####################
	#      Softmax      #
	#####################
	with tf.variable_scope('logits'):
		net = tf.image.resize_bilinear(net, size=[tf.shape(inputs)[1], tf.shape(inputs)[2]])
		net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')
	return net

