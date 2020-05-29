# --------------------------------------------------------
# Written by: Rilwan Remilekun Basaru
# --------------------------------------------------------
from collections import OrderedDict

import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def relu6(x, name='relu6'):
	return tf.nn.relu6(x, name)


def batch_norm(inputs, training, axis=-1, scale=False, name=None):
	# We set fused=True for a significant performance boost. See
	# https://www.tensorflow.org/performance/performance_guide  # common_fused_ops

	return tf.layers.batch_normalization(inputs=inputs, axis=axis, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
										 center=True, scale=scale, training=training, fused=True, name=name)


def fixed_padding(inputs, kernel_size):

	"""
	Pads the input along the spatial dimensions independently of input size.

	Args:
	inputs: A tensor of size [batch, channels, height_in, width_in] or
	[batch, height_in, width_in, channels] depending on data_format.
	kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
				 Should be a positive integer.
	data_format: The input format ('channels_last' or 'channels_first').

	Returns:
	A tensor with the same format as the input with the data either intact
	(if kernel_size == 1) or padded (if kernel_size > 1).
	"""

	pad_total = kernel_size - 1
	pad_beg = pad_total // 2
	pad_end = pad_total - pad_beg

	padded_inputs = tf.pad(tensor=inputs, paddings=[[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
	return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, name=None, use_bias=False):
	"""Strided 2-D convolution with explicit padding."""
	# The padding is consistent and is based only on `kernel_size`, not on the
	# dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
	if strides > 1:
		inputs = fixed_padding(inputs, kernel_size)

	return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
							padding=('SAME' if strides == 1 else 'VALID'), use_bias=use_bias,
							kernel_initializer=tf.variance_scaling_initializer(), data_format='channels_last', name=name)


class MetaFrontEnd:

	def __init__(self):
		self.num_classes = 2
		self.end_points = None
		self.lvl = None
		self.requested_stages = None

	def update_endpoint(self, inputs):

		if self.requested_stages is not None:
			if self.lvl > max(self.requested_stages):
				return True

			if self.lvl in self.requested_stages:
				self.end_points['stage' + str(self.lvl)] = inputs
			else:
				self.end_points['stage' + str(self.lvl)] = None
		else:
			self.end_points['stage' + str(self.lvl)] = inputs
		self.lvl += 1
		return False

	def build_model(self, inputs, training, requested_stages=None):
		self.end_points = OrderedDict()
		self.lvl = 1
		self.requested_stages = requested_stages
