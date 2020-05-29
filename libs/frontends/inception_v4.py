# --------------------------------------------------------
# Written by: Rilwan Remilekun Basaru
# --------------------------------------------------------

import tensorflow as tf

from libs.frontends.frontend_util import MetaFrontEnd, batch_norm


def conv_block(x, nb_filter, nb_row, nb_col, is_training, border_mode='same', subsample=(1, 1), bias=False, ):
	channel_axis = -1

	x = tf.layers.conv2d(inputs=x, filters=nb_filter, kernel_size=(nb_row, nb_col), padding=border_mode.upper(),
						 use_bias=bias, strides=subsample)

	x = batch_norm(x, is_training, axis=channel_axis)

	x = tf.nn.relu(x)
	return x


def inception_A(input_t, is_training, idx):
	channel_axis = -1

	with tf.variable_scope('Inception-A' + str(idx + 1)):
		with tf.variable_scope('ave_pool'):
			a1 = tf.layers.average_pooling2d(input_t, (3, 3), strides=(1, 1), padding='same')
		with tf.variable_scope('1x1_conv_1_1'):
			a1 = conv_block(a1, 96, 1, 1, is_training)

		with tf.variable_scope('1x1_conv_2_1'):
			a2 = conv_block(input_t, 96, 1, 1, is_training)

		with tf.variable_scope('1x1_conv_3_1'):
			a3 = conv_block(input_t, 64, 1, 1, is_training)
		with tf.variable_scope('3x3_conv_3_2'):
			a3 = conv_block(a3, 96, 3, 3, is_training)

		with tf.variable_scope('1x1_conv_4_1'):
			a4 = conv_block(input_t, 64, 1, 1, is_training)
		with tf.variable_scope('3x3_conv_4_2'):
			a4 = conv_block(a4, 96, 3, 3, is_training)
		with tf.variable_scope('3x3_conv_4_3'):
			a4 = conv_block(a4, 96, 3, 3, is_training)

		m = tf.concat(axis=channel_axis, values=[a1, a2, a3, a4])
	return m


def inception_B(input_t, is_training, idx):
	channel_axis = -1

	with tf.variable_scope('Inception-B' + str(idx + 1)):
		with tf.variable_scope('ave_pool'):
			b1 = tf.layers.average_pooling2d(input_t, (3, 3), strides=(1, 1), padding='same')
		with tf.variable_scope('1x1_conv_1_1'):
			b1 = conv_block(b1, 128, 1, 1, is_training)

		with tf.variable_scope('1x1_conv_2_1'):
			b2 = conv_block(input_t, 384, 1, 1, is_training)

		with tf.variable_scope('1x1_conv_3_1'):
			b3 = conv_block(input_t, 192, 1, 1, is_training)
		with tf.variable_scope('7x1_conv_3_2'):
			b3 = conv_block(b3, 224, 1, 7, is_training)
		with tf.variable_scope('1x7_conv_3_3'):
			b3 = conv_block(b3, 256, 7, 1, is_training)

		with tf.variable_scope('1x1_conv_4_1'):
			b4 = conv_block(input_t, 192, 1, 1, is_training)
		with tf.variable_scope('1x7_conv_4_2'):
			b4 = conv_block(b4, 192, 7, 1, is_training)
		with tf.variable_scope('7x1_conv_4_3'):
			b4 = conv_block(b4, 224, 1, 7, is_training)
		with tf.variable_scope('1x7_conv_4_4'):
			b4 = conv_block(b4, 224, 7, 1, is_training)
		with tf.variable_scope('7x1_conv_4_5'):
			b4 = conv_block(b4, 256, 1, 7, is_training)

		m = tf.concat(axis=channel_axis, values=[b1, b2, b3, b4])
	return m


def inception_C(input_t, is_training, idx):
	channel_axis = -1

	with tf.variable_scope('Inception-C' + str(idx + 1)):
		with tf.variable_scope('ave_pool'):
			c1 = tf.layers.average_pooling2d(input_t, (3, 3), strides=(1, 1), padding='same')
		with tf.variable_scope('1x1_conv_1_1'):
			c1 = conv_block(c1, 256, 1, 1, is_training)

		with tf.variable_scope('1x1_conv_2_1'):
			c2 = conv_block(input_t, 256, 1, 1, is_training)

		with tf.variable_scope('1x1_conv_3_1'):
			c3 = conv_block(input_t, 384, 1, 1, is_training)
		with tf.variable_scope('3x1_conv_3_1_1'):
			c3_1 = conv_block(c3, 256, 1, 3, is_training)
		with tf.variable_scope('1x3_conv_3_1_2'):
			c3_2 = conv_block(c3, 256, 3, 1, is_training)

		c3 = tf.concat(axis=channel_axis, values=[c3_1, c3_2], name='concat_1')

		with tf.variable_scope('1x1_conv_4_1'):
			c4 = conv_block(input_t, 384, 1, 1, is_training)
		with tf.variable_scope('1x3_conv_4_2'):
			c4 = conv_block(c4, 448, 3, 1, is_training)
		with tf.variable_scope('3x1_conv_4_3'):
			c4 = conv_block(c4, 512, 1, 3, is_training)
		with tf.variable_scope('3x1_conv_4_3_1'):
			c4_1 = conv_block(c4, 256, 1, 3, is_training)
		with tf.variable_scope('3x1_conv_4_3_2'):
			c4_2 = conv_block(c4, 256, 3, 1, is_training)

		c4 = tf.concat(axis=channel_axis, values=[c4_1, c4_2], name='concat_2')

		m = tf.concat(axis=channel_axis, values=[c1, c2, c3, c4])
	return m


def reduction_A(input_t, is_training):
	
	channel_axis = -1
	with tf.variable_scope('Reduction-A'):
		with tf.variable_scope('b0_Conv2d_1a_3x3'):
			r1 = conv_block(input_t, 384, 3, 3, is_training, subsample=(2, 2), border_mode='same')

		with tf.variable_scope('b1_Conv2d_0a_1x1'):
			r2 = conv_block(input_t, 192, 1, 1, is_training)
		with tf.variable_scope('b1_Conv2d_0b_3x3'):
			r2 = conv_block(r2, 224, 3, 3, is_training)
		with tf.variable_scope('b1_Conv2d_1a_3x3'):
			r2 = conv_block(r2, 256, 3, 3, is_training, subsample=(2, 2), border_mode='same')

		r3 = tf.layers.max_pooling2d(input_t, (3, 3), strides=(2, 2), padding='same')

		m = tf.concat(axis=channel_axis, values=[r1, r2, r3])
	return m


def reduction_B(input_t, is_training):
	channel_axis = -1
	with tf.variable_scope('Reduction-B'):
		with tf.variable_scope('b0_Conv2d_0a_1x1'):
			r1 = conv_block(input_t, 192, 1, 1, is_training)
		with tf.variable_scope('b0_Conv2d_1a_3x3'):
			r1 = conv_block(r1, 192, 3, 3, is_training, subsample=(2, 2), border_mode='same')

		with tf.variable_scope('b1_Conv2d_0a_1x1'):
			r2 = conv_block(input_t, 256, 1, 1, is_training)
		with tf.variable_scope('b1_Conv2d_0b_7x1'):
			r2 = conv_block(r2, 256, 1, 7, is_training)
		with tf.variable_scope('b1_Conv2d_0b_1x7'):
			r2 = conv_block(r2, 320, 7, 1, is_training)
		with tf.variable_scope('b1_onv2d_1a_3x3'):
			r2 = conv_block(r2, 320, 3, 3, is_training, subsample=(2, 2), border_mode='same')
	
		r3 = tf.layers.max_pooling2d(input_t, (3, 3), strides=(2, 2), padding='same')
	
		m = tf.concat(axis=channel_axis, values=[r1, r2, r3])
	return m


class InceptionV4(MetaFrontEnd):

	def __init__(self):
		super().__init__()

	def inception_stem(self, inputs, is_training):
		channel_axis = -1

		with tf.variable_scope('conv_1'):
			x = conv_block(inputs, 32, 3, 3, is_training, subsample=(2, 2), border_mode='same')
		if self.update_endpoint(inputs=x):  # Stage 1
			return x, True
		with tf.variable_scope('conv_2'):
			x = conv_block(x, 32, 3, 3, is_training, border_mode='same')
		with tf.variable_scope('conv_3'):
			x = conv_block(x, 64, 3, 3, is_training)

		with tf.variable_scope('3x3_max_pool'):
			x1 = tf.layers.max_pooling2d(x, (3, 3), strides=(2, 2), padding='same')
		with tf.variable_scope('3x3_Conv'):
			x2 = conv_block(x, 96, 3, 3, is_training, subsample=(2, 2), border_mode='same')

		with tf.variable_scope('filt_concat_1'):
			x = tf.concat(axis=channel_axis, values=[x1, x2])
		if self.update_endpoint(inputs=x):  # Stage 2
			return x, True

		with tf.variable_scope('1x1_Conv_1_1'):
			x1 = conv_block(x, 64, 1, 1, is_training)
		with tf.variable_scope('3x3_Conv_1_2'):
			x1 = conv_block(x1, 96, 3, 3, is_training, border_mode='same')

		with tf.variable_scope('1x1_Conv_2_1'):
			x2 = conv_block(x, 64, 1, 1, is_training)
		with tf.variable_scope('1x7_Conv_2_2'):
			x2 = conv_block(x2, 64, 1, 7, is_training)
		with tf.variable_scope('7x1_Conv_2_3'):
			x2 = conv_block(x2, 64, 7, 1, is_training)
		with tf.variable_scope('3x3_Conv_2_4'):
			x2 = conv_block(x2, 96, 3, 3, is_training, border_mode='same')

		with tf.variable_scope('filt_concat_2'):
			x = tf.concat(axis=channel_axis, values=[x1, x2])

		with tf.variable_scope('3x3_Conv_3_1'):
			x1 = conv_block(x, 192, 3, 3, is_training, subsample=(2, 2), border_mode='same')
		with tf.variable_scope('max_pool_4_1'):
			x2 = tf.layers.max_pooling2d(x, (3, 3), strides=(2, 2), padding='same')

		with tf.variable_scope('filt_concat_3'):
			x = tf.concat(axis=channel_axis, values=[x1, x2])
		if self.update_endpoint(inputs=x):  # Stage 3
			return x, True
		return x, False

	def build_model(self, inputs, is_training, requested_stages=None):
		super().build_model(inputs, is_training, requested_stages)
		frontend_scope = 'inception_v4'
		with tf.variable_scope(frontend_scope):
			with tf.variable_scope('Stem'):
				x, is_end = self.inception_stem(inputs, is_training)
			if is_end:
				return x, self.end_points, frontend_scope
			# 4 x Inception A
			with tf.variable_scope('4x-Inception-A'):
				for i in range(4):
					x = inception_A(x, is_training, i)

			# Reduction A
			x = reduction_A(x, is_training)
			if self.update_endpoint(inputs=x):  # Stage 4
				return x, self.end_points, frontend_scope
			# 7 x Inception B
			with tf.variable_scope('7x-Inception-B'):
				for i in range(7):
					x = inception_B(x, is_training, i)
			self.update_endpoint(inputs=None)  # Stage 5
			if self.update_endpoint(inputs=x):  # Stage 6
				return x, self.end_points, frontend_scope
			# Reduction B
			x = reduction_B(x, is_training)
			if self.update_endpoint(inputs=x):  # Stage 7
				return x, self.end_points, frontend_scope

			# 3 x Inception C
			with tf.variable_scope('3-xInception-C'):
				for i in range(3):
					x = inception_C(x, is_training, i)
			if self.update_endpoint(inputs=x):  # Stage 8
				return x, self.end_points, frontend_scope

			# Average Pooling
			with tf.variable_scope('Ave_pool'):
				x = tf.layers.average_pooling2d(x, (1, 1), strides=(1, 1), padding='same')

			# Dropout
			x = tf.layers.dropout(x, rate=0.8)

			with tf.variable_scope('fc_layer'):
				x = tf.layers.flatten(x)
				logits = tf.layers.dense(x, self.num_classes, use_bias=True)

		# tf.summary.FileWriter('C:/Users/Onaria Technologies/PycharmProjects/Semantic-Segmentation-Suite/Graph/Train',
		# 					  tf.get_default_graph())

		return logits, self.end_points, frontend_scope


if __name__ == "__main__":
	input_img = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3], name="input_img")
	train_flag = tf.Variable(False, trainable=False, name='train_mode')
	model = InceptionV4()
	inception_v4 = model.build_model(input_img, train_flag)
