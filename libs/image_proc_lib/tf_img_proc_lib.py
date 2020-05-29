import os
import numpy as np
import tensorflow as tf

from libs.image_proc_lib.np_img_proc_lib import prepare_image


def bw_thresh(in_prob, thresh=0.68, name_scope="prob2bw"):
	with tf.variable_scope(name_scope):
		pred_bool = tf.math.greater_equal(in_prob, tf.constant(thresh))
		pred_bool = tf.cast(pred_bool, tf.uint8)
		# pred_bool * tf.constant(1, dtype=tf.uint8)
	return pred_bool


def consolidate_patches(composite_img, composite_img_count, name_scope="consolidate"):
	with tf.variable_scope(name_scope):
		composite_img = tf.cast(composite_img, tf.float32)
		composite_img_count = tf.cast(composite_img_count, tf.float32)
		composite_img = tf.reduce_sum(composite_img, axis=1)
		composite_img_count = tf.reduce_sum(composite_img_count, axis=1)
		pred_prob = tf.divide(composite_img, composite_img_count)
		return pred_prob


def img_to_patches(raw_input, _patch_size=(128, 128), _stride=100):

	with tf.variable_scope('im2_patches'):
		patches = tf.image.extract_image_patches(
			images=raw_input,
			ksizes=[1, _patch_size[0], _patch_size[1], 1],
			strides=[1, _stride, _stride, 1],
			rates=[1, 1, 1, 1],
			padding='SAME'
		)

		h = tf.shape(patches)[1]
		w = tf.shape(patches)[2]
		patches = tf.reshape(patches, (patches.shape[0], -1, _patch_size[0], _patch_size[1], 3))
	return patches, (h, w)


def dummy_predict(in_img):
	patched_prediction = tf.divide(in_img[:, :, :, 0], 255.0)
	return patched_prediction


def patches_to_img(update, _block_shape, _input_img_shape=None, _patch_size=(128, 128), _stride=100):
	with tf.variable_scope('patches2im'):
		_h = _block_shape[0]
		_w = _block_shape[1]
		wout = (_w - 1) * _stride + _patch_size[1]
		hout = (_h - 1) * _stride + _patch_size[0]

		x, y = tf.meshgrid(list(range(_patch_size[1])), list(range(_patch_size[0])))
		x = tf.reshape(x, (1, _patch_size[0], _patch_size[1], 1))
		y = tf.reshape(y, (1, _patch_size[0], _patch_size[1], 1))
		xstart, ystart = tf.meshgrid(tf.range(0, (wout - _patch_size[1]) + 1, _stride),
									 tf.range(0, (hout - _patch_size[0]) + 1, _stride))

		yy = y + tf.reshape(ystart, (-1, 1, 1, 1))
		xx = x + tf.reshape(xstart, (-1, 1, 1, 1))
		dd = tf.zeros((1, _patch_size[0], _patch_size[1], 1), dtype=tf.int32) + tf.reshape(tf.range(_w * _h), (-1, 1, 1, 1))
		idx = tf.concat([yy, xx, dd], -1)

		composite_img = tf.scatter_nd(idx, update, (hout, wout, _w * _h))
		composite_img = tf.transpose(composite_img, (2, 0, 1))

		composite_img_count = tf.scatter_nd(idx, tf.ones_like(update), (hout, wout, _w * _h))
		composite_img_count = tf.transpose(composite_img_count, (2, 0, 1))
		if _input_img_shape is not None:
			with tf.variable_scope('crop'):
				off_h = tf.cast(tf.floor(tf.cast(hout - _input_img_shape[0], tf.float32) / 2.0), tf.int32)
				off_w = tf.cast(tf.floor(tf.cast(wout - _input_img_shape[1], tf.float32) / 2.0), tf.int32)
				composite_img = composite_img[:, off_h: off_h + _input_img_shape[0], off_w: off_w + _input_img_shape[1]]
				composite_img_count = composite_img_count[:, off_h: off_h + _input_img_shape[0], off_w: off_w + _input_img_shape[1]]

		with tf.variable_scope("consolidate"):
			composite_img = tf.reduce_sum(composite_img, axis=0)
			composite_img_count = tf.reduce_sum(composite_img_count, axis=0)
			pred_prob = tf.divide(composite_img, composite_img_count)
			pred_prob = tf.cast(pred_prob, tf.float32)
		return pred_prob, composite_img


if __name__ == "__main__":

	border = 200
	resizeFactor = 0.2800
	filename = os.path.join('sample_img.jpg')
	Im = prepare_image(filename, resizeFactor, border)
	Im = np.expand_dims(Im, axis=0)

	patch_size = (128, 128)
	stride = 150

	input_img = tf.placeholder(dtype=tf.float32, shape=(1, None, None, 3), name="input_img")

	extracted_patches, block_shape = img_to_patches(input_img, _patch_size=patch_size, _stride=stride)

	patchified_prediction = dummy_predict(extracted_patches[0])

	raw_input_shape = tf.shape(input_img)
	input_shape = (raw_input_shape[1], raw_input_shape[2])
	pred, pred0 = patches_to_img(patchified_prediction, block_shape, _input_img_shape=input_shape, _patch_size=patch_size, _stride=stride)

	with tf.Session() as sess:
		arg0, arg1, arg2, arg3 = sess.run([pred, pred0, patchified_prediction, extracted_patches], feed_dict={input_img: Im})
		print(arg0)
