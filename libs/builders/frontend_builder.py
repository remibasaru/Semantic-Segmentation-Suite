# --------------------------------------------------------
# Written by: Rilwan Remilekun Basaru
# --------------------------------------------------------

from libs.frontends import resnet_model

import os
import tensorflow as tf
from libs.frontends.inception_v4 import InceptionV4
from libs.frontends.mobilenet_v2 import MobileNetV2
from libs.models import pretrained_util

FRONT_END_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')


def get_stable_params():
	stable_params = tf.trainable_variables()
	# return [param.name for param in stable_params]
	return stable_params


def build_frontend(inputs, frontend, is_training, pre_trained_dir=None, load_pre_train=False,
				   requested_stages=None):
	init_fn = None
	if pre_trained_dir is None:
		pre_trained_dir = FRONT_END_MODEL_PATH
	stable_params = []
	if frontend == 'ResNet50':
		model = resnet_model.ResNet(resnet_size=50, bottleneck=True, resnet_version=2)
		logits, end_points, frontend_scope = model.build_model(inputs, is_training, requested_stages)
		if load_pre_train:
			pre_trained_path = os.path.join(pre_trained_dir, 'res_net', '50', 'resnet_v2_50.ckpt')
			stable_params = get_stable_params()
			init_fn = pretrained_util.load_pretrain_model(pre_trained_path, var_list=stable_params,
														  ignore_missing_vars=True)

	elif frontend == 'ResNet101':
		model = resnet_model.ResNet(resnet_size=101, bottleneck=True, resnet_version=2)
		logits, end_points, frontend_scope = model.build_model(inputs, is_training, requested_stages)
		if load_pre_train:
			pre_trained_path = os.path.join(pre_trained_dir, 'res_net', '101', 'resnet_v2_101.ckpt')
			stable_params = get_stable_params()
			init_fn = pretrained_util.load_pretrain_model(pre_trained_path, var_list=stable_params,
														  ignore_missing_vars=True)

	elif frontend == 'ResNet152':
		model = resnet_model.ResNet(resnet_size=152, bottleneck=True, resnet_version=2)
		logits, end_points, frontend_scope = model.build_model(inputs, is_training, requested_stages)
		if load_pre_train:
			pre_trained_path = os.path.join(pre_trained_dir, 'res_net', '152', 'resnet_v2_152.ckpt')
			stable_params = get_stable_params()
			init_fn = pretrained_util.load_pretrain_model(pre_trained_path, var_list=stable_params,
														  ignore_missing_vars=True)

	elif frontend == 'ResNet200':
		model = resnet_model.ResNet(resnet_size=200, bottleneck=True, resnet_version=2)
		logits, end_points, frontend_scope = model.build_model(inputs, is_training, requested_stages)
		if load_pre_train:
			# TODO: Implement pretrained model for Resnet 200
			stable_params = get_stable_params()
			print('No pre-trained model for ResNet200')

	elif frontend == 'MobileNetV2':
		model = MobileNetV2()
		logits, end_points, frontend_scope = model.build_model(inputs, is_training, requested_stages)
		if load_pre_train:
			pre_trained_path = os.path.join(pre_trained_dir, 'mobilenet_v2', 'mobilenet_v2.ckpt')
			stable_params = get_stable_params()
			init_fn = pretrained_util.load_pretrain_model(model_path=pre_trained_path, var_list=stable_params,
														  ignore_missing_vars=True)
	elif frontend == 'InceptionV4':
		model = InceptionV4()
		logits, end_points, frontend_scope = model.build_model(inputs, is_training, requested_stages)
		if load_pre_train:
			pre_trained_path = os.path.join(pre_trained_dir, 'inceptionnet', 'inception_v4.ckpt')
			stable_params = get_stable_params()
			init_fn = pretrained_util.load_pretrain_model(model_path=pre_trained_path, var_list=stable_params,
														  ignore_missing_vars=True)

	# TODO: Implement front end for Xception net
	elif frontend == 'Xception':
		raise NotImplementedError
	else:
		raise ValueError("Unsupported fronetnd model '%s'. This function only "
						 "supports ResNet50, ResNet101, ResNet152, and MobileNetV2" % (frontend))

	# tf.summary.FileWriter('C:/Users/Onaria Technologies/PycharmProjects/Semantic-Segmentation-Suite/Graph/Sample',
	# 					  tf.get_default_graph())
	return logits, end_points, frontend_scope, init_fn, stable_params


if __name__ == '__main__':

	tf.reset_default_graph()
	input_img = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3], name="input_img")
	train_flag = tf.Variable(False, trainable=False, name='train_mode')
	build_frontend(inputs=input_img, frontend='ResNet50', is_training=train_flag, requested_stages=None,
				   load_pre_train=True)


