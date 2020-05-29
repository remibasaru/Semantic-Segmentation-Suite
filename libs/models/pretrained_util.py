# --------------------------------------------------------
# Written by Rilwan Basaru
# --------------------------------------------------------
import sys

import tensorflow as tf


def load_pretrain_model(model_path, var_list, ignore_missing_vars=True):

	model = tf.train.NewCheckpointReader(model_path)
	model_map = model.get_variable_to_shape_map()
	updated_var_list = []
	if ignore_missing_vars:
		for var in var_list:
			var_name = var.name.split(':')[0]
			if var_name in model_map:
				if model_map[var_name] == var.shape:
					updated_var_list.append(var)
				else:
					print('"' + var_name + '" with shape : ' + str(tuple(model_map[var_name])) + ' and '
						  + str(var.shape) + ' not compatible.')
	else:
		updated_var_list = var_list

	saver = tf.train.Saver(updated_var_list)

	def callback(session):
		saver.restore(session, model_path)
	return callback
