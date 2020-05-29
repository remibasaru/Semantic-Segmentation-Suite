from libs.frontends.frontend_util import MetaFrontEnd
from libs.frontends.mobilenet_util import res_block
from libs.frontends.mobilenet_util import pwise_block
from libs.frontends.mobilenet_util import conv2d_block, global_avg, flatten, conv_1x1
import tensorflow as tf


class MobileNetV2(MetaFrontEnd):

	def build_model(self, inputs, is_training, requested_stages=None):
		super().build_model(inputs, is_training, requested_stages)
		frontend_scope = 'mobilenet_v2'
		exp = 6  # expansion ratio
		with tf.variable_scope(frontend_scope):
			net = conv2d_block(inputs, 32, 3, 2, is_training, name='conv1_1')  # size/2
			if self.update_endpoint(net):  # Stage 1
				return net, self.end_points, frontend_scope
			net = res_block(net, 1, 16, 1, is_training, name='res2_1', is_pw=False)

			net = res_block(net, exp, 24, 2, is_training, name='res3_1')  # size/4
			if self.update_endpoint(net):  # Stage 2
				return net, self.end_points, frontend_scope
			net = res_block(net, exp, 24, 1, is_training, name='res3_2')

			net = res_block(net, exp, 32, 2, is_training, name='res4_1')  # size/8
			if self.update_endpoint(net):  # Stage 3
				return net, self.end_points, frontend_scope
			net = res_block(net, exp, 32, 1, is_training, name='res4_2')
			net = res_block(net, exp, 32, 1, is_training, name='res4_3')

			net = res_block(net, exp, 64, 2, is_training, name='res5_1')
			if self.update_endpoint(net):  # Stage 4
				return net, self.end_points, frontend_scope
			net = res_block(net, exp, 64, 1, is_training, name='res5_2')
			net = res_block(net, exp, 64, 1, is_training, name='res5_3')
			net = res_block(net, exp, 64, 1, is_training, name='res5_4')

			self.update_endpoint(None)  # Stage 5

			net = res_block(net, exp, 96, 1, is_training, name='res6_1')  # size/16
			if self.update_endpoint(net):  # Stage 6
				return net, self.end_points, frontend_scope
			net = res_block(net, exp, 96, 1, is_training, name='res6_2')
			net = res_block(net, exp, 96, 1, is_training, name='res6_3')

			net = res_block(net, exp, 160, 2, is_training, name='res7_1')  # size/32
			if self.update_endpoint(net):  # Stage 7
				return net, self.end_points, frontend_scope
			net = res_block(net, exp, 160, 1, is_training, name='res7_2')
			net = res_block(net, exp, 160, 1, is_training, name='res7_3')

			net = res_block(net, exp, 320, 1, is_training, name='res8_1', shortcut=False)
			if self.update_endpoint(net):  # Stage 8
				return net, self.end_points, frontend_scope

			net = pwise_block(net, 1280, is_training, name='conv9_1', scale=True)
			net = global_avg(net)
			logits = flatten(conv_1x1(net, self.num_classes, name='logits'))

			return logits, self.end_points, frontend_scope


if __name__ == "__main__":
	input_img = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3], name="input_img")
	train_flag = tf.Variable(False, trainable=False, name='train_mode')
	model = MobileNetV2()
	inception_v4 = model.build_model(input_img, train_flag)
