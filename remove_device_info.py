import os
import tensorflow as tf


PATH_TO_CKPT = os.path.join('Networks', 'net.pb')

graph = tf.Graph()
with graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')

	input_graph_def = graph.as_graph_def()
	for node in input_graph_def.node:
		print(node.device)
		node.device = ""
	with tf.gfile.GFile("net.pb", "wb") as f:
		f.write(input_graph_def.SerializeToString())
