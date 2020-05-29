# --------------------------------------------------------
# Written by Rilwan Basaru
# --------------------------------------------------------

import tensorflow as tf
import os

# PATH_TO_CKPT = os.path.join( 'Networks', 'net.pb')
PATH_TO_CKPT = os.path.join('Networks', 'teeth-label-net.pb')
# PATH_TO_CKPT = os.path.join('Networks', 'teeth-label-net-100.pb')
PATH_TO_CKPT = os.path.join('Networks', 'teeth-label-net-101.pb')
# PATH_TO_CKPT = os.path.join('Exp_2', 'net.pb')
# PATH_TO_CKPT = os.path.join('Networks', 'ssd_mobilenet_v1_android_export.pb')
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

with tf.Session(graph=detection_graph) as sess:
    f = sess.run(detection_graph.get_tensor_by_name("normalize/mean_pixel_val_1:0"))
    tf.summary.FileWriter('Graph\Sample', sess.graph)
