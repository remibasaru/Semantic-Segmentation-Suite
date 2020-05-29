# --------------------------------------------------------
# Written by: Rilwan Remilekun Basaru
# --------------------------------------------------------

from __future__ import print_function

import pickle

from Trainer import cnn_train, update_session_with_optimum_epoch_state
import os
import tensorflow as tf

from dataset import DataLoader
from libs.builders.model_builder import Network
import numpy as np

if __name__ == "__main__":

    opts = {}
    param = dict()

    opts['expDir'] = os.path.join("Exp", "Teeth-White-Exp")
    opts['gpu'] = True
    opts['batchSize'] = 60
    opts['numEpochs'] = 5
    opts['trainPerImage'] = True

    opts['learningRate'] = 1e-6
    opts['weightDecay'] = 0.0005
    opts['momentum'] = 0.09
    opts['saveMomentum'] = True
    opts['continue'] = None  # set which epoch to continue from. Set to None to be ignored.

    opts['data_loader_train'] = DataLoader([0, 1, 2, 3, 4, 5, 6])
    opts['data_loader_val'] = DataLoader([7, 8, 9])

    param['mean_pixel_val'] = np.array([0.0, 0.0, 0.0])
    param['std_pixel_val'] = np.array([1.0, 1.0, 1.0])

    param['filename'] = os.path.join('.', 'Networks', 'net.pb')
    param['graphPathTrain'] = os.path.join('.', "Graph", 'Train')
    param['graphPathTest'] = os.path.join('.', "Graph", 'Test')
    param['gpu'] = opts['gpu']
    param['batchSize'] = opts['batchSize']
    param['debug'] = False
    param['applyWeight'] = False
    param['load_pre_train'] = False
    param['apply_resize'] = True

    opts['DeviceAgnostic'] = True  # remove device information from frozen graph

    net = Network(param, model_name='SmallUNet')
    # net = Network(param, model_name='DeepLabV3_plus', frontend='ResNet50')
    # net = Network(param, model_name='DeepLabV3_plus', frontend='ResNet101')
    net.build_model()
    net.add_optimizer(opts)
    net.saver = tf.train.Saver()
    net, sess, stats = cnn_train(net, opts)

    # Harvest all trained weights, convert variable tensors to constants and save
    sess = update_session_with_optimum_epoch_state(opts['expDir'], sess, net.saver)
    output_graph_def = net.freeze_net(sess, opts['DeviceAgnostic'])
    with tf.gfile.GFile(param['filename'], "wb") as f:
        f.write(output_graph_def.SerializeToString())

    sess.close()
