# --------------------------------------------------------
# Written by Rilwan Basaru
# --------------------------------------------------------

from __future__ import print_function
import os
import re
import errno

import numpy as np
import random


from dataset import DataLoader
import tensorflow as tf
import pickle
from utils.timer import Timer
import matplotlib.pyplot as plt


def arg_parse(opts, user_opts):
    for key in user_opts:
        opts[key] = user_opts[key]
    return opts


def find_last_checkpoint(modelDir):
    epoch = 0
    for f in os.listdir(os.path.join('.', modelDir)):
        if re.match(r'net-epoch-\d+', f):
            tmp_epoch = int((re.search(r'\d+', f)).group(0))
            if tmp_epoch > epoch:
                epoch = tmp_epoch
    return epoch


def load_state(model_folder_path, model_path, saver, sess):
    saver.restore(sess, model_path)
    with open(os.path.join(model_folder_path, 'stats.pickle'), 'rb') as handle:
        stats = pickle.load(handle)

    return stats, sess


def update_session_with_optimum_epoch_state(modelDir, sess, saver, loss_name='loss'):
    last_epoch_id = find_last_checkpoint(modelDir)

    epoch_stats_path = os.path.join(modelDir, 'net-epoch-' + str(last_epoch_id), 'stats.pickle')
    epoch_stats = pickle.load(open(epoch_stats_path, "rb"))
    min_loss = np.Inf
    min_loss_ep_id = -1
    for ep_id in range(last_epoch_id):
        ep_loss_ave = epoch_stats['val'][ep_id][loss_name]['average']
        if ep_loss_ave < min_loss:
            min_loss = ep_loss_ave
            min_loss_ep_id = ep_id
    min_loss_ep_path = os.path.join(modelDir, 'net-epoch-' + str(min_loss_ep_id + 1))
    if min_loss_ep_id >= 0:
        _, sess = load_state(min_loss_ep_path, os.path.join(min_loss_ep_path, 'model.ckpt'), saver, sess)
    return sess


def process_epoch(net, state, params, timer, mode, training=True):

    epoch = params['epoch']

    stats = dict()
    if not state:
        state['stats'] = dict()
        # state['thresh'] = dict()

    for l in net.losses.keys():
        stats[l] = {'count': 0, 'average': 0}

    batch_size = params['batchSize']
    sess = params['session']

    ite = 1
    data_loader = params['data_loader_' + mode]
    data_loader.reset_ite()

    # Keep training until reach max iterations
    while data_loader.is_next():
        timer.tic()
        disp = mode + ': epoch ' + str(epoch) + ':   ' + str(data_loader.ite + 1) + '/' + str(data_loader.size) + ': '

        batch_x, batch_y, weight = data_loader.next_img(batch_size)

        blobs = {"input_img": batch_x, "output_img": batch_y, "pos_weight": weight}
        if training:
            stats = net.train_step(sess, blobs, stats, mode)
        else:
            stats = net.test_and_display_step(sess, blobs, stats)

        timer.toc()

        average_speed = timer.average_time
        if not params['silent']:
            print(disp + "{0:0.2f}".format(average_speed) + '(' + "{0:0.2f}".format(timer.diff) + ') Hz objective: ' +
                  "{0:0.4f}".format(stats["loss"]['average']))

        ite += 1

    # Save back to state
    state['stats'][mode] = stats
    state['sess'] = sess
    # state['thresh'] = thresh

    return net, state


def save_state(save_path_folder, save_path, net, state, silent=False):
    try:
        os.mkdir(save_path_folder)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    save_path = net.saver.save(state['sess'], save_path)
    if not silent:
        print("Model saved in path: %s" % save_path)
    return True


def save_stats(stats_path, stats):
    # TO DO: Save stats to pickle
    with open(os.path.join(stats_path, 'stats.pickle'), 'wb') as handle:
        pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return None


def plot_stats(fig, stats, model_figure_path):
    fig.clf()
    train_objective, val_objective, epoch_count = transpose_stats(stats)

    num_var = len(train_objective)
    dim = np.ceil(num_var ** .5)
    dim = int(dim * 100 + dim * 10)
    idx = np.arange(epoch_count).astype(np.int) + 1

    for i, vars in enumerate(train_objective.keys()):
        plt.subplot(dim + i + 1)

        plt.plot(idx, val_objective[vars], 'ro-', label='val')
        plt.plot(idx, train_objective[vars], 'bo-', label='train')

        plt.title(vars)
        plt.xlabel('epoch')
        plt.grid(True)
        plt.legend()

    plt.show(block=False)
    fig.canvas.draw()

    plt.savefig(model_figure_path)
    return None


def transpose_stats(stats):
    train_stats = stats['train']
    val_stats = stats['val']
    train_objective = {}
    val_objective = {}
    epoch_count = len(val_stats)

    for idx, train_stat in enumerate(train_stats):
        val_stat = val_stats[idx]
        for objective in train_stat.keys():
            if objective not in train_objective:
                train_objective[objective] = []
            if objective not in val_objective:
                val_objective[objective] = []
            train_objective[objective].append(train_stat[objective]['average'])
            val_objective[objective].append(val_stat[objective]['average'])

    return train_objective, val_objective, epoch_count


def cnn_train(net, user_opts, is_training=True):

    opts = dict()
    opts['expDir'] = 'Exp'
    opts['continue'] = -1
    opts['batchSize'] = 10
    opts['numSubBatches'] = 1
    opts['train'] = []
    opts['val'] = []
    opts['gpu'] = []
    opts['numEpochs'] = 5
    opts['learningRate'] = 0.001
    opts['weightDecay'] = 0.0005
    opts['momentum'] = 0.9
    opts['randomSeed'] = 0
    opts['folder_map'] = []
    opts['test_data_idx'] = []
    opts['train_data_idx'] = []
    opts['plotStatistics'] = True
    opts['silent'] = False

    opts = arg_parse(opts, user_opts)
    if 'data_loader_train' not in opts or 'data_loader_val' not in opts:
        opts['data_loader_train'] = DataLoader(opts['train_data_idx'], opts['teethRegionDetector'])
        opts['data_loader_val'] = DataLoader(opts['test_data_idx'], opts['teethRegionDetector'])

    if not os.path.isdir(opts['expDir']):
        os.mkdir(opts['expDir'])
    evaluate_mode = False

    def model_path(ep):
        return os.path.join(opts['expDir'], 'net-epoch-' + str(ep), 'model.ckpt')

    def model_folder_path(ep):
        return os.path.join(opts['expDir'], 'net-epoch-' + str(ep))

    model_figure_path = os.path.join(opts['expDir'], 'net-train.pdf')
    fig = plt.gcf()
    fig.show()
    fig.canvas.draw()
    sess = tf.Session(config=net.config)
    if opts['continue'] is not None:
        prev_pos_1 = max(0, min(opts['continue'], find_last_checkpoint(opts['expDir'])))
    else:
        prev_pos_1 = max(0, find_last_checkpoint(opts['expDir']))

    start_1 = prev_pos_1 + 1
    if prev_pos_1 >= 1:
        print('Resuming by loading epoch', str(prev_pos_1))
        stats, sess = load_state(model_folder_path(prev_pos_1), model_path(prev_pos_1), net.saver, sess)

        if sess is None or stats is None:
            sess.run(tf.global_variables_initializer())
            stats = dict()
            stats['train'] = []
            stats['val'] = []
            print('Failed to load. Starting with epoch ', str(start_1), '\n')
        else:
            print('Continuing at epoch ', str(start_1), '\n')
    else:
        sess.run(tf.global_variables_initializer())
        if net.load_pre_trained_model_func is not None:
            net.load_pre_trained_model_func(sess)

        stats = dict()
        stats['train'] = []
        stats['val'] = []
        print('Starting at epoch ', str(start_1), '\n')

    state = dict()
    opts['session'] = sess

    timer = Timer()
    for ep in range(start_1 - 1, opts['numEpochs']):
        # Set the random seed based on the epoch and opts.randomSeed.
        # This is important for reproducibility, including when training
        # is restarted from a checkpoint.
        epoch = ep + 1
        print('     Epoch # ' + str(epoch))
        random.seed(epoch + opts['randomSeed'])

        # Train for one epoch
        params = opts
        params['epoch'] = epoch

        if is_training:
            [net, state] = process_epoch(net, state, params, timer, 'train')
            [net, state] = process_epoch(net, state, params, timer, 'val')
        else:
            [net, state] = process_epoch(net, state, params, timer, 'val', training=False)

        if not evaluate_mode:
            save_state(model_folder_path(epoch), model_path(epoch), net, state, opts['silent'])
        last_stats = state['stats']

        stats['train'].append(last_stats['train'])
        stats['val'].append(last_stats['val'])

        save_stats(model_folder_path(epoch), stats)

        if opts['plotStatistics']:
            plot_stats(fig, stats, model_figure_path)

    plt.close(fig)
    return net, sess, stats
