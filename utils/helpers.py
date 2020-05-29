import tensorflow as tf


def pad_up_to(t, max_in_dims, constant_values):
    with tf.name_scope("pad"):
        s = tf.shape(t)
        paddings = [[0, 0], [0, max_in_dims[1] - s[1]], [0, max_in_dims[2] - s[2]], [0, 0]]
        t = tf.pad(t, paddings, 'CONSTANT', constant_values=constant_values)
    return t

