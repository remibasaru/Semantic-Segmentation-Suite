import tensorflow as tf
import numpy as np

from libs.image_proc_lib.np_img_proc_lib import patches2im, im2patches


def bw_thresh(in_prob, thresh=0.68, name_scope="prob2bw"):
    with tf.variable_scope(name_scope):
        pred_bool = tf.math.greater_equal(in_prob, tf.constant(thresh))
        pred_bool = tf.cast(pred_bool, tf.uint8)
        # pred_bool * tf.constant(1, dtype=tf.uint8)
    return pred_bool


def patches_to_comp_img(patches_tensor, patches_shape, stride=10, name_scope="patches2im"):
    with tf.variable_scope(name_scope):
        all_comp_img = []
        all_comp_img_count = []
        for i_y in tf.unstack(patches_tensor):
            comp_img, comp_img_count = tf.py_func(patches2im, [i_y, patches_shape, stride], [tf.uint8, tf.uint8])

            all_comp_img.append(comp_img)
            all_comp_img_count.append(comp_img_count)
        all_comp_img = tf.stack(all_comp_img, axis=0)
        all_comp_img_count = tf.stack(all_comp_img_count, axis=0)
        pred_prob = consolidate_patches(all_comp_img, all_comp_img_count)
        return all_comp_img, all_comp_img_count, pred_prob


def consolidate_patches(composite_img, composite_img_count, name_scope="consolidate"):
    with tf.variable_scope(name_scope):
        composite_img = tf.cast(composite_img, tf.float32)
        composite_img_count = tf.cast(composite_img_count, tf.float32)
        composite_img = tf.reduce_sum(composite_img, axis=1)
        composite_img_count = tf.reduce_sum(composite_img_count, axis=1)
        pred_prob = tf.divide(composite_img, composite_img_count)
        return pred_prob


def img_to_patches(input_tensor, patch_win_size=(21, 21), stride=10, name_scope="im2patches"):
    with tf.variable_scope(name_scope):
        returned_shape = None
        all_patches = []
        for i_x in tf.unstack(input_tensor):
            patches, _ = tf.py_func(im2patches, [i_x, patch_win_size, stride], [tf.uint8, tf.uint8])
            patches_shape = tf.shape(patches)

            reshaped_patches = tf.reshape(patches, (-1, patch_win_size[0], patch_win_size[1], input_tensor.shape[-1]))
            all_patches.append(reshaped_patches)
            returned_shape = (patches_shape[0], patches_shape[1])
        all_patches = tf.stack(all_patches, axis=0)

    return all_patches, returned_shape


if __name__ == '__main__':
    batch_size = 2
    x = tf.placeholder(tf.uint8, shape=(batch_size, None, None, 3))

    with tf.Session() as sess:

        rand_array = np.random.rand(batch_size, 200, 100, 3) * 255
        rand_array = rand_array.astype(np.uint8)

        patches, patch_shape = img_to_patches(x)
        cI, cI_count, pb = patches_to_comp_img(patches[:, :, :, :, 0], patch_shape)

        patch_arr, s, count, pb, pbt = (sess.run([patches, cI, cI_count, pb], feed_dict={x: rand_array}))

    f = rand_array[0, :, :, 0]
    f1 = s[0, 0, :, :, 0]
    f2 = s[0, 1, :, :, 0]
    f3 = s[0, 2, :, :, 0]
    pb = pb[0, :, :, 0]

