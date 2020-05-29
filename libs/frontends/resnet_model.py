# --------------------------------------------------------
# Written by: Rilwan Remilekun Basaru
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import tensorflow as tf

from libs.frontends.frontend_util import MetaFrontEnd, _BATCH_NORM_DECAY, _BATCH_NORM_EPSILON, conv2d_fixed_padding, \
    batch_norm

DEFAULT_VERSION = 2


################################################################################
# ResNet block definitions.
################################################################################
def _building_block_v1(inputs, filters, training, projection_shortcut, strides):
    """A single block for ResNet v1, without a bottleneck.

        Convolution then batch normalization then ReLU as described by:
        Deep Residual Learning for Image Recognition
        https://arxiv.org/pdf/1512.03385.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

        Args:
        inputs: A tensor of size [batch, channels, height_in, width_in] or
          [batch, height_in, width_in, channels] depending on data_format.
        filters: The number of filters for the convolutions.
        training: A Boolean for whether the model is in training or inference
          mode. Needed for batch normalization.
        projection_shortcut: The function to use for projection shortcuts
          (typically a 1x1 convolution when downsampling the input).
        strides: The block's stride. If greater than 1, this block will ultimately
          downsample the input.
        data_format: The input format ('channels_last' or 'channels_first').

        Returns:
        The output tensor of the block; shape should match inputs.
    """

    with tf.variable_scope('building_block_v1'):
        shortcut = inputs

        if projection_shortcut is not None:
            shortcut = projection_shortcut(inputs)
            shortcut = batch_norm(inputs=shortcut, training=training)

        inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=strides)
        inputs = batch_norm(inputs, training)
        inputs = tf.nn.relu(inputs)

        inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=1)
        inputs = batch_norm(inputs, training)
        inputs += shortcut
        inputs = tf.nn.relu(inputs)

    return inputs


def _building_block_v2(inputs, filters, training, projection_shortcut, strides):
    """A single block for ResNet v2, without a bottleneck.

    Batch normalization then ReLu then convolution as described by:
    Identity Mappings in Deep Residual Networks
    https://arxiv.org/pdf/1603.05027.pdf
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

    Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

    Returns:
    The output tensor of the block; shape should match inputs.
    """
    with tf.variable_scope('building_block_v2'):
        shortcut = inputs
        inputs = batch_norm(inputs, training)
        inputs = tf.nn.relu(inputs)

        # The projection shortcut should come after the first batch norm and ReLU
        # since it performs a 1x1 convolution.
        if projection_shortcut is not None:
            shortcut = projection_shortcut(inputs)

        inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=strides)

        inputs = batch_norm(inputs, training)
        inputs = tf.nn.relu(inputs)
        inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=1)

        inputs = inputs + shortcut
    return inputs


def _bottleneck_block_v1(inputs, filters, training, projection_shortcut,
                         strides):
    """A single block for ResNet v1, with a bottleneck.

    Similar to _building_block_v1(), except using the "bottleneck" blocks
    described in:
    Convolution then batch normalization then ReLU as described by:
      Deep Residual Learning for Image Recognition
      https://arxiv.org/pdf/1512.03385.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

    Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

    Returns:
    The output tensor of the block; shape should match inputs.
    """
    with tf.variable_scope('bottleneck_v1'):
        shortcut = inputs

        if projection_shortcut is not None:
            shortcut = projection_shortcut(inputs)
            shortcut = batch_norm(inputs=shortcut, training=training)

        inputs = conv2d_fixed_padding(
          inputs=inputs, filters=filters, kernel_size=1, strides=1)
        inputs = batch_norm(inputs, training)
        inputs = tf.nn.relu(inputs)

        inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=strides)
        inputs = batch_norm(inputs, training)
        inputs = tf.nn.relu(inputs)

        inputs = conv2d_fixed_padding(inputs=inputs, filters=4 * filters, kernel_size=1, strides=1)
        inputs = batch_norm(inputs, training)
        inputs += shortcut
        inputs = tf.nn.relu(inputs)

    return inputs


def _bottleneck_block_v2(inputs, filters, training, projection_shortcut,
                         strides):
    """A single block for ResNet v2, with a bottleneck.

    Similar to _building_block_v2(), except using the "bottleneck" blocks
    described in:
    Convolution then batch normalization then ReLU as described by:
      Deep Residual Learning for Image Recognition
      https://arxiv.org/pdf/1512.03385.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

    Adapted to the ordering conventions of:
    Batch normalization then ReLu then convolution as described by:
      Identity Mappings in Deep Residual Networks
      https://arxiv.org/pdf/1603.05027.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

    Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

    Returns:
    The output tensor of the block; shape should match inputs.
    """
    with tf.variable_scope('bottleneck_v2'):
        shortcut = inputs
        inputs = batch_norm(inputs, training)
        inputs = tf.nn.relu(inputs)

        # The projection shortcut should come after the first batch norm and ReLU
        # since it performs a 1x1 convolution.
        if projection_shortcut is not None:
            shortcut = projection_shortcut(inputs)

        inputs = conv2d_fixed_padding(
          inputs=inputs, filters=filters, kernel_size=1, strides=1, name='conv1', use_bias=False)

        inputs = batch_norm(inputs, training, name='conv1/BatchNorm')
        inputs = tf.nn.relu(inputs)
        inputs = conv2d_fixed_padding(
          inputs=inputs, filters=filters, kernel_size=3, strides=strides, name='conv2', use_bias=False)

        inputs = batch_norm(inputs, training, name='conv2/BatchNorm')
        inputs = tf.nn.relu(inputs)
        inputs = conv2d_fixed_padding(
          inputs=inputs, filters=4 * filters, kernel_size=1, strides=1, name='conv3', use_bias=True)
        inputs = inputs + shortcut
    return inputs


def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides,
                training, name, namescope):
    """Creates one layer of blocks for the ResNet model.

    Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    bottleneck: Is the block created a bottleneck block.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').

    Returns:
    The output tensor of the block layer.
    """
    with tf.variable_scope(namescope):
        # Bottleneck blocks end with 4x the number of filters as they start with
        filters_out = filters * 4 if bottleneck else filters

        def projection_shortcut(inputs):
            return conv2d_fixed_padding(
                inputs=inputs, filters=filters_out, kernel_size=1, strides=strides, name='shortcut', use_bias=True)

        # Only the first block per block_layer uses projection_shortcut and strides
        with tf.variable_scope('unit_1'):
            inputs = block_fn(inputs, filters, training, projection_shortcut, strides)

        for i in range(1, blocks):
            with tf.variable_scope('unit_' + str(i + 1)):
                inputs = block_fn(inputs, filters, training, None, 1)

    return tf.identity(inputs, name)


def _get_block_sizes(resnet_size):
    """Retrieve the size of each block_layer in the ResNet model.

    The number of block layers used for the Resnet model varies according
    to the size of the model. This helper grabs the layer set we want, throwing
    an error if a non-standard size has been selected.

    Args:
    resnet_size: The number of convolutional layers needed in the model.

    Returns:
    A list of block sizes to use in building the model.

    Raises:
    KeyError: if invalid resnet_size is received.
    """
    choices = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [3, 24, 36, 3]
    }

    try:
        return choices[resnet_size]
    except KeyError:
        err = ('Could not find layers for selected Resnet size.\n'
               'Size received: {}; sizes allowed: {}.'.format(
            resnet_size, choices.keys()))
    raise ValueError(err)


class ResNet(MetaFrontEnd):

    def __init__(self, resnet_size, bottleneck, num_classes=2,
                 resnet_version=DEFAULT_VERSION):
        super()

        self.resnet_size = resnet_size
        self.resnet_version = resnet_version

        if resnet_version not in (1, 2):
            raise ValueError(
              'Resnet version should be 1 or 2. See README for citations.')

        self.bottleneck = bottleneck
        if bottleneck:
            if resnet_version == 1:
                self.block_fn = _bottleneck_block_v1
            else:
                self.block_fn = _bottleneck_block_v2
        else:
            if resnet_version == 1:
                self.block_fn = _building_block_v1
            else:
                self.block_fn = _building_block_v2

        self.num_classes = num_classes
        self.num_filters = 64
        self.kernel_size = 7
        self.conv_stride = 2
        self.first_pool_size = 3
        self.first_pool_stride = 2
        self.block_sizes = _get_block_sizes(resnet_size)
        self.block_strides = [1, 2, 2, 2]
        self.pre_activation = resnet_version == 2
        self.end_points = None
        self.lvl = None

    # def update_endpoint(self, inputs, requested_stages):
    #
    #     if requested_stages is not None:
    #         if self.lvl > max(requested_stages):
    #             return True
    #     if self.lvl in requested_stages:
    #         self.end_points['stage' + str(self.lvl)] = inputs
    #     self.lvl += 1
    #     return False

    def build_model(self, inputs, is_training, requested_stages=None):
        """Add operations to classify a batch of input images.

        Args:
          inputs: A Tensor representing a batch of input images.
          is_training: A boolean. Set to True to add operations required only when
            training the classifier.
          requested_stages: A list of requested stages

        Returns:
          A logits Tensor with shape [<batch_size>, self.num_classes].
        """

        super().build_model(inputs, is_training, requested_stages)
        frontend_scope = 'resnet_v' + str(self.resnet_version) + '_' + str(self.resnet_size)
        with tf.variable_scope(frontend_scope):
            # TODO: Consider converting the inputs from NHWC to NCHW to improve GPU performance
            #  See https://www.tensorflow.org/performance/performance_guide

            inputs = conv2d_fixed_padding(inputs=inputs, filters=self.num_filters, kernel_size=self.kernel_size,
                                          strides=self.conv_stride, use_bias=True)
            inputs = tf.identity(inputs, 'initial_conv')

            # We do not include batch normalization or activation functions in V2
            # for the initial conv1 because the first ResNet unit will perform these
            # for both the shortcut and non-shortcut paths as part of the first
            # block's projection. Cf. Appendix of [2].
            if self.resnet_version == 1:
                inputs = batch_norm(inputs, is_training)
                inputs = tf.nn.relu(inputs)
            if self.update_endpoint(inputs):  # Stage 1
                return inputs, self.end_points, frontend_scope
            if self.first_pool_size:
                inputs = tf.layers.max_pooling2d(
                    inputs=inputs, pool_size=self.first_pool_size,
                    strides=self.first_pool_stride, padding='SAME',
                    data_format='channels_last')
                inputs = tf.identity(inputs, 'initial_max_pool')

            for i, num_blocks in enumerate(self.block_sizes):
                num_filters = self.num_filters * (2**i)
                inputs = block_layer(
                    inputs=inputs, filters=num_filters, bottleneck=self.bottleneck,
                    block_fn=self.block_fn, blocks=num_blocks,
                    strides=self.block_strides[i], training=is_training,
                    name='block_layer{}'.format(i + 1), namescope="block" + str(i + 1))

                if self.update_endpoint(inputs):    # Stage 2 - 5
                    return inputs, self.end_points, frontend_scope

            if self.update_endpoint(None):  # Stage 6
                return inputs, self.end_points, frontend_scope

            # Only apply the BN and ReLU for model that does pre_activation in each
            # building/bottleneck block, eg resnet V2.
            if self.pre_activation:
                inputs = batch_norm(inputs, is_training)
                inputs = tf.nn.relu(inputs)
                if self.update_endpoint(inputs):  # Stage 7
                    return inputs, self.end_points, frontend_scope
            else:
                if self.update_endpoint(None):  # Stage 7
                    return inputs, self.end_points, frontend_scope
            # The current top layer has shape
            # `batch_size x pool_size x pool_size x final_size`.
            # ResNet does an Average Pooling layer over pool_size,
            # but that is the same as doing a reduce_mean. We do a reduce_mean
            # here because it performs better than AveragePooling2D.
            axes = [1, 2]
            inputs = tf.reduce_mean(input_tensor=inputs, axis=axes, keepdims=True)
            inputs = tf.identity(inputs, 'final_reduce_mean')

            inputs = tf.squeeze(inputs, axes)
            inputs = tf.layers.dense(inputs=inputs, units=self.num_classes)
            inputs = tf.identity(inputs, 'final_dense')
            self.update_endpoint(inputs)    # Stage 8

            return inputs, self.end_points, frontend_scope


if __name__ == "__main__":
    input_img = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3], name="input_img")
    train_flag = tf.Variable(False, trainable=False, name='train_mode')
    model = ResNet(resnet_size=152, bottleneck=True, resnet_version=2)
    inception_v4 = model.build_model(input_img, train_flag)
