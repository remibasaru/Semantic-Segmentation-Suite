# ----------------------------------------------------
# Written by Rilwan Remilekun Basaru
# ----------------------------------------------------

# Adapted from https://github.com/GeorgeSeif/Semantic-Segmentation-Suite.git

import os
import tensorflow as tf
import numpy as np
import cv2

from libs.models_utils import UNet
from libs.models_utils.UNet import build_unet, build_small_unet

from libs.models_utils.FC_DenseNet_Tiramisu import build_fc_densenet
from libs.models_utils.Encoder_Decoder import build_encoder_decoder
from libs.models_utils.RefineNet import build_refinenet
from libs.models_utils.FRRN import build_frrn
from libs.models_utils.MobileUNet import build_mobile_unet
from libs.models_utils.PSPNet import build_pspnet
from libs.models_utils.GCN import build_gcn
from libs.models_utils.DeepLabV3 import build_deeplabv3
from libs.models_utils.DeepLabV3_plus import build_deeplabv3_plus
from libs.models_utils.AdapNet import build_adaptnet
from libs.models_utils.custom_model import build_custom
from libs.models_utils.DenseASPP import build_dense_aspp
from libs.models_utils.DDSC import build_ddsc
from libs.models_utils.BiSeNet import build_bisenet

SUPPORTED_MODELS = ["FC-DenseNet34", "FC-DenseNet45", "FC-DenseNet58", "FC-DenseNet56", "FC-DenseNet67", "FC-DenseNet103",
                    "UNet", "Encoder-Decoder", "Encoder-Decoder-Skip",
                    "RefineNet", "FRRN-A", "FRRN-B", "MobileUNet", "MobileUNet-Skip", "PSPNet", "GCN", "DeepLabV3",
                    "DeepLabV3_plus", "AdapNet", "DenseASPP", "DDSC", "BiSeNet", "SmallUNet", "custom"]

SUPPORTED_FRONTENDS = ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "ResNet200", "MobileNetV2", "InceptionV4"]


# TO DO: Implement windows auto download of needed modules
def download_checkpoints(model_name):
    NotImplementedError("Download the model: " + model_name + '. See "utils/get_pretrained_checkpoints.py"')
    # subprocess.check_output(["python", "utils/get_pretrained_checkpoints.py", "--model=" + model_name])


class Network:
    def __init__(self, param, model_name="FC-DenseNet56", num_classes=2, frontend="ResNet50"):

        # self.train_flag = None
        self.graph_path = param['graphPathTrain']
        self.graph_pred_path = param['graphPathTest']
        self.load_pre_train = param['load_pre_train']
        self.saver = None
        self.model_graph = None
        self.model_name = model_name
        self.num_classes = num_classes
        self.frontend = frontend
        self.target_img = None
        self.input_img = None
        self.weight = None
        self.is_normalised = False
        self.apply_resize = param['apply_resize']
        if 'mean_pixel_val' in param and 'mean_pixel_val' in param:
            self.mean_pixel_val = param['mean_pixel_val']
            self.std_pixel_val = param['std_pixel_val']
            self.is_normalised = True
        else:
            self.mean_pixel_val = None
            self.std_pixel_val = None
        self.layers = {}
        self.batch_size = param['batchSize']
        self.handle_admin(param)
        self.train_op = None
        self.probe = {}
        self.train_flag = None
        self.gpu = param['gpu']
        self.losses = {}
        self.output_node_names = []
        self.config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        self.load_pre_trained_model_func = None
        self.stable_trainable_tensors = []  # tensors that are updated at a smaller learning rate

    def handle_admin(self, param):
        self.input_img = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3], name="input_img")
        self.target_img = tf.placeholder(dtype=tf.float32, shape=[None, None, None, self.num_classes],
                                         name="target_img")
        if param['applyWeight']:
            self.weight = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128], name="weight")

        # self.input_img = tf.placeholder(dtype=tf.float32, shape=[None, 900, 500, 3],
        #                                 name="input_img")
        # self.target_img = tf.placeholder(dtype=tf.float32, shape=[None, 900, 500, self.num_classes],
        #                                 name="target_img")

        if not param['debug']:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        self.gpu = param['gpu']

        if not os.path.isdir(param['graphPathTrain']):
            os.mkdir(param['graphPathTrain'])
        if not os.path.isdir(param['graphPathTest']):
            os.mkdir(param['graphPathTest'])

    @staticmethod
    def handle_frontend(frontend):
        if frontend not in SUPPORTED_FRONTENDS:
            raise ValueError(
                "The frontend you selected is not supported. The following models are currently supported: {0}".format(
                    SUPPORTED_FRONTENDS))

        if "ResNet50" == frontend and not os.path.isfile("models/res_net/resnet_v2_50.ckpt"):
            download_checkpoints("ResNet50")
        if "ResNet101" == frontend and not os.path.isfile("models/res_net/resnet_v2_101.ckpt"):
            download_checkpoints("ResNet101")
        if "ResNet152" == frontend and not os.path.isfile("models/res_net/resnet_v2_152.ckpt"):
            download_checkpoints("ResNet152")
        if "ResNet200" == frontend and not os.path.isfile("models/res_net/resnet_v2_200.ckpt"):
            download_checkpoints("ResNet200")
        if "MobileNetV2" == frontend and not os.path.isfile(
                "models/mobilenet_v2/mobilenet_v2.ckpt.data-00000-of-00001"):
            download_checkpoints("MobileNetV2")
        if "InceptionV4" == frontend and not os.path.isfile("models/inceptionnet/inception_v4.ckpt"):
            download_checkpoints("InceptionV4")

    def add_image_normalise_layer(self, image):
        with tf.variable_scope("normalize"):
            mean_pixel_val = tf.constant(self.mean_pixel_val, dtype=tf.float32, name='mean_pixel_val')
            std_pixel_val = tf.constant(self.std_pixel_val, dtype=tf.float32,  name='mean_pixel_val')
            image -= mean_pixel_val
            image /= std_pixel_val
            return image

    @staticmethod
    def add_image_resize_layer(input_img, resize_func=tf.image.resize_bilinear, factor=32.0, scope='resize'):
        with tf.variable_scope(scope):
            input_height = tf.cast(tf.shape(input_img)[1], tf.float32)
            input_width = tf.cast(tf.shape(input_img)[2], tf.float32)
            processed_height = tf.ceil(tf.ceil(tf.divide(input_height, tf.constant(factor))) * tf.constant(factor))
            processed_width = tf.ceil(tf.ceil(tf.divide(input_width, tf.constant(factor))) * tf.constant(factor))
            processed_height = tf.cast(processed_height, tf.int32)
            processed_width = tf.cast(processed_width, tf.int32)
            return resize_func(input_img, (processed_height, processed_width))

    def build_model(self, crop_width=128, crop_height=128, ):
        if self.gpu:
            device = "/device:GPU:0"
        else:
            device = "/cpu:0"

        with tf.device(device):
            # Get the selected model.
            # Some of them require pre-trained ResNet
            # print("Preparing the model ...")

            if self.model_name not in SUPPORTED_MODELS:
                raise ValueError("The model you selected is not supported. "
                                 "The following models are currently supported: {0}".format(SUPPORTED_MODELS))
            self.handle_frontend(self.frontend)
            if self.apply_resize:
                net_input = self.add_image_resize_layer(self.input_img)
            else:
                net_input = self.input_img

            net_input = self.add_image_normalise_layer(net_input)
            self.train_flag = tf.Variable(False, trainable=False, name='train_mode')

            init_fn = None
            stable_param = []
            # if self.model_name == "FC-DenseNet56" or self.model_name == "FC-DenseNet67" or \
            #         self.model_name == "FC-DenseNet103":
            if self.model_name.find("FC-DenseNet") == 0:
                # Paper: https://arxiv.org/pdf/1611.09326
                network = build_fc_densenet(net_input, preset_model=self.model_name, num_classes=self.num_classes)

            elif self.model_name == "UNet":
                # Paper: https://arxiv.org/pdf/1505.04597
                network, init_fn, stable_param = build_unet(net_input, self.train_flag)

            elif self.model_name == "SmallUNet":
                # Paper: https://arxiv.org/pdf/1505.04597
                network, init_fn, stable_param, self.probe = build_small_unet(net_input, self.train_flag)

            elif self.model_name == "FRRN-A" or self.model_name == "FRRN-B":
                # Paper: https://arxiv.org/pdf/1611.08323.pdf
                network = build_frrn(net_input, self.train_flag, preset_model=self.model_name,
                                     num_classes=self.num_classes)

            elif self.model_name == "Encoder-Decoder" or self.model_name == "Encoder-Decoder-Skip":
                # Paper: https://arxiv.org/pdf/1511.00561
                network = build_encoder_decoder(net_input, preset_model=self.model_name, num_classes=self.num_classes)

            elif self.model_name == "MobileUNet" or self.model_name == "MobileUNet-Skip":
                # Paper (U-Net): https://arxiv.org/pdf/1505.04597
                # Paper (MobileNet): https: // arxiv.org / pdf / 1801.04381
                network = build_mobile_unet(net_input, preset_model=self.model_name, num_classes=self.num_classes)

            elif self.model_name == "AdapNet":
                # Paper: http://ais.informatik.uni-freiburg.de/publications/papers/valada17icra.pdf
                network = build_adaptnet(net_input, num_classes=self.num_classes)

            elif self.model_name == "RefineNet":
                # RefineNet requires pre-trained ResNet weights
                # Paper: https://arxiv.org/pdf/1611.06612
                network, init_fn, stable_param = build_refinenet(net_input, preset_model=self.model_name,
                                                                 frontend=self.frontend, num_classes=self.num_classes,
                                                                 is_training=self.train_flag,
                                                                 load_pre_train=self.load_pre_train)

            elif self.model_name == "PSPNet":
                # Paper: https://arxiv.org/pdf/1612.01105
                # Image size is required for PSPNet
                # PSPNet requires pre-trained ResNet weights
                network, init_fn, stable_param = build_pspnet(net_input, label_size=[crop_height, crop_width],
                                                              preset_model=self.model_name, frontend=self.frontend,
                                                              num_classes=self.num_classes, is_training=self.train_flag,
                                                              load_pre_train=self.load_pre_train)

            elif self.model_name == "GCN":  # Global Convolution Network
                # GCN requires pre-trained ResNet weights
                # Paper: https://arxiv.org/pdf/1703.02719
                network, init_fn, stable_param = build_gcn(net_input, preset_model=self.model_name,
                                                           frontend=self.frontend, num_classes=self.num_classes,
                                                           is_training=self.train_flag,
                                                           load_pre_train=self.load_pre_train)

            elif self.model_name == "DeepLabV3":
                # Paper: https://arxiv.org/pdf/1706.05587.pdf
                # DeepLabV requires pre-trained ResNet weights
                network, init_fn, stable_param = build_deeplabv3(net_input, preset_model=self.model_name,
                                                                 frontend=self.frontend, num_classes=self.num_classes,
                                                                 is_training=self.train_flag,
                                                                 load_pre_train=self.load_pre_train)

            elif self.model_name == "DeepLabV3_plus":
                # Paper: https://arxiv.org/pdf/1802.02611.pdf
                # DeepLabV3+ requires pre-trained ResNet weights
                # TODO: Potentially needs larger crop height and width
                network, init_fn, stable_param = build_deeplabv3_plus(net_input, preset_model=self.model_name,
                                                                      frontend=self.frontend, num_classes=self.num_classes,
                                                                      is_training=self.train_flag,
                                                                      load_pre_train=self.load_pre_train)

            elif self.model_name == "DenseASPP":
                # Paper: http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.pdf
                # DenseASPP requires pre-trained ResNet weights
                network, init_fn, stable_param = build_dense_aspp(net_input, preset_model=self.model_name,
                                                                  frontend=self.frontend, num_classes=self.num_classes,
                                                                  is_training=self.train_flag,
                                                                  load_pre_train=self.load_pre_train)

            elif self.model_name == "DDSC":
                # TODO: Fix ResNeXt error with input image size
                # Paper: http://openaccess.thecvf.com/content_cvpr_2018/papers/Bilinski_Dense_Decoder_Shortcut_CVPR_2018_paper.pdf
                # DDSC requires pre-trained ResNet weights
                # Takes time because of the loading of the ResNeXt block
                network, init_fn, stable_param = build_ddsc(net_input, preset_model=self.model_name,
                                                            frontend=self.frontend,  num_classes=self.num_classes,
                                                            is_training=self.train_flag,
                                                            load_pre_train=self.load_pre_train)

            elif self.model_name == "BiSeNet":
                # Paper: https://arxiv.org/pdf/1808.00897.pdf
                # BiSeNet requires pre-trained ResNet weights
                network, init_fn, stable_param = build_bisenet(net_input, preset_model=self.model_name,
                                                               frontend=self.frontend, num_classes=self.num_classes,
                                                               is_training=self.train_flag,
                                                               load_pre_train=self.load_pre_train)

            elif self.model_name == "custom":
                network = build_custom(net_input, self.num_classes)

            else:
                raise ValueError("Error: the model %d is not available."
                                 " Try checking which models are available using the command python main.py --help")
            if self.apply_resize:
                network = tf.image.resize_bilinear(network, (tf.shape(self.input_img)[1], tf.shape(self.input_img)[2]))

            self.load_pre_trained_model_func = init_fn
            self.stable_trainable_tensors = stable_param

            self.layers['output_logit'] = network
            with tf.variable_scope('Softmax'):
                self.layers['soft'] = tf.nn.softmax(network, axis=3, name='soft')
            with tf.variable_scope('Arg_min'):
                self.layers['predicted_label'] = tf.argmin(network, axis=3, name='predicted_label')
            self.output_node_names.append(self.layers['soft'].op.name)

            self.model_graph = tf.get_default_graph()
            self.write_graph_for_tensorboard()

            self.add_loss()
            return network, init_fn

    def update_graph_with_bnorm(self, item_list):
        input_graph_def = tf.get_default_graph().as_graph_def()
        # input_graph_def = Network.fix_bnorm_bugs(input_graph_def)
        tf.reset_default_graph()
        id = ''
        tf.import_graph_def(input_graph_def, name=id)
        update_graph = tf.get_default_graph()
        up_l = []
        for item in item_list:
            item = update_graph.get_tensor_by_name(id + min(len(id), 1) * '/' + item.name)
            up_l.append(item)
        return up_l

    def add_loss(self):
        with tf.variable_scope("loss"):
            entropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.target_img,
                                                                      logits=self.layers['output_logit'])
            if self.weight is not None:
                weighted_entropy_loss = tf.multiply(entropy_loss, tf.stop_gradient(self.weight))
            else:
                weighted_entropy_loss = entropy_loss
            mean_entropy_loss = tf.reduce_mean(weighted_entropy_loss)
            self.layers['entropy_loss'] = entropy_loss
            self.layers['weighted_entropy_loss'] = weighted_entropy_loss
            self.losses['loss'] = mean_entropy_loss

    def add_optimizer(self, opts, _type="Adam"):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            if _type == "Adam":
                optimizer = tf.train.AdamOptimizer(learning_rate=opts['learningRate'])
            elif _type == "RMS":
                optimizer = tf.train.RMSPropOptimizer(learning_rate=opts['learningRate'], decay=opts['weightDecay'])
            elif _type == "Momentum":
                optimizer = tf.train.MomentumOptimizer(learning_rate=opts['learningRate'], momentum=opts['momentum'])
            else:
                raise ValueError("Unknown optimizer")
            gvs = optimizer.compute_gradients(self.losses['loss'])
            if self.load_pre_train:
                final_gvs = []
                stable_trainable_tensors_name = [param.name for param in self.stable_trainable_tensors]
                with tf.variable_scope('Gradient_Mult'):
                    for grad, var in gvs:
                        if var.name in stable_trainable_tensors_name and grad is not None:
                            scale = 0.001
                            grad = tf.multiply(grad, scale)
                        final_gvs.append((grad, var))
            else:
                final_gvs = gvs
            self.train_op = optimizer.apply_gradients(final_gvs)

    def write_graph_for_tensorboard(self):
        tf.summary.FileWriter(self.graph_path, self.model_graph)

    def update_stats(self, q_names, out_measures, stats):

        for idx, name in enumerate(q_names):
            loss_val = np.mean(out_measures[idx])
            cur_stats = stats[name]
            tot_loss = cur_stats['average'] * cur_stats['count'] + loss_val
            cur_stats['count'] = cur_stats['count'] + 1
            cur_stats['average'] = tot_loss / cur_stats['count']
            stats[name] = cur_stats
        return stats

    def extract_loss_from_layer(self, stats):
        q_set = []
        q_names = []
        for name in stats.keys():
            q_set.append(self.losses[name])
            q_names.append(name)
        return q_set, q_names

    def train_step(self, sess, blobs, stats, mode="train"):

        feed_dict = {self.input_img: blobs["input_img"], self.target_img: blobs["output_img"],
                     self.train_flag: False}
        if self.weight is not None:
            feed_dict[self.weight] = blobs["pos_weight"]

        if mode == 'train':
            # Run optimization op (back propagation) only in the training phase
            feed_dict[self.train_flag] = True
            sess.run([self.train_op], feed_dict=feed_dict)

        q_set, q_names = self.extract_loss_from_layer(stats)
        feed_dict[self.train_flag] = False

        # q_set.append(self.layers['soft'])
        out_measures = sess.run(q_set, feed_dict=feed_dict)
        # thresh_cost = self.test_threshold(out_measures.pop(), blobs["output_img"])
        stats = self.update_stats(q_names, out_measures, stats)

        return stats

    @staticmethod
    def test_threhold(pred_out, gt_out):
        thresh_range = np.arange(10)/10
        pred_out = pred_out[:, :, :, 0]
        gt_out = gt_out[:, :, :, 0].astype(np.bool)
        gt_out = np.expand_dims(gt_out, axis=-1)
        bin_pred_out = np.expand_dims(pred_out, axis=-1) - np.expand_dims(thresh_range, axis=0)
        bin_pred_out = bin_pred_out > 0
        tp = (gt_out & bin_pred_out).astype(np.int64)
        fp = (~gt_out & bin_pred_out).astype(np.int64)
        fn = (gt_out & ~bin_pred_out).astype(np.int64)
        tp = np.sum(tp, axis=(0, 1, 2))  # True Positive
        fp = np.sum(fp, axis=(0, 1, 2))  # False Positive
        fn = np.sum(fn, axis=(0, 1, 2))  # False Negative
        dice_cost = 2 * tp / (2 * tp + fp + fn)
        dice_cost_per_thresh = dict(zip(thresh_range, dice_cost))
        return dice_cost_per_thresh

    def norm_img(self, img_set, orig_im):
        conct_im = None
        for idx, im in enumerate(img_set):
            im = im.astype(np.float32)
            im = (im - im.min()) / (im.max() - im.min())
            if idx == 0:
                conct_im = im
            else:
                conct_im = np.concatenate((conct_im, im), axis=1)
        tmp_im = np.split(orig_im, 3, axis=2)
        im_rgb = np.squeeze(np.stack((tmp_im[2], tmp_im[1], tmp_im[0]), axis=2))

        return conct_im, im_rgb.astype(np.uint8)

    def test_and_display_step(self, sess, blobs, stats):
        feed_dict = {self.input_img: blobs["input_img"], self.target_img: blobs["output_img"], self.train_flag: False}

        # q_set, q_names = self.extract_loss_from_layer(stats)

        # g = sess.graph.get_tensor_by_name("logits/BiasAdd" + ':0')
        # feed_dict[self.train_flag] = False
        [softmax_out, pred_lab] = sess.run([self.layers['soft'],
                                                   self.layers['predicted_label']], feed_dict=feed_dict)

        softmax_out = softmax_out[0, :, :, 0]
        pred_lab = pred_lab[0, :, :]
        gt_lab = blobs["output_img"][0, :, :, 0]
        composite_img, orig_im = self.norm_img([softmax_out, pred_lab, gt_lab], blobs["input_img"][0])

        def gen_pretty(im):
            orig_img_cpy = im.copy()
            result = []
            for x in np.split(orig_img_cpy, 3, axis=-1):
                x[np.where(softmax_out > 0.4)] *= 0
                result.append(x)
            result = np.concatenate(result, axis=-1)
            return result.astype(np.uint8)

        r = gen_pretty(orig_im)
        cv2.imshow("cv2Im scaled", r)
        cv2.imshow("orig im", orig_im)
        cv2.waitKey(0)

        # id_list, node_list = self.prepare_probe()
        # node_state = sess.run(node_list, feed_dict=feed_dict)
        # result = dict(zip(id_list, node_state))
        # import scipy.io as sio
        # sio.savemat('matlab.mat', result)
        return stats

    def prepare_probe(self):
        id_list = []
        node_list = []
        for node_name in self.probe.keys():
            id_list.append(node_name)
            node_list.append(self.probe[node_name])
        return id_list, node_list

    @staticmethod
    def fix_bnorm_bugs(input_graph_def):

        # for fixing the bug of batch norm
        for node in input_graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr:
                    del node.attr['use_locking']
            elif node.op == 'AssignAdd':
                node.op = 'Add'
                if 'use_locking' in node.attr:
                    del node.attr['use_locking']
        return input_graph_def

    def freeze_net(self, sess, device_agnostic=True):
        # Specify the real node name
        graph = tf.get_default_graph()

        sess.run(tf.assign(self.train_flag, False))
        input_graph_def = graph.as_graph_def()

        # input_graph_def = Network.fix_bnorm_bugs(input_graph_def)
        if device_agnostic:
            for node in input_graph_def.node:
                node.device = ""
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, input_graph_def, self.output_node_names)

        tf.summary.FileWriter(self.graph_pred_path, output_graph_def)

        # Finally we serialize and dump the output graph to the filesystem
        # with tf.gfile.GFile(os.path.join(self.output_net_path, 'net.pb'), "wb") as f:
        #     f.write(output_graph_def.SerializeToString())
        # print("%d ops in the final graph." % len(output_graph_def.node))

        return output_graph_def
