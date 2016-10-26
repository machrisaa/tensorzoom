import tensorflow as tf

import numpy as np

VGG_MEAN = [103.939, 116.779, 123.68]


class Discriminator:
    """
    The discriminative network of the TensorZoom.
    This is a implementation of Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    https://arxiv.org/abs/1609.04802

    This implementation used initial pre-trained data from VGG19. So there are some VGG-related calculations, like VGG_MEAN

    Initialize the network with a pre-trained npy data
    Call build to create the layers

    Access the layer by the fields. e.g. prob is the final result
    """

    def __init__(self, npy_path=None, trainable=True, input_size=224):
        if npy_path is not None:
            self.data_dict = np.load(npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.var_dict_name = {}
        self.trainable = trainable
        self.input_size = input_size

    # noinspection PyAttributeOutsideInit
    def build(self, rgb, train_mode=None, parent=None):
        if parent is not None:
            self.var_dict = parent.var_dict
            self.var_dict_name = parent.var_dict_name

        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(3, 3, rgb_scaled)
        assert red.get_shape().as_list()[1:] == [self.input_size, self.input_size, 1]
        assert green.get_shape().as_list()[1:] == [self.input_size, self.input_size, 1]
        assert blue.get_shape().as_list()[1:] == [self.input_size, self.input_size, 1]
        bgr = tf.concat(3, [
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [self.input_size, self.input_size, 3]

        self.train_mode = train_mode

        self.conv1_1 = self.conv_layer(bgr, 3, 64, 1, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, 2, "conv1_2")
        self.bn1 = self.bn_layer(self.conv1_2, 64, "bn1")

        self.conv2_1 = self.conv_layer(self.bn1, 64, 128, 1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, 2, "conv2_2")
        self.bn2 = self.bn_layer(self.conv2_2, 128, "bn2")

        self.conv3_1 = self.conv_layer(self.bn2, 128, 256, 1, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, 2, "conv3_2")
        self.bn3 = self.bn_layer(self.conv3_2, 256, "bn3")

        self.conv4_1 = self.conv_layer(self.bn3, 256, 512, 1, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, 2, "conv4_2")
        self.bn4 = self.bn_layer(self.conv4_2, 512, "bn4")

        self.conv5_1 = self.conv_layer(self.bn4, 512, 512, 1, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, 2, "conv5_2")
        self.bn5 = self.bn_layer(self.conv5_2, 512, "bn5")

        self.desen1 = self.fc_layer(self.bn5, ((self.input_size / (2 ** 5)) ** 2) * 512, 1024, "desen1")
        self.relu6 = tf.nn.relu(self.desen1)

        self.desen2 = self.fc_layer(self.relu6, 1024, 1, "desen2")
        self.prob = tf.sigmoid(self.desen2, name="prob")

        self.data_dict = None

    def conv_layer(self, bottom, in_channels, out_channels, stride, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, stride, stride, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)

            # relu = tf.nn.relu(bias)
            # use leaky relu of 0.2 instead:
            relu = tf.maximum(0.2 * bias, bias)

            return relu

    def bn_layer(self, x, size, name, declay=0.99):
        offset = self.get_var(tf.constant(0.0, tf.float32, [size]), name + '_offset', 0, name + '_offset')
        scale = self.get_var(tf.constant(1.0, tf.float32, [size]), name + '_scale', 0, name + '_scale')

        ema_mean = self.get_var(tf.constant(0, tf.float32, [size]), name + '_ema_mean', 0, name + '_ema_mean')
        ema_var = self.get_var(tf.constant(0, tf.float32, [size]), name + '_ema_var', 0, name + '_ema_var')

        def train_bn():
            current_mean, current_variance = tf.nn.moments(x, [0, 1, 2])

            mean_op = ema_mean.assign_sub((ema_mean - current_mean) * (1 - declay))
            var_op = ema_var.assign_sub((ema_var - current_variance) * (1 - declay))

            with tf.control_dependencies([mean_op, var_op]):
                # use ema value even for training stage in order to support adversarial training
                # return tf.nn.batch_normalization(x, current_mean, current_variance, offset, scale, 1e-8)
                return tf.nn.batch_normalization(x, ema_mean, ema_var, offset, scale, 0.01)

        def non_train_bn():
            return tf.nn.batch_normalization(x, ema_mean, ema_var, offset, scale, 0.01)

        if self.trainable is False:
            bn = non_train_bn()
        elif self.train_mode is None:
            if self.trainable:
                bn = train_bn()
            else:
                bn = non_train_bn()
        else:
            bn = tf.cond(self.train_mode, train_bn, non_train_bn)

        return bn

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if (name, idx) in self.var_dict:
            var = self.var_dict[(name, idx)]
            assert var.get_shape() == initial_value.get_shape()
            return var
        else:
            if self.data_dict is not None and name in self.data_dict:
                value = self.data_dict[name][idx]
            else:
                value = initial_value

            if self.trainable:
                var = tf.Variable(value, name=var_name)
            else:
                var = tf.constant(value, dtype=tf.float32, name=var_name)

            self.var_dict[(name, idx)] = var
            self.var_dict_name[var_name] = var

            # print var_name, var.get_shape().as_list()
            assert var.get_shape() == initial_value.get_shape()

            return var

    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in self.var_dict.items():
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print("file saved", npy_path)
        return npy_path

    def get_var_count(self):
        count = 0
        for v in self.var_dict.values():
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count

    def get_all_var(self):
        return self.var_dict.values()
