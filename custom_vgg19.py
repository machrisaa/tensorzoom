import tensorflow as tf

import time
from tensoflow_vgg import vgg19

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg19(vgg19.Vgg19):
    """
    A simplified VGG19 network removed all the fully connected layers.
    This class need to reference to tensorflow_vgg project:
    https://github.com/machrisaa/tensorflow-vgg
    Check the instruction to get the pre-trained npy data for this class.
    """

    def __init__(self, vgg19_npy_path=None, var_map=None):
        vgg19.Vgg19.__init__(self, vgg19_npy_path)
        self.var_map = var_map

    # Input should be an rgb image [batch, height, width, 3]
    # values scaled [0, 1]
    def build(self, rgb, train=False):
        start_time = time.time()
        print "build model started"
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(3, 3, rgb_scaled)
        bgr = tf.concat(3, [
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')
        # self.conv1_2 = self.conv_layer2(self.conv1_1, "conv1_2")
        # self.pool1 = self.conv1_2

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')
        # self.conv2_2 = self.conv_layer2(self.conv2_1, "conv2_2")
        # self.pool2 = self.conv2_2

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, "conv3_4")
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')
        # self.conv3_4 = self.conv_layer2(self.conv3_3, "conv3_4")
        # self.pool3 = self.conv3_4

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, "conv4_4")
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')
        # self.conv4_4 = self.conv_layer2(self.conv4_3, "conv4_4")
        # self.pool4 = self.conv4_4

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, "conv5_4")
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')
        # self.conv5_4 = self.conv_layer2(self.conv5_3, "conv5_4")
        # self.pool5 = self.conv5_4

        self.data_dict = None
        print "build model finished: %ds" % (time.time() - start_time)

    def get_conv_filter(self, name):
        var = None
        if self.var_map is not None:
            var = self.var_map[('filter', name)]
        if var is None:
            var = vgg19.Vgg19.get_conv_filter(self, name)
        return var

    def get_fc_weight(self, name):
        var = None
        if self.var_map is not None:
            var = self.var_map[('weight', name)]
        if var is None:
            var = vgg19.Vgg19.get_fc_weight(self, name)
        return var

    def get_bias(self, name):
        var = None
        if self.var_map is not None:
            var = self.var_map[('bias', name)]
        if var is None:
            var = vgg19.Vgg19.get_bias(self, name)
        return var
