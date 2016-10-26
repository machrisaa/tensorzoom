import tensorflow as tf

import numpy as np

PRINT_LAYER = False


class TensorZoomNet:
    """
    The generative network of the TensorZoom.
    This is a implementation of Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    https://arxiv.org/abs/1609.04802

    Due to limitation of the conv2d_transpose that require a output size.
    Unlike many purely convolution network, input_shape is need to specified in order to support dynamic size input.
    This is particularly useful for mobile usage.

    Initialize the network with a pre-trained npy data
    Call build to create the layers
    Call save_npy to save the variables into a npy file
    Call save_graph to export the network as a standard Tensorflow GraphDef for mobile/native usage

    Access the layer by the fields. e.g. output is the final result
    """

    def __init__(self, npy_path=None, trainable=True):
        if npy_path is not None:
            self.data_dict = np.load(npy_path).item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable

        self.conv_trainable = trainable
        self.res_trainable = trainable
        self.deconv_trainable = trainable

    # noinspection PyAttributeOutsideInit
    def build(self, batch_img, train_mode=None, clear_memory=True, input_shape=None):
        assert isinstance(batch_img, tf.Tensor)

        self.train_mode = train_mode
        if input_shape is not None:
            split = tf.split(0, 3, input_shape)
            h = split[0]
            w = split[1]
            d1_out_shape = tf.concat(0, [[1], h * 2, w * 2, [32]])
            d2_out_shape = tf.concat(0, [[1], h * 4, w * 4, [16]])
            d3_out_shape = tf.concat(0, [[1], h * 4, w * 4, [3]])
        else:
            d1_out_shape = None
            d2_out_shape = None
            d3_out_shape = None

        self.conv1 = self.conv_block(batch_img, 9, 3, 64, 1, 'conv1', trainable=self.conv_trainable)

        self.res1 = self.res_block(self.conv1, 64, 'res1', trainable=self.res_trainable)
        self.res2 = self.res_block(self.res1, 64, 'res2', trainable=self.res_trainable)
        self.res3 = self.res_block(self.res2, 64, 'res3', trainable=self.res_trainable)
        self.res4 = self.res_block(self.res3, 64, 'res4', trainable=self.res_trainable)
        self.res5 = self.res_block(self.res4, 64, 'res5', trainable=self.res_trainable)
        self.res6 = self.res_block(self.res5, 64, 'res6', trainable=self.res_trainable)

        self.deconv1 = self.deconv_block(self.res6, 3, 64, 32, 2, 'deconv1',
                                         trainable=self.deconv_trainable, output_shape=d1_out_shape)
        self.deconv2 = self.deconv_block(self.deconv1, 3, 32, 16, 2, 'deconv2',
                                         trainable=self.deconv_trainable, output_shape=d2_out_shape)
        self.deconv3 = self.deconv_block(self.deconv2, 9, 16, 3, 1, 'deconv3',
                                         trainable=self.deconv_trainable, output_shape=d3_out_shape)

        self.output = (tf.tanh(self.deconv3) + 1) / 2

        if PRINT_LAYER: print self.get_var_count()

        if clear_memory:
            self.data_dict = None

        return self.output

    def conv_block(self, input, filter_size, in_channels, out_channels, strides, name, use_elu=True,
                   trainable=None):
        with tf.variable_scope(name):
            filt = self.get_conv_filter_var(filter_size, in_channels, out_channels, name, trainable=trainable)
            conv = tf.nn.conv2d(input, filt, [1, strides, strides, 1], 'SAME')

            bias = self.get_conv_bias_var(out_channels, name, trainable=trainable)
            conv = tf.nn.bias_add(conv, bias)

            if use_elu:
                conv = tf.nn.relu(conv)

            bn = self.bn_layer(conv, out_channels, name, trainable=trainable)

            return bn

    def res_block(self, input, size, name, trainable=None):
        with tf.variable_scope(name):
            filt = self.get_conv_filter_var(3, size, size, name + "_conv1", trainable=trainable)
            conv = tf.nn.conv2d(input, filt, [1, 1, 1, 1], 'SAME')

            bias = self.get_conv_bias_var(size, name + "_conv1", trainable=trainable)
            conv = tf.nn.bias_add(conv, bias)

            bn = self.bn_layer(conv, size, name + "_bn1", trainable=trainable)

            elu = tf.nn.relu(bn)

            filt = self.get_conv_filter_var(3, size, size, name + "_conv2", trainable=trainable)
            conv = tf.nn.conv2d(elu, filt, [1, 1, 1, 1], 'SAME')

            bias = self.get_conv_bias_var(size, name + "_conv2", trainable=trainable)
            conv = tf.nn.bias_add(conv, bias)

            bn = self.bn_layer(conv, size, name + "_bn2", trainable=trainable)

            output = bn + input
            return output

    def deconv_block(self, input, filter_size, in_channels, out_channels, strides, name, pure_conv=False,
                     trainable=None, output_shape=None):
        with tf.variable_scope(name):
            if input.get_shape().is_fully_defined():
                dims = input.get_shape().as_list()
                batchs = dims[0]
                out_h = dims[1] * strides
                out_w = dims[2] * strides
                output_shape = [batchs, out_h, out_w, out_channels]

            filt = self.get_conv_filter_var(filter_size, out_channels, in_channels, name,
                                            trainable=trainable)  # reversed in/out
            conv = tf.nn.conv2d_transpose(input, filt, output_shape, [1, strides, strides, 1], 'SAME')

            bias = self.get_conv_bias_var(out_channels, name, trainable=trainable)
            conv = tf.nn.bias_add(conv, bias)

            if pure_conv:
                return conv

            conv = tf.nn.relu(conv)

            bn = self.bn_layer(conv, out_channels, name, trainable=trainable)

            return bn

    def bn_layer(self, x, size, name, trainable=None, decay=0.999):
        offset = self.get_var(tf.truncated_normal([size], 0.0, 0.01), name + '_offset', trainable=trainable)
        scale = self.get_var(tf.truncated_normal([size], 1.0, 0.01), name + '_scale', trainable=trainable)

        ema_mean = self.get_var(tf.constant(0, tf.float32, [size]), name + '_ema_mean', trainable=trainable)
        ema_var = self.get_var(tf.constant(0, tf.float32, [size]), name + '_ema_var', trainable=trainable)

        def train_bn():
            current_mean, current_variance = tf.nn.moments(x, [0, 1, 2])

            mean_op = ema_mean.assign_sub((ema_mean - current_mean) * (1 - decay))
            var_op = ema_var.assign_sub((ema_var - current_variance) * (1 - decay))

            with tf.control_dependencies([mean_op, var_op]):
                # use ema value even for training stage in order to support adversarial training
                # return tf.nn.batch_normalization(x, current_mean, current_variance, offset, scale, 1e-8)
                return tf.nn.batch_normalization(x, ema_mean, ema_var, offset, scale, 0.01)

        def non_train_bn():
            return tf.nn.batch_normalization(x, ema_mean, ema_var, offset, scale, 0.01)

        if trainable is False:
            bn = non_train_bn()
        elif self.train_mode is None:
            if self.trainable:
                bn = train_bn()
            else:
                bn = non_train_bn()
        else:
            bn = tf.cond(self.train_mode, train_bn, non_train_bn)

        return bn

    def get_conv_filter_var(self, filter_size, in_channels, out_channels, name, trainable=None):
        filt = self.get_var(
            tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.01),
            name + '_filter', trainable=trainable)
        return filt

    def get_conv_bias_var(self, channels, name, trainable=None):
        bias = self.get_var(tf.truncated_normal([channels], .0, .01), name + '_bias', trainable=trainable)
        return bias

    def get_var(self, initial_value, name, trainable=None):
        if PRINT_LAYER:
            print name

        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name]
        else:
            value = initial_value

        if trainable is None:
            trainable = self.trainable

        if trainable:
            var = tf.Variable(value, name=name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=name)

        self.var_dict[name] = var
        return var

    def save_npy(self, sess, npy_path="./save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for name, var in self.var_dict.items():
            var_out = sess.run(var)
            data_dict[name] = var_out

        np.save(npy_path, data_dict)
        print("file saved", npy_path)
        return npy_path

    def get_var_count(self):
        count = 0
        for v in self.var_dict.values():
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count

    def var_list(self):
        return self.var_dict.values()

    def save_graph(self, logdir, name, input_name='input', output_name='output'):
        g = tf.Graph()
        with g.as_default():
            shape = tf.placeholder(tf.int32, shape=[3], name='input_shape')
            input = tf.placeholder(tf.float32, name=input_name)
            self.build(tf.expand_dims(input, 0), input_shape=shape, train_mode=False, clear_memory=False)
            output = tf.squeeze(self.output, [0], name=output_name)

        tf.train.write_graph(g.as_graph_def(), logdir=logdir, name=name, as_text=False)


# old test:
# def _test_size():
#     sess = tf.Session()
#     # with tf.device("/gpu:0"):
#     with tf.device("/cpu:0"):
#         net = TensorZoomNet()
#         shape = (4, 256, 256, 3)  # target
#         # shape = (2, 1024,1024, 3)
#         # shape = (8, 256, 256, 3)
#         # shape = (1, 256, 256, 3)
#         # shape = (1, 128, 256, 3)
#         # shape = (1, 128, 230, 3) , can run but result will not the same dim
#         input = tf.constant(1, tf.float32, shape)
#         output = net.build(input)
#
#         sess.run(tf.initialize_all_variables())
#         result = sess.run(output)
#
#         print np.shape(result)
#         # assert np.shape(result) == shape , may not be the same for w or h not multiple of 4
#
#         # net valid for all size:
#         for w in range(16, 40):
#             for h in range(w, 40):
#                 net = TensorZoomNet()
#                 shape = (1, w, h, 3)
#                 input = tf.constant(1, tf.float32, shape)
#                 output = net.build(input)
#
#                 sess.run(tf.initialize_all_variables())
#                 result = sess.run(output)
#
#                 print shape, np.shape(result)
#
#
# def _test_save():
#     # this test if save load produce equal result
#     sess = tf.Session()
#     with tf.device("/cpu:0"):
#         net = TensorZoomNet()
#         shape = (2, 256, 256, 3)  # target
#         input = tf.constant(1, tf.float32, shape)
#         output = net.build(input)
#         sess.run(tf.initialize_all_variables())
#         result = sess.run(output)
#
#         print np.shape(result)
#         net.save_npy(sess, './test.npy')
#
#         net2 = TensorZoomNet('./test.npy')
#         output = net2.build(input)
#         sess.run(tf.initialize_all_variables())
#         result2 = sess.run(output)
#
#         assert np.array_equal(result, result2)


def _test_export_graph_def():
    """
    Sample usage of how to export the network to GraphDef
    """
    with tf.device('/cpu:0'):
        net = TensorZoomNet(npy_path='./results/tz6-s-stitch-sblur-lowtv/tz6-s-stitch-sblur-lowtv-gen.npy'
                            , trainable=False)
        net.save_graph(logdir='./', name='export.pb')


# test
if __name__ == '__main__':
    PRINT_LAYER = True
    # _size_test()
    # _test_save()

    _test_export_graph_def()
