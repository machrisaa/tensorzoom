"""
The core method define the training the TensorZoom network.
This implementation is based on the paper
Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
https://arxiv.org/abs/1609.04802

In order to run the training, 2 items are required:
1.  The custom_vgg19 network is needed which is referening to the tensorflow_vgg project:
    https://github.com/machrisaa/tensorflow-vgg
    Check the instruction to get the pre-trained npy data for this class.
    After download, replace the path for the constant: VGG_NPY_PATH
2.  The Microsoft Coco2014 data set: http://mscoco.org/dataset/#download
    We are using 2014 Training images [80K/13GB] data set.
    After download, replace the path for the constant: COCO2014_PATH

Here is the basic steps about the training:
1. blur the input image (in order to increase the difficulty
2. slice the blured image into 16 images
3. concat the 16 images into a batch and resize the image by 0.25 times
4. pass the image to the generative network (our TensorZoomNet) and get gen-result
5. split the batch in the gen-result and stitch the results into a large gen-image
6. train discriminator and generator alternatively:
    - for the dis-training phase,
        pass the gen-image and the input image to the discriminator to train it
    - for the gen-training phase,
        pass the gen-image and large image to custom_vgg19 to get content-cost
        pass the gen-image to discriminator to get dis-cost
        pass the gen-image to get_invariant_cost2 to get tv-cost
        weighted sum these 3 costs to train the generative network

"""

import tensorflow as tf

import time
import math
import skimage
import skimage.io
import skimage.transform

from dataloader_coco2014 import DataSet

from tensorzoom_net import TensorZoomNet
from discriminator_net import Discriminator
import custom_vgg19

# see: https://github.com/machrisaa/tensorflow-vgg
VGG_NPY_PATH = '../tensoflow_vgg/vgg19.npy'

# get it from: http://msvocds.blob.core.windows.net/coco2014/train2014.zip
COCO2014_PATH = '../../datasets/coco2014/train2014'

# set them to None to start the training from scratch
DIS_NPY = './results/tz6-s-stitch-sblur-lowtv/tz6-s-stitch-sblur-lowtv-dis.npy'
GEN_NPY = './results/tz6-s-stitch-sblur-lowtv/tz6-s-stitch-sblur-lowtv-gen.npy'

SIZE = 256  # the size of input which will be split into 16 smaller images
TRAIN_DIR = "./train"
SUMMARY_FOLDER = TRAIN_DIR + "/summary"


def train(ds, dis_learning_rate, gen_learning_rate):
    with tf.Session() as sess:
        start_time = time.time()

        in_train_gen = tf.placeholder(tf.bool)
        in_train_dis = tf.placeholder(tf.bool)
        in_large = tf.placeholder(tf.float32, [1, SIZE, SIZE, 3])

        # extra difficulty: blur the large image:
        blur_filter = tf.constant(1, shape=[5, 5, 1, 1], dtype=tf.float32) / 25
        blur_filter = tf.tile(blur_filter, [1, 1, 3, 1])
        in_large_blur = tf.nn.depthwise_conv2d(in_large, blur_filter, strides=[1, 1, 1, 1], padding='SAME')

        # reduce the size to smaller
        in_small = tf.nn.avg_pool(in_large_blur, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

        # use stitch training method, slice the image into tiles and concat as batches
        t = create_tiles(in_small, SIZE / 4, SIZE / 4, 4)
        in_stitch = tf.concat(0, [tf.concat(0, t[y]) for y in xrange(4)])  # row1, row2, ...

        generator = TensorZoomNet(trainable=True, npy_path=GEN_NPY)
        with tf.name_scope("generator"):
            generator.build(in_stitch, train_mode=in_train_gen)

        # stitch the tiles back together after split the batches
        gen_split = tf.split(0, 4 * 4, generator.output)
        gen_result = tf.concat(1, [tf.concat(2, [gen_split[x] for x in xrange(4 * y, 4 * y + 4)])
                                   for y in xrange(4)])

        discriminator_truth = Discriminator(trainable=True, input_size=SIZE,
                                            npy_path=DIS_NPY)
        with tf.name_scope('dis_truth'):
            discriminator_truth.build(in_large, train_mode=in_train_dis)

        discriminator_gen = Discriminator(trainable=True, input_size=SIZE)
        with tf.name_scope('dis_gen'):
            discriminator_gen.build(gen_result, train_mode=in_train_dis, parent=discriminator_truth)

        vgg_content = custom_vgg19.Vgg19(vgg19_npy_path=VGG_NPY_PATH)
        with tf.name_scope("content_vgg"):
            vgg_content.build(in_large)

        vgg_var = custom_vgg19.Vgg19(var_map=vgg_content.var_map)
        with tf.name_scope("variable_vgg"):
            vgg_var.build(gen_result)

        prob_truth = discriminator_truth.prob
        prob_gen = discriminator_gen.prob

        prob_truth_mean = tf.reduce_mean(prob_truth)
        prob_gen_mean = tf.reduce_mean(prob_gen)

        with tf.name_scope("cost"):
            gen_cost_content = tf.sqrt(tf.reduce_mean(tf.square(vgg_var.conv2_2 - vgg_content.conv2_2)))
            gen_cost_generator = -tf.log(tf.clip_by_value(prob_gen_mean, 1e-10, 1.0)) * 2
            gen_cost_invariant = get_invariant_cost2(gen_result)

            # for pre-train (purely with conv22): don't set these 2 cost
            # gen_cost_generator = tf.constant(0.0)  # for pre train
            # gen_cost_invariant = tf.constant(0.0)  # for pre train

            gen_cost = gen_cost_content + gen_cost_generator + gen_cost_invariant

            dis_cost = tf.reduce_mean(
                -(tf.log(prob_truth) + tf.log(tf.clip_by_value(1 - prob_gen, 1e-10, 1.0))))

        with tf.name_scope("train"):
            gen_step = tf.Variable(0, name='gen_step', trainable=False)
            gen_train = tf.train.AdamOptimizer(learning_rate=gen_learning_rate) \
                .minimize(gen_cost, gen_step, var_list=generator.var_list())

            dis_train = tf.train.AdamOptimizer(learning_rate=dis_learning_rate) \
                .minimize(dis_cost, var_list=discriminator_truth.get_all_var())

        print "Net generated: %d" % (time.time() - start_time)
        start_time = time.time()

        # analysis
        for name, var in generator.var_dict.items():
            tf.histogram_summary(name, var)
        for name, var in discriminator_truth.var_dict_name.items():
            tf.histogram_summary(name, var)
        tf.scalar_summary("gen_cost", gen_cost)
        tf.scalar_summary("gen_cost_content", gen_cost_content)
        tf.scalar_summary("gen_cost_generator", gen_cost_generator)
        tf.scalar_summary("gen_cost_invariant", gen_cost_invariant)
        tf.scalar_summary("dis_cost", dis_cost)
        tf.scalar_summary("prob_truth", prob_truth_mean)
        tf.scalar_summary("prob_gen", prob_gen_mean)
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(SUMMARY_FOLDER, graph=sess.graph)

        saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state(TRAIN_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print "save restored:" + ckpt.model_checkpoint_path
        else:
            tf.initialize_all_variables().run()
            print "all variables init"

        print "Var init: %d" % (time.time() - start_time)

        start_time = time.time()
        for i in xrange(80000):
            # disable this part for pre-train with conv22
            # train discriminator:
            feed_dict = {in_large: get_next_batch(ds), in_train_dis: True, in_train_gen: False}

            _, \
            dis_cost_out, \
            prob_truth_out, \
            prob_gen_out \
                = sess.run([
                dis_train,
                dis_cost,
                prob_truth_mean,
                prob_gen_mean
            ], feed_dict)

            print "dis-step:\t\t\t\t\t " \
                  "dis-cost:%.10f\t\t " \
                  "prob_gen:%.10f\t " \
                  "prob_truth:%.10f" \
                  % (
                      dis_cost_out,
                      prob_gen_out,
                      prob_truth_out
                  )

            if math.isnan(dis_cost_out):
                raise Exception("error found")

            # train generator:
            feed_dict = {in_large: get_next_batch(ds), in_train_dis: False, in_train_gen: True}

            step_out, \
            _, \
            gen_cost_out, \
            cost_content_out, \
            cost_generator_out, \
            cost_invariant_out, \
            prob_gen_out \
                = sess.run([
                gen_step,
                gen_train,
                gen_cost,
                gen_cost_content,
                gen_cost_generator,
                gen_cost_invariant,
                prob_gen_mean
            ], feed_dict)

            duration = time.time() - start_time
            print "step: %d, " \
                  "\t(%.1f sec)\t " \
                  "gen-cost:%.10f\t " \
                  "prob_gen:%.10f,\t " \
                  "gen_cost_content:%.2f,\t " \
                  "gen_cost_generator:%.5f,\t " \
                  "gen_cost_invariant:%.5f" \
                  % (
                      step_out,
                      duration,
                      gen_cost_out,
                      prob_gen_out,
                      cost_content_out,
                      cost_generator_out,
                      cost_invariant_out
                  )

            if math.isnan(gen_cost_out):
                raise Exception("error found")

            if i == 0 or i == 9 or i == 49 or step_out % 100 == 0:
                feed_dict[in_train_dis] = False
                feed_dict[in_train_gen] = False

                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step_out)

                if step_out % 2000 == 0:
                    generator.save_npy(sess, TRAIN_DIR + "/save-gen-%d.npy" % step_out)
                    discriminator_truth.save_npy(sess, TRAIN_DIR + "/save-dis-%d.npy" % step_out)
                else:
                    generator.save_npy(sess, TRAIN_DIR + "/save-gen.npy")
                    discriminator_truth.save_npy(sess, TRAIN_DIR + "/save-dis.npy")

                saved_path = saver.save(sess, TRAIN_DIR + "/saves", global_step=gen_step,
                                        write_meta_graph=False)
                print "net saved: " + saved_path

                # print image
                gen_out = sess.run(gen_result, feed_dict)
                img_in_path = TRAIN_DIR + "/%d-input.jpg" % step_out
                img_out_path = TRAIN_DIR + "/%d-output.jpg" % step_out
                skimage.io.imsave(img_in_path, feed_dict[in_large][0])
                skimage.io.imsave(img_out_path, gen_out[0])
                print "img saved:", img_in_path, img_out_path


def get_next_batch(ds):
    batch = ds.next_batch()
    while batch[0].shape != (SIZE, SIZE, 3):
        print 'in correct size found: ', batch[0].shape
        batch = ds.next_batch()
    return batch


def get_invariant_cost2(fast_output):
    h_filter = tf.constant([1, -1], tf.float32, [2, 1])
    h_filter = tf.reshape(h_filter, [2, 1, 1, 1])
    h_filter = tf.tile(h_filter, [1, 1, 3, 1])
    assert h_filter.get_shape().as_list() == [2, 1, 3, 1]

    w_filter = tf.constant([1, -1], tf.float32, [1, 2])
    w_filter = tf.reshape(w_filter, [1, 2, 1, 1])
    w_filter = tf.tile(w_filter, [1, 1, 3, 1])
    assert w_filter.get_shape().as_list() == [1, 2, 3, 1]

    return tf.reduce_mean(tf.square(tf.nn.conv2d(fast_output, h_filter, [1, 1, 1, 1], 'VALID'))) \
           + tf.reduce_mean(tf.square(tf.nn.conv2d(fast_output, w_filter, [1, 1, 1, 1], 'VALID')))


def create_tiles(large, height, width, num):
    h_stride = height / num
    w_stride = width / num
    t_tiles = []
    for y in xrange(num):
        row = []
        for x in xrange(num):
            t_tile = tf.slice(large, [0, y * h_stride, x * w_stride, 0], [1, h_stride, w_stride, 3])
            row.append(t_tile)
        t_tiles.append(row)
    return t_tiles


def train_loop():
    """
    Call this method to start the training
    """
    ds = DataSet(SIZE, SIZE, 1, directory=COCO2014_PATH)
    ds.start_loading()

    # for pre-train:
    # gen_learning_rate = 0.00001

    # for GAN training:
    dis_learning_rate = 0.00001
    gen_learning_rate = 0.00001

    trails = 0
    while True:
        # sometimes a too large learning rate would create NAN is the network. the histogram will throw exception
        # use this method to adjust the training rate and retry:
        try:
            trails += 1
            dis_learning_rate_trails = dis_learning_rate / 2 ** trails
            gen_lr_rate_trails = gen_learning_rate / 2 ** trails
            print 'start new train: trail=%d, dis-lr=%d, gen-lr=%d' % (
                trails, dis_learning_rate_trails, gen_lr_rate_trails)
            train(ds, dis_learning_rate=dis_learning_rate_trails,
                  gen_learning_rate=gen_lr_rate_trails)
        except:
            import traceback

            print traceback.format_exc()

            tf.reset_default_graph()


def view():
    """
    Call this method to view the data with Tensorboard. Can be called during training
    :return:
    """
    import tensorflow.tensorboard.tensorboard as tb
    tb.flags.FLAGS.__setattr__("logdir", SUMMARY_FOLDER)
    tb.main()


if __name__ == "__main__":
    # view()
    train_loop()
