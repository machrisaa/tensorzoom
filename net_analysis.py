import time

import skimage
import skimage.io
import skimage.transform
import tensorflow as tf
import os

from tensorzoom_net import TensorZoomNet


def load_image(path, height=None, width=None):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    if height is not None and width is not None:
        ny = height
        nx = width
    elif height is not None:
        ny = height
        nx = img.shape[1] * ny / img.shape[0]
    elif width is not None:
        nx = width
        ny = img.shape[0] * nx / img.shape[1]
    else:
        ny = img.shape[0]
        nx = img.shape[1]
    return skimage.transform.resize(img, (ny, nx))


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


def render(pb_path, img_path):
    with tf.Session() as sess:
        img = load_image(img_path)
        contents = tf.expand_dims(tf.constant(img, tf.float32), 0)

        net = TensorZoomNet(pb_path, False)
        net.build(contents)
        fast_output = net.output

        start_time = time.time()
        output = sess.run(fast_output)
        duration = time.time() - start_time
        print "output calculated: %.10f sec" % duration

        # print image
        _, pb_name = os.path.split(pb_path)
        pb_name, _ = os.path.splitext(pb_name)
        name, ext = os.path.splitext(img_path)
        out_path = name + "_" + pb_name + ext
        skimage.io.imsave(out_path, output[0])
        print "img saved:", out_path


def render_sliced(pb_path, img_path, side_num):
    with tf.Session() as sess:
        img = load_image(img_path)
        contents = tf.expand_dims(tf.constant(img, tf.float32), 0)

        # use stitch training method, slice the image into tiles and concat as batches
        tiles = create_tiles(contents, img.shape[0], img.shape[1], side_num)
        batch = tf.concat(0, [tf.concat(0, tiles[y]) for y in xrange(side_num)])  # row1, row2, ...

        net = TensorZoomNet(pb_path, False)
        net.build(batch)

        # stitch the tiles back together after split the batches
        split = tf.split(0, side_num ** 2, net.output)
        fast_output = tf.concat(1, [
            tf.concat(2, [split[x] for x in xrange(side_num * y, side_num * y + side_num)])
            for y in xrange(side_num)])

        start_time = time.time()
        output = sess.run(fast_output)
        duration = time.time() - start_time
        print "output calculated: %.10f sec" % duration

        # print image
        _, pb_name = os.path.split(pb_path)
        pb_name, _ = os.path.splitext(pb_name)
        name, ext = os.path.splitext(img_path)
        out_path = name + "_" + pb_name + ext
        skimage.io.imsave(out_path, output[0])
        print "img saved:", out_path


if __name__ == "__main__":
    # with tf.device("/gpu:0"):
    with tf.device("/cpu:0"):
        # for small image/ icon/ thumbnail, use non-deblur version has better result
        render(pb_path='./results/tz6-s-stitch/tz6-s-stitch-gen.npy', img_path="./analysis/cat_h.jpg")

        # example for large image / photos from camera, deblur version looks better
        # warning: this example will consume lots of memory (around 9.xxGB)
        # render(pb_path='./results/tz6-s-stitch-sblur-lowtv/tz6-s-stitch-sblur-lowtv-gen.npy',
        #             img_path="./analysis/london2.jpg")

        # instead, slice the image into 4 smaller images and then join together to form a big one
        # less memory is used (<1GB) but there will be defects on the boundary of the tiles
        # render_sliced(pb_path='./results/tz6-s-stitch-sblur-lowtv/tz6-s-stitch-sblur-lowtv-gen.npy',
        #               img_path="./analysis/london2.jpg", side_num=4)
