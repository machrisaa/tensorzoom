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


def view_layers(pb_path, img_path):
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


if __name__ == "__main__":
    # with tf.device("/gpu:0"):
    with tf.device("/cpu:0"):
        # for small image/ icon/ thumbnail, use non-deblur version has better result
        # view_layers(pb_path='./results/tz6-s-stitch/tz6-s-stitch-gen.npy', img_path="./analysis/cat_h.jpg")

        # for large image/ photos from camera, deblur version looks better
        view_layers(pb_path='./results/tz6-s-stitch-sblur-lowtv/tz6-s-stitch-sblur-lowtv-gen.npy',
                    img_path="./analysis/london2.jpg")
