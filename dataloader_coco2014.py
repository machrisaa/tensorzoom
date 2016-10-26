"""
Data loader for the The Microsoft Coco2014 data set: http://mscoco.org/dataset/#download
We are using 2014 Training images [80K/13GB] data set.
"""

import os
import random
from skimage import io
import numpy as np
import Queue
import thread


def load_image(path, height, width):
    # load image
    img = io.imread(path)
    img = img / 255.0

    # center crop
    shape = np.shape(img)
    h = shape[0]
    w = shape[1]
    y0 = (h - height) / 2
    x0 = (w - width) / 2
    img = img[y0:y0 + height, x0:x0 + width, :]

    # remove monotone
    if len(np.shape(img)) == 2:
        print "monotone found"
        coloured = np.expand_dims(img, 2)
        img = np.concatenate((coloured, coloured, coloured), 2)
    return img


class DataSet:
    def __init__(self, height, width, batch_size, directory, buffer_size=10):
        self.files = [os.path.join(directory, f) for f in os.listdir(directory)]
        self.shuffle()
        self.pos = 0

        self.height = height
        self.width = width

        self.batch_size = batch_size
        self.queue = Queue.Queue(buffer_size)
        self.stopped = False
        self.print_path = False

    def shuffle(self):
        random.shuffle(self.files)

    def _prepare_next_batch(self):
        batch = self.files[self.pos:self.pos + self.batch_size]
        extra = self.batch_size - len(batch)
        if extra == 0:
            self.shuffle()
            batch += self.files[0:extra]  # assume next batch will be enough
            self.pos = extra
        else:
            self.pos += self.batch_size

        img_batch = [load_image(path, self.height, self.width) for path in batch]

        # return img_batch
        self.queue.put((img_batch, batch))

    def _prepare(self):
        while not self.stopped:
            try:
                self._prepare_next_batch()
            except:
                print 'error found in batch: ', self.files[self.pos:self.pos + self.batch_size]
                pass

    def start_loading(self):
        thread.start_new_thread(self._prepare, ())

    def stop_loading(self):
        self.stopped = True

    def set_print_path(self, enable):
        self.print_path = enable

    def next_batch(self):
        (img_batch, batch) = self.queue.get()
        if self.print_path:
            for path in batch: print path
        return img_batch

# if __name__ == '__main__':
#     ds = DataSet(256, 256, 10, '../../datasets/coco2014/train2014')
#     print ds.files[0]
#     print ds.files[1]
#
#     batch = ds.next_batch()
#     print len(batch)
