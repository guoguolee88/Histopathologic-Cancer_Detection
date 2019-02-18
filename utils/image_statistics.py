import numpy as np
import os
import pandas as pd
import cv2
from sklearn.utils import shuffle
from tqdm import tqdm_notebook

import tensorflow as tf


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('label_path',
                    '/home/ace19/dl_data/histopathologic_cancer_detection/train_labels.csv',
                    # None,
                    'Path to label')
flags.DEFINE_string('train_path',
                    '/home/ace19/dl_data/histopathologic_cancer_detection/train',
                    # None,
                    'Path to train image')


def readImage(image_path):
    # OpenCV reads the image in bgr format by default
    bgr_img = cv2.imread(image_path)
    # We flip it to rgb for visualization purposes
    b, g, r = cv2.split(bgr_img)
    rgb_img = cv2.merge([r, g, b])

    return rgb_img / 255


def get_statistics(shuffled_data):
    # As we count the statistics, we can check if there are any completely black or white images
    dark_th = 10 / 255      # If no pixel reaches this threshold, image is considered too dark
    bright_th = 245 / 255   # If no pixel is under this threshold, image is considerd too bright
    too_dark_idx = []
    too_bright_idx = []

    x_tot = np.zeros(3)
    x2_tot = np.zeros(3)
    counted_ones = 0
    for i, idx in tqdm_notebook(enumerate(shuffled_data['id']), 'computing statistics...(220025 it total)'):
        path = os.path.join(FLAGS.train_path, idx)
        imagearray = readImage(path + '.png').reshape(-1 ,3)
        # is this too dark
        if(imagearray.max() < dark_th):
            too_dark_idx.append(idx)
            continue # do not include in statistics
        # is this too bright
        if(imagearray.min() > bright_th):
            too_bright_idx.append(idx)
            continue # do not include in statistics
        x_tot += imagearray.mean(axis=0)
        x2_tot += (imagearray**2).mean(axis=0)
        counted_ones += 1

    channel_avr = x_tot / counted_ones
    channel_std = np.sqrt(x2_tot / counted_ones - channel_avr**2)

    return channel_avr, channel_std


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    data = pd.read_csv(FLAGS.label_path)
    # random sampling
    shuffled_data = shuffle(data)
    channel_avr, channel_std = get_statistics(shuffled_data)

    # MEAN = [0.70244707, 0.54624322, 0.69645334]
    # STD = [0.23889325, 0.28209431, 0.21625058]
    tf.logging.info('avr: %s, std: %s' % (channel_avr, channel_std))


if __name__ == '__main__':
    tf.app.run()