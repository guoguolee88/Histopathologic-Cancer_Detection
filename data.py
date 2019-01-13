from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import os
import random
import numpy as np

from PIL import Image


class Data(object):
    def __init__(self, dataset_dir, labels, image_to_label, validation_percentage):
        self.dataset_dir = dataset_dir
        self.label_to_index = {}
        self._prepare_data(labels, image_to_label, validation_percentage)

    def get_data(self, mode):
        return self.data_index[mode]

    def get_size(self, mode):
        """Calculates the number of samples in the dataset partition.
        Args:
          mode: Which partition, must be 'training', 'validation'.
        Returns:
          Number of samples in the partition.
        """
        return len(self.data_index[mode])

    def _prepare_data(self, labels, image_to_label, validation_percentage):
        for index, label in enumerate(labels):
            self.label_to_index[label] = index
        self.data_index = {'validation': [], 'training': []}

        images = os.listdir(self.dataset_dir)
        random.shuffle(images)

        val_num = len(images) // validation_percentage
        for idx, img in enumerate(images):
            image_path = os.path.join(self.dataset_dir, img)

            lbl = image_to_label[img[:-4]]
            if idx < val_num:
                self.data_index['validation'].append({'label': lbl, 'file': image_path})
            else:
                self.data_index['training'].append({'label': lbl, 'file': image_path})

        for set_index in ['validation', 'training']:
            random.shuffle(self.data_index[set_index])

        tf.logging.info("data prepared.")


class Dataset(object):
    """
    Wrapper class around the new Tensorflows dataset pipeline.

    Handles loading, partitioning, and preparing training data.
    """

    def __init__(self, dataset, height, weight, batch_size, shuffle=True):
        self.data_size = dataset.get_size('training')
        self.resize_h = height
        self.resize_w = weight

        images, labels = self._get_data(dataset.get_data('training'),
                                        dataset.label_to_index)

        # create dataset, Creating a source
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.map(self._parse_func, num_parallel_calls=8)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=(int(self.data_size * 0.4) + 3 * batch_size))
        dataset = dataset.batch(batch_size)
        self.dataset = dataset.repeat()


    def _get_data(self, data, label_to_index):
        num = len(data)

        images = []
        labels = []
        for idx in range(num):
            img = data[idx]
            images.append(img['file'])
            labels.append(label_to_index[img['label']])

        # # convert lists to TF tensor
        # image_paths = convert_to_tensor(image_paths, dtype=dtypes.string)
        # labels = convert_to_tensor(labels, dtype=dtypes.float64)

        return images, labels


    def _parse_func(self, filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_png(image_string, channels=3)
        # cropped_image = tf.image.central_crop(image_decoded, 0.7)
        # rotated_image = tf.image.rot90(image_decoded, 1)
        resized_image = tf.image.resize_images(image_decoded,
                                               [self.resize_h, self.resize_w])
        # image = tf.cast(image_decoded, tf.float32)
        image = tf.image.convert_image_dtype(resized_image, dtype=tf.float32)
        # Finally, rescale to [-1,1] instead of [0, 1)
        # image = tf.subtract(image, 0.5)
        # image = tf.multiply(image, 2.0)
        return image, label
