from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Dataset(object):
    """
    Wrapper class around the new Tensorflows dataset pipeline.

    Handles loading, partitioning, and preparing training data.
    """

    def __init__(self, tfrecord_path, batch_size, height, width):
        self.resize_h = height
        self.resize_w = width

        dataset = tf.data.TFRecordDataset(tfrecord_path,
                                          compression_type='GZIP',
                                          num_parallel_reads=batch_size * 4)
        # dataset = dataset.map(self._parse_func, num_parallel_calls=8)
        # The map transformation takes a function and applies it to every element
        # of the dataset.
        dataset = dataset.map(self.decode, num_parallel_calls=8)
        dataset = dataset.map(self.augment, num_parallel_calls=8)
        dataset = dataset.map(self.normalize, num_parallel_calls=8)

        # Prefetches a batch at a time to smooth out the time taken to load input
        # files for shuffling and processing.
        dataset = dataset.prefetch(buffer_size=batch_size)
        # The shuffle transformation uses a finite-sized buffer to shuffle elements
        # in memory. The parameter is the number of elements in the buffer. For
        # completely uniform shuffling, set the parameter to be the same as the
        # number of elements in the dataset.
        dataset = dataset.shuffle(1000 + 3 * batch_size)
        dataset = dataset.repeat(1)
        self.dataset = dataset.batch(batch_size)


    def decode(self, serialized_example):
        """Parses an image and label from the given `serialized_example`."""
        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'image/filename': tf.FixedLenFeature([], tf.string),
                'image/encoded': tf.FixedLenFeature([], tf.string),
                # 'image/label': tf.FixedLenFeature([], tf.int64),
            })

        # Convert from a scalar string tensor to a float32 tensor with shape
        image_decoded = tf.image.decode_png(features['image/encoded'], channels=3)
        image = tf.image.resize_images(image_decoded, [self.resize_h, self.resize_w])

        filename = features['image/filename']

        return image, filename


    def augment(self, image, filename):
        """Placeholder for data augmentation."""
        # OPTIONAL: Could reshape into a 28x28 image and apply distortions
        # here.  Since we are not applying any distortions in this
        # example, and the next step expects the image to be flattened
        # into a vector, we don't bother.
        image = tf.image.central_crop(image, 0.5)
        image = tf.image.resize_images(image, [self.resize_h, self.resize_w])

        return image, filename


    def normalize(self, image, filename):
        """Convert `image` from [0, 255] -> [-0.5, 0.5] floats."""
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
        return image, filename
