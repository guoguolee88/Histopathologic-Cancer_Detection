"""
Convert PatchCamelyon (PCam) dataset to TFRecord for classification.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os
import random

from PIL import Image, ImageStat
import tensorflow as tf

from dataset_tools import dataset_util


RANDOM_SEED = 4357


TRAIN = 'train'
VALIDATE = 'validate'
TEST = 'test'


flags = tf.app.flags
flags.DEFINE_string('dataset_dir',
                    '/home/ace19/dl_data/histopathologic_cancer_detection',
                    'Root Directory to raw PCam dataset.')
flags.DEFINE_string('output_path',
                    '/home/ace19/dl_data/histopathologic_cancer_detection/' + VALIDATE + '.record',
                    'Path to output TFRecord')
flags.DEFINE_string('label_map_path',
                    '/home/ace19/dl_data/histopathologic_cancer_detection/train_labels.csv',
                    # None,
                    'Path to label map')

FLAGS = flags.FLAGS


def get_label_map_dict(label_map_path):
    # read labels from .cvs
    label_map_dict = {}
    with open(label_map_path, 'r') as reader:
        for line in reader:
            fields = line.strip().split(',')
            label_map_dict[fields[0]] = fields[1]

    return label_map_dict


def dict_to_tf_example(image_name,
                       dataset_directory,
                       label_map_dict=None,
                       image_subdirectory=VALIDATE):
    """
    Args:
      image: a single image name
      dataset_directory: Path to root directory holding PCam dataset
      label_map_dict: A map from string label names to integers ids.
      image_subdirectory: String specifying subdirectory within the
        PCam dataset directory holding the actual image data.

    Returns:
      example: The converted tf.Example.

    Raises:
      ValueError: if the image pointed to by image is not a valid PNG
    """
    full_path = os.path.join(dataset_directory, image_subdirectory, image_name)
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded = fid.read()
    encoded_io = io.BytesIO(encoded)
    image = Image.open(encoded_io)
    width, height = image.size
    format = image.format
    image_stat = ImageStat.Stat(image)
    mean = image_stat.mean
    std = image_stat.stddev
    if image.format != 'PNG':
        raise ValueError('Image format not PNG')
    key = hashlib.sha256(encoded).hexdigest()
    if image_subdirectory.lower() == VALIDATE:
        label = int(label_map_dict[image_name[:-4]])
    else:
        label = -1

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(image_name.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(image_name.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded),
        'image/format': dataset_util.bytes_feature(format.encode('utf8')),
        'image/label': dataset_util.int64_feature(label),
        # 'image/text': dataset_util.bytes_feature('label_text'.encode('utf8'))
        'image/mean': dataset_util.float_list_feature(mean),
        'image/std': dataset_util.float_list_feature(std)
    }))
    return example


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path, options=options)

    label_map_dict = None
    if FLAGS.label_map_path:
        label_map_dict = get_label_map_dict(FLAGS.label_map_path)

    tf.logging.info('Reading from PCam dataset.')
    dataset_path = os.path.join(FLAGS.dataset_dir, VALIDATE)
    filenames = sorted(os.listdir(dataset_path))
    random.seed(RANDOM_SEED)
    random.shuffle(filenames)

    for idx, image in enumerate(filenames):
        if idx % 100 == 0:
            tf.logging.info('On image %d of %d', idx, len(filenames))

        tf_example = dict_to_tf_example(image, FLAGS.dataset_dir, label_map_dict)
        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()
