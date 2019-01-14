import os
import sys
import argparse

from PIL import Image

import tensorflow as tf

FLAGS = None


def main(_):
    dir = os.path.join(FLAGS.target_dir, 'train_ori')
    target_list = os.listdir(dir)
    for target in target_list:
        path = os.path.join(dir, target)
        outfile = os.path.join(FLAGS.target_dir, 'train', target[:-4] + '.png')
        try:
            im = Image.open(path)
            print("Generating png for %s" % target)
            im.thumbnail(im.size)
            im.save(outfile, "PNG", quality=100)
        except Exception as e:
            print(e)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--target_dir',
    type=str,
    default='/home/ace19/dl_data/histopathologic_cancer_detection',
    help='Where is images to convert.')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
