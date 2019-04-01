import os
import sys
import argparse

from PIL import Image

import tensorflow as tf

FLAGS = None


def main(_):
    dir = os.path.join(FLAGS.target_path, FLAGS.src_dir)
    target_list = os.listdir(dir)
    total = len(target_list)
    for i, target in enumerate(target_list):
        path = os.path.join(dir, target)
        outfile = os.path.join(FLAGS.target_path, FLAGS.target_dir, target[:-4] + '.png')
        try:
            im = Image.open(path)
            print("Generating .png image for %s, %d/%d" % (target, i, total))
            im.thumbnail(im.size)
            im.save(outfile, "PNG", quality=100)
        except Exception as e:
            print(e)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--target_path',
      type=str,
      default='/home/ace19/dl_data/histopathologic_cancer_detection_ori',
      help='Where is images to convert.')
  parser.add_argument(
      '--src_dir',
      type=str,
      default='raw_train',
      help='Where is src dir to convert.')
  parser.add_argument(
      '--target_dir',
      type=str,
      default='train',
      help='Where is target dir to convert.')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
