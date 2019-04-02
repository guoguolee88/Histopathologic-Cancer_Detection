from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import numpy as np

import argparse
import os
import sys
import datetime
import csv

import eval_data
import model
from utils import aug_utils

slim = tf.contrib.slim


FLAGS = None

PCAM_EVAL_DATA_SIZE = 57458


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    labels = FLAGS.labels.split(',')
    num_classes = len(labels)

    X = tf.placeholder(tf.float32, [None, FLAGS.height, FLAGS.width, 3])

    logits, _ = model.hcd_model(X,
                                num_classes=num_classes,
                                is_training=False,
                                keep_prob=1.0)

    predicted_labels = tf.argmax(logits, axis=1, name='prediction')
    # prediction = tf.nn.softmax(logits)
    # predicted_labels = tf.argmax(prediction, 1)

    ###############
    # Prepare data
    ###############
    filenames = tf.placeholder(tf.string, shape=[])
    eval_dataset = eval_data.Dataset(filenames,
                                     FLAGS.batch_size,
                                     FLAGS.height,
                                     FLAGS.width)
    iterator = eval_dataset.dataset.make_initializable_iterator()
    next_batch = iterator.get_next()

    # TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
    sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session(config=sess_config) as sess:
        # sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        # Create a saver object which will save all the variables
        saver = tf.train.Saver()
        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path
        saver.restore(sess, checkpoint_path)

        global_step = checkpoint_path.split('/')[-1].split('-')[-1]

        # ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        # if ckpt and ckpt.model_checkpoint_path:
        #     saver.restore(sess, ckpt.model_checkpoint_path)
        #     # Assuming model_checkpoint_path looks something like:
        #     #   /my-favorite-path/imagenet_train/model.ckpt-0,
        #     # extract global_step from it.
        #     global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        #     print('Successfully loaded model from %s at step=%s.' % (
        #         ckpt.model_checkpoint_path, global_step))
        # else:
        #     print('No checkpoint file found at %s' % FLAGS.checkpoint_path)
        #     return

        # Get the number of prediction steps
        batches = int(PCAM_EVAL_DATA_SIZE / FLAGS.batch_size)
        if PCAM_EVAL_DATA_SIZE % FLAGS.batch_size > 0:
            batches += 1

        ##################################################
        # prediction & make results into csv file.
        ##################################################
        start_time = datetime.datetime.now()
        print("Start prediction: {}".format(start_time))

        id2name = {i: name for i, name in enumerate(labels)}
        submission = {}

        eval_filenames = os.path.join(FLAGS.dataset_dir, 'test.record')

        # Test Time Augmentation (TTA)
        predictions = []
        for i in range(FLAGS.num_tta):
            tf.logging.info('Start TTA %d : ' % i)
            sess.run(iterator.initializer, feed_dict={filenames: eval_filenames})

            batch_pred = []
            batch_filename = []
            for i in range(batches):
                batch_xs, filename = sess.run(next_batch)
                # # Verify image
                # n_batch = batch_xs.shape[0]
                # for i in range(n_batch):
                #     img = batch_xs[i]
                #     # scipy.misc.toimage(img).show()
                #     # Or
                #     img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
                #     cv2.imwrite('/home/ace19/Pictures/' + str(i) + '.png', img)
                #     # cv2.imshow(str(fnames), img)
                #     cv2.waitKey(100)
                #     cv2.destroyAllWindows()

                # random augmentation for TTA
                augmented_batch_xs = aug_utils.aug(batch_xs)

                pred = sess.run(predicted_labels, feed_dict={X: augmented_batch_xs})

                batch_pred.extend(pred)
                batch_filename.extend(filename)

            predictions.append(batch_pred)

        pred = np.mean(predictions, axis=0) # [0:57458]
        # TODO: TTA 계산하는 법 리서치 필요.
        # log_preds,y = learn.TTA()
        # probs = np.mean(np.exp(log_preds),0)
        pred1 = np.ceil(pred)   # ??

        size = len(batch_filename)
        for n in range(size):
            submission[batch_filename[n].decode('UTF-8')[:-4]] = id2name[pred1[n]]

        tf.logging.info('Total count: #%d' % size)

        end_time = datetime.datetime.now()
        tf.logging.info('#%d Data, End prediction: %s' % (PCAM_EVAL_DATA_SIZE, end_time))
        tf.logging.info('prediction waste time: %s' % (end_time - start_time))


    ######################################
    # make submission.csv for kaggle
    ######################################
    if not os.path.exists(FLAGS.result_dir):
        os.makedirs(FLAGS.result_dir)

    fout = open(
        os.path.join(FLAGS.result_dir,
                     FLAGS.model_architecture + '-#' +
                     global_step + '.csv'),
        'w', encoding='utf-8', newline='')
    writer = csv.writer(fout)
    writer.writerow(['id', 'label'])
    for key in sorted(submission.keys()):
        writer.writerow([key, submission[key]])
    fout.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default='/home/ace19/dl_data/histopathologic_cancer_detection',
        help="""\
            Where to find the image testing data to.
            """)
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default=os.getcwd() + '/models',
        help='Directory where to read training checkpoints.')
    parser.add_argument(
        '--model_architecture',
        type=str,
        default='resnet_v2_50',
        help='What model architecture to use')
    parser.add_argument(
        '--height',
        type=int,
        default=96,
        help='how do you want image resize height.')
    parser.add_argument(
        '--width',
        type=int,
        default=96,
        help='how do you want image resize width.')
    parser.add_argument(
        '--labels',
        type=str,
        default='0,1',
        help='Labels to use', )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='How many items to predict with at once')
    parser.add_argument(
        '--num_tta',    # Test Time Augmentation
        type=int,
        default=3,
        help='Number of Test Time Augmentation', )
    parser.add_argument(
        '--result_dir',
        type=str,
        default=os.getcwd() + '/result',
        help='Directory to write submission.csv file.')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
