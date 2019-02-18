from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import os
import cv2
import numpy as np

from slim.nets import inception_v4

import data
from utils import train_utils

slim = tf.contrib.slim


flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_string('train_logdir', './models',
                    'Where the checkpoint and logs are stored.')
flags.DEFINE_string('ckpt_name_to_save', 'inception_v4.ckpt',
                    'Name to save checkpoint file')
flags.DEFINE_integer('log_steps', 10,
                     'Display logging information at every log_steps.')
flags.DEFINE_integer('save_interval_secs', 1200,
                     'How often, in seconds, we save the model to disk.')
flags.DEFINE_boolean('save_summaries_images', False,
                     'Save sample inputs, labels, and semantic predictions as '
                     'images to summary.')
flags.DEFINE_string('summaries_dir', './models/train_logs',
                     'Where to save summary logs for TensorBoard.')

flags.DEFINE_enum('learning_policy', 'poly', ['poly', 'step'],
                  'Learning rate policy for training.')
flags.DEFINE_float('base_learning_rate', .001,
                   'The base learning rate for model training.')
flags.DEFINE_float('learning_rate_decay_factor', 0.01,
                   'The rate to decay the base learning rate.')
flags.DEFINE_float('learning_rate_decay_step', .2000,
                   'Decay the base learning rate at a fixed step.')
flags.DEFINE_float('learning_power', 0.9,
                   'The power value used in the poly learning policy.')
flags.DEFINE_float('training_number_of_steps', 30000,
                   'The number of steps used for training.')
flags.DEFINE_float('momentum', 0.9, 'The momentum value to use')

flags.DEFINE_float('last_layer_gradient_multiplier', 1.0,
                   'The gradient multiplier for last layers, which is used to '
                   'boost the gradient of last layers if the value > 1.')

# Set to False if one does not want to re-use the trained classifier weights.
flags.DEFINE_boolean('initialize_last_layer', True,
                     'Initialize the last layer.')
flags.DEFINE_boolean('last_layers_contain_logits_only', False,
                     'Only consider logits as last layers or not.')
flags.DEFINE_integer('slow_start_step', 0,
                     'Training model with small learning rate for few steps.')
flags.DEFINE_float('slow_start_learning_rate', 1e-4,
                   'Learning rate employed during slow start.')

# Settings for fine-tuning the network.
flags.DEFINE_string('pre_trained_checkpoint',
                    './pre-trained/inception_v4.ckpt',
                    # None,
                    'The pre-trained checkpoint in tensorflow format.')
flags.DEFINE_string('checkpoint_exclude_scopes',
                    'InceptionV4/AuxLogits,InceptionV4/logits',
                    # None,
                    'Comma-separated list of scopes of variables to exclude '
                    'when restoring from a checkpoint.')
flags.DEFINE_string('trainable_scopes',
                    # 'ssd_300_vgg/block4_box, ssd_300_vgg/block7_box, \
                    #  ssd_300_vgg/block8_box, ssd_300_vgg/block9_box, \
                    #  ssd_300_vgg/block10_box, ssd_300_vgg/block11_box',
                    None,
                    'Comma-separated list of scopes to filter the set of variables '
                    'to train. By default, None would train all the variables.')
flags.DEFINE_string('checkpoint_model_scope',
                    None,
                    'Model scope in the checkpoint. None if the same as the trained model.')
flags.DEFINE_string('model_name',
                    'InceptionV4',
                    'The name of the architecture to train.')
flags.DEFINE_boolean('ignore_missing_vars',
                     False,
                     'When restoring a checkpoint would ignore missing variables.')

# Dataset settings.
flags.DEFINE_string('dataset_dir',
                    '/home/ace19/dl_data/histopathologic_cancer_detection',
                    'Where the dataset reside.')

flags.DEFINE_integer('how_many_training_epochs', 200,
                     'How many training loops to run')
flags.DEFINE_integer('batch_size', 64, 'batch size')
flags.DEFINE_integer('height', 224, 'height')
flags.DEFINE_integer('width', 224, 'width')
flags.DEFINE_string('labels', 'negative,positive', 'Labels to use')


# temporary constant
# PCAM_TRAIN_DATA_SIZE = 220025
PCAM_TRAIN_DATA_SIZE = 50000
PCAM_VALIDATE_DATA_SIZE = 2011


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    labels = FLAGS.labels.split(',')
    num_classes = len(labels)

    tf.gfile.MakeDirs(FLAGS.train_logdir)
    tf.logging.info('Creating train logdir: %s', FLAGS.train_logdir)

    with tf.Graph().as_default() as graph:
        global_step = tf.train.get_or_create_global_step()

        X = tf.placeholder(tf.float32, [None, FLAGS.height, FLAGS.width, 3])
        ground_truth = tf.placeholder(tf.int64, [None], name='ground_truth')
        is_training = tf.placeholder(tf.bool)
        dropout_keep_prob = tf.placeholder(tf.float32, [])
        # learning_rate = tf.placeholder(tf.float32, [])

        with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
            logits, end_points = \
                inception_v4.inception_v4(X,
                                          num_classes=num_classes,
                                          is_training=is_training,
                                          dropout_keep_prob=dropout_keep_prob)

        # Print name and shape of each tensor.
        tf.logging.info("++++++++++++++++++++++++++++++++++")
        tf.logging.info("Layers")
        tf.logging.info("++++++++++++++++++++++++++++++++++")
        for k, v in end_points.items():
            tf.logging.info('name = %s, shape = %s' % (v.name, v.get_shape()))

        # Print name and shape of parameter nodes  (values not yet initialized)
        tf.logging.info("++++++++++++++++++++++++++++++++++")
        tf.logging.info("Parameters")
        tf.logging.info("++++++++++++++++++++++++++++++++++")
        for v in slim.get_model_variables():
            tf.logging.info('name = %s, shape = %s' % (v.name, v.get_shape()))

        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        prediction = tf.argmax(logits, 1, name='prediction')
        correct_prediction = tf.equal(prediction, ground_truth)
        confusion_matrix = tf.confusion_matrix(
            ground_truth, prediction, num_classes=num_classes)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        summaries.add(tf.summary.scalar('accuracy', accuracy))

        # Define loss
        tf.losses.sparse_softmax_cross_entropy(labels=ground_truth,
                                               logits=logits)

        # Gather update_ops. These contain, for example,
        # the updates for the batch_norm variables created by model.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # # Add summaries for model variables.
        # for model_var in slim.get_model_variables():
        #     summaries.add(tf.summary.histogram(model_var.op.name, model_var))

        # Add summaries for losses.
        for loss in tf.get_collection(tf.GraphKeys.LOSSES):
            summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

        learning_rate = train_utils.get_model_learning_rate(
            FLAGS.learning_policy, FLAGS.base_learning_rate,
            FLAGS.learning_rate_decay_step, FLAGS.learning_rate_decay_factor,
            FLAGS.training_number_of_steps, FLAGS.learning_power,
            FLAGS.slow_start_step, FLAGS.slow_start_learning_rate)
        optimizer = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)
        summaries.add(tf.summary.scalar('learning_rate', learning_rate))

        for variable in slim.get_model_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))

        total_loss, grads_and_vars = train_utils.optimize(optimizer)
        total_loss = tf.check_numerics(total_loss, 'Loss is inf or nan.')
        summaries.add(tf.summary.scalar('total_loss', total_loss))

        # Modify the gradients for biases and last layer variables.
        last_layers = train_utils.get_extra_layer_scopes(
            FLAGS.last_layers_contain_logits_only)
        grad_mult = train_utils.get_model_gradient_multipliers(
            last_layers, FLAGS.last_layer_gradient_multiplier)
        if grad_mult:
            grads_and_vars = slim.learning.multiply_gradients(
                grads_and_vars, grad_mult)

        # Create gradient update op.
        grad_updates = optimizer.apply_gradients(
            grads_and_vars, global_step=global_step)
        update_ops.append(grad_updates)
        update_op = tf.group(*update_ops)
        with tf.control_dependencies([update_op]):
            train_op = tf.identity(total_loss, name='train_op')

        # Add the summaries. These contain the summaries
        # created by model and either optimize() or _gather_loss().
        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # Merge all summaries together.
        summary_op = tf.summary.merge(list(summaries))
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir, graph)
        validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation', graph)

        ###############
        # Prepare data
        ###############
        tfrecord_filenames = tf.placeholder(tf.string, shape=[])
        dataset = data.Dataset(tfrecord_filenames, FLAGS.batch_size, FLAGS.how_many_training_epochs,
                                  FLAGS.height, FLAGS.width)
        iterator = dataset.dataset.make_initializable_iterator()
        next_batch = iterator.get_next()

        sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        with tf.Session(config = sess_config) as sess:
            sess.run(tf.global_variables_initializer())

            # Create a saver object which will save all the variables
            saver = tf.train.Saver()
            if FLAGS.pre_trained_checkpoint:
                train_utils.restore_fn(FLAGS)

            start_epoch = 0
            # Get the number of training/validation steps per epoch
            tr_batches = int(PCAM_TRAIN_DATA_SIZE / FLAGS.batch_size)
            if PCAM_TRAIN_DATA_SIZE % FLAGS.batch_size > 0:
                tr_batches += 1
            val_batches = int(PCAM_VALIDATE_DATA_SIZE / FLAGS.batch_size)
            if PCAM_VALIDATE_DATA_SIZE % FLAGS.batch_size > 0:
                val_batches += 1

            # The filenames argument to the TFRecordDataset initializer can either be a string,
            # a list of strings, or a tf.Tensor of strings.
            train_record_filenames = os.path.join(FLAGS.dataset_dir, 'train.record')
            validate_record_filenames = os.path.join(FLAGS.dataset_dir, 'validate.record')
            ############################
            # Training loop.
            ############################
            for num_epoch in range(start_epoch, FLAGS.how_many_training_epochs):
                print("------------------------------------")
                print(" Epoch {} ".format(num_epoch))
                print("------------------------------------")

                sess.run(iterator.initializer, feed_dict={tfrecord_filenames: train_record_filenames})
                for step in range(tr_batches):
                    train_batch_xs, train_batch_ys = sess.run(next_batch)

                    # # Verify image
                    # # assert not np.any(np.isnan(train_batch_xs))
                    # n_batch = train_batch_xs.shape[0]
                    # # n_view = train_batch_xs.shape[1]
                    # for i in range(n_batch):
                    #     img = train_batch_xs[i]
                    #     # scipy.misc.toimage(img).show()
                    #     # Or
                    #     img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
                    #     cv2.imwrite('/home/ace19/Pictures/' + str(i) + '.png', img)
                    #     # cv2.imshow(str(train_batch_ys[idx]), img)
                    #     cv2.waitKey(100)
                    #     cv2.destroyAllWindows()

                    # # Run the graph with this batch of training data.
                    lr, train_summary, train_accuracy, train_loss, _ = \
                        sess.run([learning_rate, summary_op, accuracy, total_loss, train_op],
                                 feed_dict={
                                     X: train_batch_xs,
                                     ground_truth: train_batch_ys,
                                     # learning_rate:FLAGS.base_learning_rate,
                                     is_training: True,
                                     dropout_keep_prob: 0.5
                                 })

                    train_writer.add_summary(train_summary, num_epoch)
                    tf.logging.info('Epoch #%d, Step #%d, rate %.15f, accuracy %.1f%%, loss %f' %
                                    (num_epoch, step, lr, train_accuracy * 100, train_loss))

                ###################################################
                # Validate the model on the validation set
                ###################################################
                tf.logging.info('--------------------------')
                tf.logging.info(' Start validation ')
                tf.logging.info('--------------------------')
                # Reinitialize iterator with the validation dataset
                sess.run(iterator.initializer, feed_dict={tfrecord_filenames: validate_record_filenames})
                total_val_accuracy = 0
                validation_count = 0
                total_conf_matrix = None
                for step in range(val_batches):
                    validation_batch_xs, validation_batch_ys = sess.run(next_batch)
                    # Run a validation step and capture training summaries for TensorBoard
                    # with the `merged` op.
                    validation_summary, validation_accuracy, conf_matrix = sess.run(
                        [summary_op, accuracy, confusion_matrix],
                        feed_dict={
                            X: validation_batch_xs,
                            ground_truth: validation_batch_ys,
                            is_training: False,
                            dropout_keep_prob: 1
                        })

                    validation_writer.add_summary(validation_summary, num_epoch)

                    total_val_accuracy += validation_accuracy
                    validation_count += 1
                    if total_conf_matrix is None:
                        total_conf_matrix = conf_matrix
                    else:
                        total_conf_matrix += conf_matrix

                total_val_accuracy /= validation_count

                tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
                tf.logging.info('Validation accuracy = %.1f%% (N=%d)' %
                                (total_val_accuracy * 100, PCAM_VALIDATE_DATA_SIZE))


                # Save the model checkpoint periodically.
                if (num_epoch <= FLAGS.how_many_training_epochs-1):
                    checkpoint_path = os.path.join(FLAGS.train_logdir, FLAGS.ckpt_name_to_save)
                    tf.logging.info('Saving to "%s-%d"', checkpoint_path, num_epoch)
                    saver.save(sess, checkpoint_path, global_step=num_epoch)


if __name__ == '__main__':
    tf.app.run()
