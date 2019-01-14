from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import os
import cv2
import numpy as np

from slim.nets import resnet_v2

import data
from utils import train_utils

slim = tf.contrib.slim


flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_string('train_logdir', './models',
                    'Where the checkpoint and logs are stored.')
flags.DEFINE_integer('log_steps', 10,
                     'Display logging information at every log_steps.')
flags.DEFINE_integer('save_interval_secs', 1200,
                     'How often, in seconds, we save the model to disk.')
flags.DEFINE_boolean('save_summaries_images', False,
                     'Save sample inputs, labels, and semantic predictions as '
                     'images to summary.')
flags.DEFINE_string('summaries_dir', './models/train_logs',
                     'Where to save summary logs for TensorBoard.')

flags.DEFINE_enum('learning_policy', 'step', ['step'],
                  'Learning rate policy for training.')
flags.DEFINE_float('base_learning_rate', .0001,
                   'The base learning rate for model training.')
flags.DEFINE_float('learning_rate_decay_factor', 0.1,
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
flags.DEFINE_string('tf_initial_checkpoint',
                    # './pre-trained/resnet_v2_101_2017_04_14/resnet_v2_101.ckpt',
                    None,
                    'The initial checkpoint in tensorflow format.')

# Dataset settings.
flags.DEFINE_string('dataset_dir',
                    '/home/ace19/dl_data/histopathologic_cancer_detection',
                    'Where the dataset reside.')

flags.DEFINE_string('labels_path',
                    '/home/ace19/dl_data/histopathologic_cancer_detection/train_labels.csv',
                    'Where the dataset reside.')

flags.DEFINE_integer('how_many_training_epochs', 100,
                     'How many training loops to run')

flags.DEFINE_integer('batch_size', 64, 'batch size')
flags.DEFINE_integer('height', 224, 'height')
flags.DEFINE_integer('width', 224, 'width')
flags.DEFINE_string('labels', '0,1', 'Labels to use')
# flags.DEFINE_integer('validation_percentage', 5,
#                      'What percentage of wavs to use as a validation set.')
flags.DEFINE_string('ckpt_name_to_save', 'resnet_v2.ckpt',
                    'name to save checkpoint file')


# temporary constant
PCAM_TRAINING_DATA_SIZE = 220025


def get_init_fn():
    """Returns a function run by the chief worker to warm-start the training."""
    checkpoint_exclude_scopes = ["InceptionV1/Logits", "InceptionV1/AuxLogits"]

    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    variables_to_restore = []
    for var in slim.get_model_variables():
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                break
        else:
            variables_to_restore.append(var)

    return slim.assign_from_checkpoint_fn(FLAGS.tf_initial_checkpoint,
                                          variables_to_restore)


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
        learning_rate = tf.placeholder(tf.float32, [])

        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, end_points = \
                resnet_v2.resnet_v2_101(X, num_classes=num_classes)

        # Print name and shape of each tensor.
        tf.logging.info("Layers")
        for k, v in end_points.items():
            print('name = {}, shape = {}'.format(v.name, v.get_shape()))

        # Print name and shape of parameter nodes  (values not yet initialized)
        tf.logging.info("\n")
        tf.logging.info("Parameters")
        for v in slim.get_model_variables():
            print('name = {}, shape = {}'.format(v.name, v.get_shape()))

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

        # Add summaries for model variables.
        for model_var in slim.get_model_variables():
            summaries.add(tf.summary.histogram(model_var.op.name, model_var))

        # Add summaries for losses.
        for loss in tf.get_collection(tf.GraphKeys.LOSSES):
            # for loss in tf.get_collection(tf.GraphKeys.LOSSES):
            summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

        # learning_rate = train_utils.get_model_learning_rate(
        #     FLAGS.learning_policy, FLAGS.base_learning_rate,
        #     FLAGS.learning_rate_decay_step, FLAGS.learning_rate_decay_factor,
        #     None, FLAGS.learning_power,
        #     FLAGS.slow_start_step, FLAGS.slow_start_learning_rate)
        optimizer = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)
        summaries.add(tf.summary.scalar('learning_rate', learning_rate))

        # for variable in slim.get_model_variables():
        #     summaries.add(tf.summary.histogram(variable.op.name, variable))

        total_loss, grads_and_vars = train_utils.optimize(optimizer)
        # total_loss, grads_and_vars = train_utils.optimize(optimizer)
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

        ###############
        # Prepare data
        ###############
        filenames = tf.placeholder(tf.string, shape=[])
        tr_dataset = data.Dataset(filenames, FLAGS.batch_size, FLAGS.how_many_training_epochs,
                                  FLAGS.height, FLAGS.width)
        iterator = tr_dataset.dataset.make_initializable_iterator()
        next_batch = iterator.get_next()

        sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        with tf.Session(config = sess_config) as sess:
            sess.run(tf.global_variables_initializer())

            # Create a saver object which will save all the variables
            saver = tf.train.Saver()
            if FLAGS.tf_initial_checkpoint:
                saver.restore(sess, FLAGS.tf_initial_checkpoint)
            # # Restore only the layers up to fc7 (included)
            # # Calling function `init_fn(sess)` will load all the pretrained weights.
            # variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['vgg_16/fc8'])
            # init_fn = tf.contrib.framework.assign_from_checkpoint_fn(FLAGS.tf_initial_checkpoint, variables_to_restore)
            # init_fn(sess)  # load the pretrained weights
            # List of trainable variables of the layers we want to train
            #
            # """
            # We want to train only the weights and biases of the two
            # fully connected layers.
            # """
            # vars_to_optimize = [v for v in tf.trainable_variables() \
            #                     if v.name.startswith('my_vgg16/wd') \
            #                     or v.name.startswith('my_vgg16/bd')]
            # print '\nvariables to optimize'
            # for v in vars_to_optimize:
            #     print
            #     v.name, v.get_shape().as_list()
            #
            # learning_rate = 0.001 / 10.
            # loss = tf.reduce_mean(
            #     tf.nn.sparse_softmax_cross_entropy_with_logits(vgg.logits, tf.to_int64(ys)))
            # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            # """
            # minimize with list of variables to update
            # """
            # train_op = optimizer.minimize(loss, var_list=vars_to_optimize)

            start_epoch = 0
            # Get the number of training/validation steps per epoch
            tr_batches = int(PCAM_TRAINING_DATA_SIZE / FLAGS.batch_size)
            if PCAM_TRAINING_DATA_SIZE % FLAGS.batch_size > 0:
                tr_batches += 1
            # v_batches = int(dataset.data_size() / FLAGS.batch_size)
            # if val_data.data_size() % FLAGS.batch_size > 0:
            #     v_batches += 1

            # The filenames argument to the TFRecordDataset initializer can either be a string,
            # a list of strings, or a tf.Tensor of strings.
            training_filenames = os.path.join(FLAGS.dataset_dir, 'train.record')
            ############################
            # Training loop.
            ############################
            for training_epoch in range(start_epoch, FLAGS.how_many_training_epochs):
                print("------------------------------------")
                print(" Epoch {} ".format(training_epoch))
                print("------------------------------------")

                sess.run(iterator.initializer, feed_dict={filenames: training_filenames})
                for step in range(tr_batches):
                    # Pull the image batch we'll use for training.
                    # train_batch_xs, train_batch_ys = dataset.next_batch(FLAGS.batch_size)
                    train_batch_xs, train_batch_ys = sess.run(next_batch)

                    # # Verify image
                    # n_batch = train_batch_xs.shape[0]
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
                                 feed_dict={X: train_batch_xs,
                                            ground_truth: train_batch_ys,
                                            learning_rate:FLAGS.base_learning_rate,
                                            is_training: True,
                                            dropout_keep_prob: 0.5})

                    train_writer.add_summary(train_summary, training_epoch)
                    tf.logging.info('Epoch #%d, Step #%d, rate %.10f, accuracy %.1f%%, loss %f' %
                                    (training_epoch, step, lr, train_accuracy * 100, train_loss))

                ###################################################
                # TODO: Validate the model on the validation set
                ###################################################
                # sess.run(iterator.initializer, feed_dict={filenames: validation_filenames})
                # for step in range(val_batches):


                # Save the model checkpoint periodically.
                if (training_epoch <= FLAGS.how_many_training_epochs-1):
                    checkpoint_path = os.path.join(FLAGS.train_logdir, FLAGS.ckpt_name_to_save)
                    tf.logging.info('Saving to "%s-%d"', checkpoint_path, training_epoch)
                    saver.save(sess, checkpoint_path, global_step=training_epoch)



if __name__ == '__main__':
    tf.app.run()
