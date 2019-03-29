from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nets import resnet_v2

import tensorflow as tf

slim = tf.contrib.slim


def hcd_model(inputs,
              num_classes,
              is_training=True,
              keep_prob=0.5,
              scope='HCD_model'):
    '''
    :param inputs: N x H x W x C tensor
    :return:
    '''

    # with tf.variable_scope(scope, 'HCD_model', [inputs]):
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        net, end_points = \
            resnet_v2.resnet_v2_101(inputs,
                                    num_classes=num_classes,
                                    is_training=is_training,
                                    scope='resnet_v2_101')

    # out1 = GlobalMaxPooling2D()(x)
    net1 = tf.reduce_max(net, [1, 2], name='GlobalMaxPooling2D')
    # out2 = GlobalAveragePooling2D()(x)
    net2 = tf.reduce_mean(net, [1, 2], name='GlobalAveragePooling2D')
    # out3 = Flatten()(x)
    net3 = slim.flatten(net)
    # out = Concatenate(axis=-1)([out1, out2, out3])
    net = tf.concat([net1, net2, net3], axis=-1)
    # out = Dropout(0.5)(out)
    net = slim.dropout(net, keep_prob=keep_prob, is_training=is_training)
    # out = Dense(1, activation="sigmoid", name="3_")(out)
    logits = slim.fully_connected(net, num_classes, activation_fn=None, scope='logits')

    return logits, end_points
