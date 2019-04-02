from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets import resnet_v2

slim = tf.contrib.slim


batch_norm_params = {
  'decay': 0.997,    # batch_norm_decay
  'epsilon': 1e-5,   # batch_norm_epsilon
  'scale': True,     # batch_norm_scale
  'updates_collections': tf.GraphKeys.UPDATE_OPS,    # batch_norm_updates_collections
  'is_training': True,  # is_training
  'fused': None,  # Use fused batch norm if possible.
}


def hcd_model(inputs,
              num_classes,
              is_training=True,
              keep_prob=0.8,
              attention_module=None,
              scope='HCD_model'):
    '''
    :param inputs: N x H x W x C tensor
    :return:
    '''

    # with tf.variable_scope(scope, 'HCD_model', [inputs]):
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        net, end_points = \
            resnet_v2.resnet_v2_50(inputs,
                                    num_classes=num_classes,
                                    is_training=is_training,
                                    attention_module=attention_module,
                                    scope='resnet_v2_50')

    # out1 = GlobalMaxPooling2D()(x)
    net1 = tf.reduce_max(net, axis=[1, 2], keep_dims=True, name='GlobalMaxPooling2D')
    # out2 = GlobalAveragePooling2D()(x)
    net2 = tf.reduce_mean(net, axis=[1, 2], keep_dims=True, name='GlobalAveragePooling2D')
    # out3 = Flatten()(x)
    # net3 = slim.flatten(net)
    # out = Concatenate(axis=-1)([out1, out2, out3])
    net = tf.concat([net1, net2], axis=-1)
    net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')

    # out = Dropout(0.5)(out)
    net = slim.dropout(net, keep_prob=keep_prob, is_training=is_training)
    # out = Dense(1, activation="sigmoid", name="3_")(out)
    net = slim.fully_connected(net,
                               768,
                               normalizer_fn=slim.batch_norm,
                               normalizer_params=batch_norm_params,
                               scope='fc1')
    net = slim.dropout(net, keep_prob=keep_prob, is_training=is_training)
    net = slim.fully_connected(net,
                               256,
                               normalizer_fn=slim.batch_norm,
                               normalizer_params=batch_norm_params,
                               scope='fc2')
    net = slim.dropout(net, keep_prob=keep_prob, is_training=is_training)
    logits = slim.fully_connected(net,
                                  num_classes,
                                  activation_fn=None,
                                  scope='logits')

    return logits, end_points
