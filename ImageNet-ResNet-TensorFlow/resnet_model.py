#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: resnet_model.py

import tensorflow as tf
import math


from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.tfutils import get_current_tower_context
from tensorpack.models import (
    Conv2D, GlobalAvgPooling, BatchNorm, FullyConnected,
    LinearWrap, layer_register)


@layer_register(log_shape=True)
def GroupNorm(x, group=32, gamma_initializer=tf.constant_initializer(1.)):
    """
    https://arxiv.org/abs/1803.08494
    """
    shape = x.get_shape().as_list()
    ndims = len(shape)
    assert ndims in [2, 4]
    chan = shape[1]

    assert chan % group == 0, chan
    group_size = chan // group

    orig_shape = tf.shape(x)
    h, w = orig_shape[2], orig_shape[3]

    x = tf.reshape(x, tf.stack([-1, group, group_size, h, w]))

    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)

    new_shape = [1, group, group_size, 1, 1]

    beta = tf.get_variable('beta', [chan], initializer=tf.constant_initializer())
    beta = tf.reshape(beta, new_shape)

    gamma = tf.get_variable('gamma', [chan], initializer=gamma_initializer)
    gamma = tf.reshape(gamma, new_shape)

    out = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5, name='output')
    return tf.reshape(out, orig_shape, name='output')


def GNReLU(x, name=None):
    x = GroupNorm('gn', x)
    return tf.nn.relu(x, name=name)


def resnet_shortcut(l, n_out, stride, activation=tf.identity):
    n_in = l.get_shape().as_list()[1]
    if n_in != n_out:   # change dimension when channel is not the same
        return Conv2D('convshortcut', l, n_out, 1, strides=stride, activation=activation)
    else:
        return l


def get_gn(zero_init=False):
    """
    Zero init gamma is good for resnet. See https://arxiv.org/abs/1706.02677.
    """
    if zero_init:
        return lambda x, name=None: GroupNorm('gn', x, gamma_initializer=tf.zeros_initializer())
    else:
        return lambda x, name=None: GroupNorm('gn', x)


def resnet_bottleneck(l, ch_out, stride, stride_first=False):
    """
    stride_first: original resnet put stride on first conv. fb.resnet.torch put stride on second conv.
    """
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 1, strides=stride if stride_first else 1, activation=GNReLU)
    if stride == 1:
        l = Conv2D('conv2', l, ch_out, 3, strides=1 if stride_first else stride, activation=GNReLU)
    else:
        l = tf.pad(l, [[0, 0], [0, 0], [1, 1], [1, 1]])
        l = Conv2D('conv2', l, ch_out, 3, strides=1 if stride_first else
                   stride, activation=GNReLU, padding='VALID')
    l = Conv2D('conv3', l, ch_out * 4, 1, activation=get_gn(zero_init=True))
    return tf.nn.relu(l +
        resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_gn(zero_init=False)))

def resnet_group(l, name, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                l = block_func(l, features, stride if i == 0 else 1)
    return l


def resnet_backbone(image, num_blocks, group_func, block_func):
    with argscope(Conv2D, use_bias=False,
                  kernel_initializer=tf.variance_scaling_initializer(
                      scale=2.0, mode='fan_out', distribution='untruncated_normal')):
        logits = (LinearWrap(image)
                  .tf.pad([[0, 0], [0, 0], [3, 3], [3, 3]])
                  .Conv2D('conv0', 64, 7, strides=2, activation=GNReLU, padding='VALID')
                  .tf.pad([[0, 0], [0, 0], [1, 1], [1, 1]])
                  .MaxPooling('pool0', shape=3, stride=2, padding='VALID')
                  .apply(group_func, 'group0', block_func, 64, num_blocks[0], 1)
                  .apply(group_func, 'group1', block_func, 128, num_blocks[1], 2)
                  .apply(group_func, 'group2', block_func, 256, num_blocks[2], 2)
                  .apply(group_func, 'group3', block_func, 512, num_blocks[3], 2)
                  .GlobalAvgPooling('gap')
                  .FullyConnected('linear', 1000,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01))())
    return logits
