# -*- coding: utf-8 -*-
# File: imagenet_utils.py


import cv2
import os
import numpy as np
import tqdm
import multiprocessing
import tensorflow as tf
from abc import abstractmethod

from tensorpack import *
from tensorpack.dataflow.imgaug import AugmentorList
from tensorpack import ModelDesc
from tensorpack.input_source import QueueInput, StagingInput
from tensorpack.dataflow import (
    imgaug, dataset, AugmentImageComponent, MultiProcessRunnerZMQ,
    BatchData, MultiThreadMapData)
from tensorpack.predict import PredictConfig, FeedfreePredictor
from tensorpack.utils.stats import RatioCounter
from tensorpack.models import regularize_cost
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils import logger


"""
====== DataFlow =======
"""


def fbresnet_augmentor(isTrain):
    """
    Augmentor used in fb.resnet.torch, for BGR images in range [0,255].
    """
    if isTrain:
        augmentors = [
            imgaug.GoogleNetRandomCropAndResize(),
            imgaug.Flip(horiz=True),
            imgaug.ToFloat32(),
            imgaug.RandomOrderAug(
                [imgaug.BrightnessScale((0.6, 1.4), clip=False),
                 imgaug.Contrast((0.6, 1.4), rgb=False, clip=False),
                 imgaug.Saturation(0.4, rgb=False),
                 # rgb-bgr conversion for the constants copied from fb.resnet.torch
                 imgaug.Lighting(0.1,
                                 eigval=np.asarray(
                                     [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                                 eigvec=np.array(
                                     [[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]],
                                     dtype='float32')[::-1, ::-1]
                                 )]),
        ]
    else:
        augmentors = [
            imgaug.ResizeShortestEdge(256, cv2.INTER_LINEAR),
            imgaug.CenterCrop((224, 224)),
            imgaug.ToFloat32(),
        ]
    return augmentors


def get_imagenet_dataflow(
        datadir, name, batch_size,
        augmentors=None, parallel=None):
    """
    Args:
        augmentors (list[imgaug.Augmentor]): Defaults to `fbresnet_augmentor(isTrain)`

    Returns: A DataFlow which produces BGR images and labels.
    """
    assert name in ['train', 'val', 'test']
    isTrain = name == 'train'
    assert datadir is not None
    if augmentors is None:
        augmentors = fbresnet_augmentor(isTrain)
    assert isinstance(augmentors, list)
    augmentors = AugmentorList(augmentors)
    if parallel is None:
        parallel = min(40, multiprocessing.cpu_count() // 2)  # assuming hyperthreading

    def mapf(dp):
        fname, label = dp
        img = cv2.imread(fname)
        img = augmentors.augment(img)
        return img, label

    if isTrain:
        ds = dataset.ILSVRC12Files(datadir, name, shuffle=True)
        ds = MultiProcessMapDataZMQ(ds, parallel, mapf, buffer_size=2000)
        ds = BatchData(ds, batch_size, remainder=False)
    else:
        ds = dataset.ILSVRC12Files(datadir, name, shuffle=False)
        ds = MultiThreadMapData(ds, parallel, mapf, buffer_size=2000, strict=True)
        ds = BatchData(ds, batch_size, remainder=True)
        ds = MultiProcessRunnerZMQ(ds, 1)
    return ds


"""
====== Model & Evaluation =======
"""


def eval_on_ILSVRC12(model, sessinit, dataflow):
    pred_config = PredictConfig(
        model=model,
        session_init=sessinit,
        input_names=['input', 'label'],
        output_names=['wrong-top1', 'wrong-top5']
    )
    acc1, acc5 = RatioCounter(), RatioCounter()

    # This does not have a visible improvement over naive predictor,
    # but will have an improvement if image_dtype is set to float32.
    pred = FeedfreePredictor(pred_config, StagingInput(QueueInput(dataflow), device='/gpu:0'))
    for _ in tqdm.trange(dataflow.size()):
        top1, top5 = pred()
        batch_size = top1.shape[0]
        acc1.feed(top1.sum(), batch_size)
        acc5.feed(top5.sum(), batch_size)

    print("Top1 Error: {}".format(acc1.ratio))
    print("Top5 Error: {}".format(acc5.ratio))


class ImageNetModel(ModelDesc):
    image_shape = 224

    """
    uint8 instead of float32 is used as input type to reduce copy overhead.
    It might hurt the performance a liiiitle bit.
    The pretrained models were trained with float32.
    """
    image_dtype = tf.uint8

    """
    Either 'NCHW' or 'NHWC'
    """
    data_format = 'NCHW'

    """
    Whether the image is BGR or RGB. If using DataFlow, then it should be BGR.
    """
    image_bgr = True

    weight_decay = 1e-4

    """
    To apply on normalization parameters, use '.*/W|.*/gamma|.*/beta'
    """
    weight_decay_pattern = '.*/W'

    """
    Scale the loss, for whatever reasons (e.g., gradient averaging, fp16 training, etc)
    """
    loss_scale = 1.

    """
    Label smoothing (See tf.losses.softmax_cross_entropy)
    """
    label_smoothing = 0.

    def inputs(self):
        return [tf.TensorSpec([None, self.image_shape, self.image_shape, 3], self.image_dtype, 'input'),
                tf.TensorSpec([None], tf.int32, 'label')]

    def build_graph(self, image, label):
        image = self.image_preprocess(image)
        assert self.data_format in ['NCHW', 'NHWC']
        if self.data_format == 'NCHW':
            image = tf.transpose(image, [0, 3, 1, 2])

        logits = self.get_logits(image)
        loss = ImageNetModel.compute_loss_and_error(
            logits, label, label_smoothing=self.label_smoothing)

        if self.weight_decay > 0:
            wd_loss = regularize_cost(self.weight_decay_pattern,
                                      tf.contrib.layers.l2_regularizer(self.weight_decay),
                                      name='l2_regularize_loss')
            add_moving_summary(loss, wd_loss)
            total_cost = tf.add_n([loss, wd_loss], name='cost')
        else:
            total_cost = tf.identity(loss, name='cost')
            add_moving_summary(total_cost)

        if self.loss_scale != 1.:
            logger.info("Scaling the total loss by {} ...".format(self.loss_scale))
            return total_cost * self.loss_scale
        else:
            return total_cost

    @abstractmethod
    def get_logits(self, image):
        """
        Args:
            image: 4D tensor of ``self.input_shape`` in ``self.data_format``

        Returns:
            Nx#class logits
        """

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
        tf.summary.scalar('learning_rate-summary', lr)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)

    def image_preprocess(self, image):
        with tf.name_scope('image_preprocess'):
            if image.dtype.base_dtype != tf.float32:
                image = tf.cast(image, tf.float32)
            mean = [0.485, 0.456, 0.406]    # rgb
            std = [0.229, 0.224, 0.225]
            if self.image_bgr:
                mean = mean[::-1]
                std = std[::-1]
            image_mean = tf.constant(mean, dtype=tf.float32) * 255.
            image_std = tf.constant(std, dtype=tf.float32) * 255.
            image = (image - image_mean) / image_std
            return image

    @staticmethod
    def compute_loss_and_error(logits, label, label_smoothing=0.):
        if label_smoothing == 0.:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        else:
            nclass = logits.shape[-1]
            loss = tf.losses.softmax_cross_entropy(
                tf.one_hot(label, nclass),
                logits, label_smoothing=label_smoothing)
        loss = tf.reduce_mean(loss, name='xentropy-loss')

        def prediction_incorrect(logits, label, topk=1, name='incorrect_vector'):
            with tf.name_scope('prediction_incorrect'):
                x = tf.logical_not(tf.nn.in_top_k(logits, label, topk))
            return tf.cast(x, tf.float32, name=name)

        wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))

        wrong = prediction_incorrect(logits, label, 5, name='wrong-top5')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5'))
        return loss
