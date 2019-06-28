#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import tensorflow as tf

from tensorpack import logger, QueueInput
from tensorpack.models import *
from tensorpack.callbacks import *
from tensorpack.train import (
    TrainConfig, SyncMultiGPUTrainerReplicated, launch_train_with_config)
from tensorpack.dataflow import FakeData
from tensorpack.tfutils import argscope, get_model_loader, varreplace
from tensorpack.utils.gpu import get_num_gpu

from imagenet_utils import (
    get_imagenet_dataflow, ImageNetModel,
    eval_on_ILSVRC12)
from resnet_model import (
    resnet_group, resnet_bottleneck, resnet_backbone)


class Model(ImageNetModel):

    weight_decay_pattern = '.*/W|.*/gamma|.*/beta'

    image_dtype = tf.float32

    depth = 50

    use_WS = False
    """
    Whether to use Weight Standardization (https://arxiv.org/abs/1903.10520)
    """

    def get_logits(self, image):

        def weight_standardization(v):
            if not self.use_WS:
                return v
            if (not v.name.endswith('/W:0')) or v.shape.ndims != 4:
                return v
            mean, var = tf.nn.moments(v, [0, 1, 2], keep_dims=True)
            v = (v - mean) / (tf.sqrt(var)+ 1e-5)
            return v

        num_blocks = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3]}[self.depth]
        block_func = resnet_bottleneck
        with argscope([Conv2D, MaxPooling, GlobalAvgPooling], data_format=self.data_format), \
                varreplace.remap_variables(weight_standardization):
            return resnet_backbone(
                image, num_blocks, resnet_group, block_func)


def get_config(model, fake=False):
    nr_tower = max(get_num_gpu(), 1)
    assert args.batch % nr_tower == 0
    batch = args.batch // nr_tower

    if fake:
        logger.info("For benchmark, batch size is fixed to 64 per tower.")
        dataset_train = FakeData(
            [[64, 224, 224, 3], [64]], 1000, random=False, dtype='uint8')
        callbacks = []
        steps_per_epoch = 100
    else:
        logger.info("Running on {} towers. Batch size per tower: {}".format(nr_tower, batch))

        dataset_train = get_imagenet_dataflow(args.data, 'train', batch)
        dataset_val = get_imagenet_dataflow(args.data, 'val', min(64, batch))
        steps_per_epoch = 1281167 // args.batch

        BASE_LR = 0.1 * args.batch / 256.0
        logger.info("BASELR: {}".format(BASE_LR))
        callbacks = [
            ModelSaver(),
            EstimatedTimeLeft(),
            GPUUtilizationTracker(),
            ScheduledHyperParamSetter(
                'learning_rate', [(0, BASE_LR), (30, BASE_LR * 1e-1), (60, BASE_LR * 1e-2),
                                  (90, BASE_LR * 1e-3)]),
        ]
        if BASE_LR > 0.1:
            callbacks.append(
                ScheduledHyperParamSetter(
                    'learning_rate', [(0, 0.1), (5 * steps_per_epoch, BASE_LR)],
                    interp='linear', step_based=True))

        infs = [ClassificationError('wrong-top1', 'val-error-top1'),
                ClassificationError('wrong-top5', 'val-error-top5')]
        if nr_tower == 1:
            # single-GPU inference with queue prefetch
            callbacks.append(InferenceRunner(QueueInput(dataset_val), infs))
        else:
            # multi-GPU inference (with mandatory queue prefetch)
            callbacks.append(DataParallelInferenceRunner(
                dataset_val, infs, list(range(nr_tower))))

    return TrainConfig(
        model=model,
        dataflow=dataset_train,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        max_epoch=100,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='ILSVRC dataset dir')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--fake', help='use fakedata to test or benchmark this model', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--batch', default=256, type=int, help='total batch size.')
    parser.add_argument('-d', '--depth', type=int, default=50, choices=[50, 101])
    parser.add_argument('--logdir', default='train_log/ResNet-GN')
    parser.add_argument('--WS', action='store_true', help='Use Weight Standardization')
    args = parser.parse_args()

    model = Model()
    model.depth = args.depth
    model.use_WS = args.WS
    if args.eval:
        batch = 128    # something that can run on one gpu
        ds = get_imagenet_dataflow(args.data, 'val', batch)
        eval_on_ILSVRC12(model, get_model_loader(args.load), ds)
    else:
        if args.fake:
            logger.set_logger_dir(os.path.join('train_log', 'tmp'), 'd')
        else:
            logger.set_logger_dir(args.logdir, 'd')

        try:
            from tensorpack.tfutils import collect_env_info
            logger.info("\n" + collect_env_info())
        except Exception:
            pass
        config = get_config(model, fake=args.fake)
        if args.load:
            config.session_init = get_model_loader(args.load)
        trainer = SyncMultiGPUTrainerReplicated(max(get_num_gpu(), 1))
        launch_train_with_config(config, trainer)
