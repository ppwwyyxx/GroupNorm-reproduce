
A collection of code in different frameworks, that __reproduces__ several experiment results in the paper:

+ [Group Normalization](https://arxiv.org/abs/1803.08494) - Best Paper Honorable Mention at ECCV 2018.

There are a number of implementations of this paper online, however it's unsafe
to trust any implementation (including your own) without reproducing results.

## Official Code

+ [Object Detection with Mask R-CNN, in Caffe2](https://github.com/facebookresearch/Detectron/tree/master/projects/GN), with models.

## Unofficial Code

+ [ImageNet Classification with ResNet-50, in TensorFlow](ImageNet-ResNet-TensorFlow/), with models. Exactly reproduce.

+ [ImageNet Classification with ResNet-50, in PyTorch](ImageNet-ResNet-PyTorch/), with models.
  0.1% different result due to fewer augmentations.

+ [ImageNet Classification with VGG-16, in TensorFlow](https://github.com/tensorpack/tensorpack/tree/master/examples/ImageNetModels#vgg16).
  Exactly reproduce.

+ [Object Detection with Mask R-CNN, in TensorFlow](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN), with models.
  Slighly better results due to different implementation of Mask R-CNN framework.
  
+ [Object Detection with Mask R-CNN, in PyTorch](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md#other-settings), with models.
  Slighly better results due to different implementation of Mask R-CNN framework.

## Not in the Paper

+ [Cifar10 Classification with ResNet, in TensorFlow](Cifar10-ResNet-TensorFlow/).


## Implementations of the GroupNorm operation:

+ PyTorch: [Implemented in C++](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Normalization.cpp),
  and available as a layer via [`torch.nn.GroupNorm`](https://pytorch.org/docs/stable/nn.html#torch.nn.GroupNorm).
+ TensorFlow: [Python implementation](ImageNet-ResNet-TensorFlow/resnet_model.py).
+ Caffe2: [Implemented in C++](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/group_norm_op.h),
	available in Python via `brew.spatial_gn`.
