
## To Train:

This command trains a ResNet50 with GN on ImageNet:
```
./imagenet-resnet-gn.py --data /path/to/imagenet
```

This script follows the exact setting in the original paper.
Trained on 8 GPUs with a total batch size of 256, it achieves __24.0, 24.0, 24.1__ top-1 validation accuracy in 3
independent runs, evaluated by the median of last 5 epochs. This matches the performance in the paper.

Training takes ~27 hours on 8 V100s. A training log is included.

The code is modified slightly from [Tensorpack ResNet examples](https://github.com/tensorpack/tensorpack/tree/master/examples/ResNet).

A trained model is available at [Tensorpack model zoo](http://models.tensorpack.com/FasterRCNN/ImageNet-R50-GroupNorm32-AlignPadding.npz).
