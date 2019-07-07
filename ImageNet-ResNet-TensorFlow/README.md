
## To Train:

This command trains a ResNet-50 with GN on ImageNet:
```
./imagenet-resnet-gn.py --data /path/to/imagenet --depth 50
```

This script tries to follow the exact setting in the original paper.

Trained on 8 GPUs with a total batch size of 256, ResNet-50 in this script
achieves __24.0, 24.0, 24.1__ top-1 validation error in 3 independent runs,
evaluated by the median of last 5 epochs. This matches the performance in the paper.

Training a ResNet-50 takes ~27 hours on 8 V100s.

Training a ResNet-101 with this script should reach 22.4~22.5 top-1 validation error.

Training logs for ResNet-50 & ResNet-101 are included.


The code is modified slightly from [Tensorpack ResNet examples](https://github.com/tensorpack/tensorpack/tree/master/examples/ResNet).

A trained ResNet-50 model is available at [Tensorpack model zoo](http://models.tensorpack.com/FasterRCNN/ImageNet-R50-GroupNorm32-AlignPadding.npz).
It can be evaluated with:
```
./imagenet-resnet-gn.py --data /path/to/imagenet -d 50 --eval --load model.npz
```
