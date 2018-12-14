
## To Train:

This command trains a ResNet50 with GN on ImageNet:
```
python main.py /path/to/imagenet --multiprocessing-distributed -j 8
```

Trained on 8 GPUs with a total batch size of 256, it should achieve 24.1~24.3 Top-1 accuracy.
This is about 0.1 worse than the paper, probably because this implementation uses fewer augmentations than the paper.

Training takes ~26 hours on 8 V100s. A training log is included.

The code is modified slightly from [pytorch official examples](https://github.com/pytorch/examples/tree/master/imagenet)
and uses the same data augmentations available there.
