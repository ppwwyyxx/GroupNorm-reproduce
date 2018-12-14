
## To Train:

```
./cifar10-resnet-gn.py
```

Trained on 2 GPUs, this code achieves 7.3% error after 400 epochs.
This is about 1.3% worse than its BatchNorm counterpart, because the Cifar10 dataset is too small and
requires more regularization.

Note that this is not an experiment in the paper.
