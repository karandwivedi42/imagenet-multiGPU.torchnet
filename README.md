# imagenet-multiGPU.torchnet

Inspired by imagenet-multiGPU.torch, uses torchnet dataloaders and engines.

- ClassificationLogger is a combination of meters which can be used anywhere.
- RandomDataset is useful for random sampling.

* Note

1. The script might not work without GPU, lots of lines (like `cutorch.synchronize()`) will have to be removed.
2. You will need to prepare the val folder into the flat folder style. See **imagenet-multiGPU.torch** for instructions.

#####Few Notes:

1. Different seed in each thread is necessary otherwise random sampling returns same data for all thread.
2. Not tested on full IN, only tested on smaller datasets.
3. setGPU() does not work with multiple GPUs (probably change the createDataParallelTable to setGPU() )
