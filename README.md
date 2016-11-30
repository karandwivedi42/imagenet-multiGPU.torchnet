# imagenet-multiGPU.torchnet

_Inspired by imagenet-multiGPU.torch, uses torchnet dataloaders and engines.

- ClassificationLogger is a combination of meters which can be used anywhere.
- RandomDataset is useful for random sampling.

### Trained Model

 [Pretrained Alexnet using this code](https://drive.google.com/open?id=0B7ZgIaKJsQhbYmlsM0RCU21QMXc) 
 (Logs at http://imgur.com/a/QWlYr)
 (Final Train Acc: 59.33% | Val Acc: 58.59% | Val mAP: 0.62 after 55 epochs)

* Note

1. For the script to work without GPU, lots of lines (like `cutorch.synchronize()`) will have to be removed.
2. You will need to prepare the val folder into the flat folder style. See **imagenet-multiGPU.torch** for instructions.
3. setGPU() does not work with multiple GPUs (probably change the createDataParallelTable to setGPU() )
