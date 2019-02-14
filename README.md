# MLP-Siamese-Network
Investigate Siamese Network for Face Verification

## To Do

- [ ] Implement locally connected layer
- [x] GPU server configuration
- [ ] Find appropriate optimizer, learning rate. (maybe with weight decay?)
- [x] Test accuracy

## Issues

1. GPU dose not computing in training session
2. We found data loader may become the bottleneck in the training
3. Official data loader of Cifar10 is tested by our model and loss function. Significant difference can be observed between using GPU and not, indicating GPU in server functions well