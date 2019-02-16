# MLP-Siamese-Network
Investigate Siamese Network for Face Verification

## To Do
### CNN Backbone 
- [x] ChopraNet (Baseline)
- [x] DeepID v1 
- [ ] DeepFace 
- [ ] FaceNet 
- [ ] ResNet 
### Loss Function
- [x] Contrastive loss
- [ ] Triplet loss
### Engineering Part
- [x] GPU server configuration - Google Cloud Computing
- [x] Test accuracy - ROC-AUC
- [ ] Implement locally connected layer
- [ ] Learning rate shceduler (Decrease with epoch)
- [ ] Batch normalization
- [ ] Data augmentation (rotation, jittering,...)

## Issues
-  GPU dose not computing in training session
-  We found data loader may become the bottleneck in the training
-  Official data loader of Cifar10 is tested on our model. Significant difference on speed can be observed between using GPU and not, indicating server GPU functions well

**2/15/2019**

- Noticed SSD and multi-thread data loader may accelerate the training process. 
