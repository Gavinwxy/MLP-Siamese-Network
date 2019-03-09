# MLP-Siamese-Network
Investigate Siamese Network for Face Verification

## To Do
### CNN Backbone 
- [x] ChopraNet (Baseline)
- [x] DeepID v1 
- [x] DeepFace 
- [ ] FaceNet 
- [x] ResNet50 https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
- [x] Xception https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/xception.py
### Loss Function
- [x] Contrastive loss
- [x] Triplet loss
### Engineering Part
- [x] GPU server configuration - Google Cloud Computing
- [x] Test accuracy - ROC-AUC
- [x] Implement locally connected layer
- [ ] Learning rate shceduler (Decrease with epoch)
- [x] Batch normalization
- [ ] Data augmentation (rotation, jittering,...)

## Issues
-  GPU dose not computing in training session
-  We found data loader may become the bottleneck in the training
-  Official data loader of Cifar10 is tested on our model. Significant difference on speed can be observed between using GPU and not, indicating server GPU functions well

**2/15/2019**

- Noticed SSD and multi-thread data loader may accelerate the training process. 
