# MLP-Siamese-Network
Investigate Siamese Network for Face Verification

## CNN Backbone

### ChopraNet

<img src="fig/ChopraNet.JPG" alt="drawing" style="width:400px;"/>

### DeepID 

<img src="fig/DeepID.JPG" alt="drawing" style="width:1000px;"/>

### DeepFace

<img src="fig/DeepFace.JPG" alt="drawing" style="width:1000px;"/>

### ResNet-50

<img src="fig/resnet50.JPG" alt="drawing" style="width:400px;"/>



## Result

<img src="fig/visual.JPG" alt="drawing" style="width:600px;"/>

<img src="fig/result.JPG" alt="drawing" style="width:1200px;"/>

## To Do
**CNN Backbone** 

- [x] ChopraNet (Baseline)
- [x] DeepID v1 
- [x] DeepFace 
- [x] ResNet50 https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
- [x] Xception https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/xception.py

**Loss Function**

- [x] Contrastive Loss
- [x] Triplet Loss
- [x] Logistic Loss
- [x] CosFace Loss 
- [x] ArcFace Loss 

**Engineering** 

- [x] GPU server configuration - Google Cloud Computing
- [x] Evaluation Metric - ROC-AUC
- [x] Implement locally connected layer
- [x] Learning rate shceduler (Cosine Annealing)
- [x] Batch normalization
- [x] Data augmentation (Random Horizontal Flip)

