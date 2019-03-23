# MLP-Siamese-Network
Investigate Siamese Network for Face Verification

## CNN Backbone

### ChopraNet

<img src="fig/ChopraNet.JPG" alt="drawing" style="width:600px;"/>

### DeepID 

<img src="fig/DeepID.JPG" alt="drawing" style="width:1000px;"/>

### DeepFace

<img src="fig/DeepFace.JPG" alt="drawing" style="width:1000px;"/>

### ResNet-50

<img src="fig/resnet50.JPG" alt="drawing" style="width:600px;"/>

## Loss Functions

### Contrastive Loss


$$
    L_{Contrastive} = (1 - Y)L_G(E_W(X_1,X_2)^i) + YL_I(E_W(X_1,X_2)^i)
$$


### Triplet Loss


$$
L_{Triplet}= max(||f(x^a_i) - f(x^p_i)||^2_2-||f(x^a_i) - f(x^p_i)||^2_2 + \alpha, 0)
$$


### Logistic Loss


$$
d(f_1, f_2) = \sum_i \alpha_i |f_1[i] - f_2[i]|
$$


### ArcFace Loss


$$
L_{ArcFace} = -\frac{1}{N}\sum_{i=1}^{N}log\frac{e^{s\cos(\theta_{y_i}+m)}}{e^{s\cos(\theta_{y_i}+m)} + \sum_{j=1,j\neq y_i}^{n}e^{s\cos\theta_j}}
\\
where\ \cos\theta_{j_i}= W^T_{j}x_i/(||W_j||\ ||x_i||)
$$

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

