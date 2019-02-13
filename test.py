from functools import partial
import torch
from torch import optim
from torchvision import datasets
import torch.nn.functional as F
import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score

import model
import loss
from utils import *
from dataset import SiameseNetworkDataset, data_loaders
from Config import Config
from test_args import get_args


def evaluate(test_loader, loop_times):
    roc_auc_scores = []
    with tqdm.tqdm(total=loop_times) as pbar_test:
        for _ in range(loop_times):
            y_true = []
            y_score = []
            for i, data in enumerate(test_loader, 0):
                img0, img1, label = data
                img0, img1, label = img0.to(device), img1.to(device), label.to(device)
                output1, output2 = model(img0, img1)
                distance = metric(output1, output2)
                y_true.append(label.item())
                y_score.append(distance.item())

            roc_auc_scores.append(roc_auc_score(y_true, y_score))
            pbar_test.set_description("ROC_AUC score: {:.4f}".format(np.mean(roc_auc_scores)))
            pbar_test.update(1)

    return np.mean(roc_auc_scores)


args = get_args()
rng = np.random.RandomState(seed=2019)

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

local_param = get_local_param(args.exp_name)

model = create_model(local_param.model)
model = model.to(device)
metric = partial(F.pairwise_distance, p=local_param.metric)
error = create_loss(local_param.loss, metric=metric)
lr = local_param.lr
optimizer = optim.Adam(model.parameters(), lr=lr)

train_dataset = datasets.ImageFolder(root=Config.train_dir)
valid_dataset = datasets.ImageFolder(root=Config.valid_dir)
test_dataset = datasets.ImageFolder(root=Config.test_dir)

_, _, test_loader = data_loaders(model, train_dataset, valid_dataset, test_dataset)
roc_auc_score = evaluate(test_loader, Config.evaluation_times)
