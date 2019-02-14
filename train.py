from functools import partial
import torch
from torch import optim
from torchvision import datasets
import torch.nn.functional as F
import tqdm
import numpy as np

import model
import loss
from utils import *
from dataset import SiameseNetworkDataset, data_loaders
from Config import Config
from train_args import get_args


def train(train_loader, valid_loader):
    loss_per_epoch = {"train_loss": [], "valid_loss": []}
    num_batch_train = len(train_loader)
    num_batch_valid = len(valid_loader)

    for epoch in range(con_epoch, Config.train_number_epochs):
        loss_per_batch = {"train_loss": [], "valid_loss": []}
        print("Epoch number: %d" % (epoch + 1))

        with tqdm.tqdm(total=num_batch_train) as pbar_train:
            for i, data in enumerate(train_loader, 0):  # Train for one batch
                img0, img1, label = data
                img0, img1, label = img0.to(device), img1.to(device), label.to(device)
                optimizer.zero_grad()
                output1, output2 = model(img0, img1)
                loss = error(output1, output2, label)
                loss.backward()
                optimizer.step()
                loss_per_batch["train_loss"].append(loss.item())
                pbar_train.set_description("train loss: {:.4f}".format(np.mean(loss_per_batch['train_loss'])))
                pbar_train.update(1)

        with tqdm.tqdm(total=num_batch_valid) as pbar_valid:
            for i, data in enumerate(valid_loader, 0):  # Evaluation for one batch
                img0, img1, label = data
                img0, img1, label = img0.to(device), img1.to(device), label.to(device)
                output1, output2 = model.forward(img0, img1)
                loss = error(output1, output2, label)
                loss_per_batch["valid_loss"].append(loss.item())
                pbar_valid.set_description("valid loss: {:.4f}".format(np.mean(loss_per_batch['valid_loss'])))
                pbar_valid.update(1)

        for key, value in loss_per_batch.items():  # Collect average loss for current epoch
            loss_per_epoch[key].append(np.mean(value))

        if loss_per_epoch["valid_loss"][-1] < local_param.best_loss:
            local_param.best_loss = np.mean(loss_per_epoch["valid_loss"][-1])
            local_param.best_index = epoch

        save_model(model, optimizer, args.exp_name, epoch, local_param.best_index, local_param.best_loss)

        if epoch == Config.train_number_epochs - 1:
            log_model(local_param)
            del_models(local_param.exp_name)

args = get_args()
rng = np.random.RandomState(seed=2019)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = create_model(args.model)
model = model.to(device)
metric = partial(F.pairwise_distance, p=args.metric)
error = create_loss(args.loss, metric=metric)
lr = args.lr
optimizer = optim.Adam(model.parameters(), lr=lr)
con_epoch = args.con_epoch

train_dataset = datasets.ImageFolder(root=Config.train_dir)
valid_dataset = datasets.ImageFolder(root=Config.valid_dir)
test_dataset = datasets.ImageFolder(root=Config.test_dir)

train_loader, valid_loader, _ = data_loaders(model, train_dataset, valid_dataset, test_dataset)
local_param = LocalParam(args.exp_name, args.model, args.loss, args.metric, args.lr)

if (con_epoch != 0):
    local_param.best_index, local_param.best_loss = load_model(model, optimizer, local_param.exp_name, con_epoch)

print(local_param.exp_name, "starts training...")
train(train_loader, valid_loader)
print("training finished")