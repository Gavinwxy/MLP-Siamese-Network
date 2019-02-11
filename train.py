from torch import optim
from torchvision import datasets
import tqdm
import numpy as np

import model
import loss
from utils import save_model,load_model,create_model,create_loss
from dataset import SiameseNetworkDataset, data_loaders
from Config import Config
from arg_extractor import get_args


def train(train_loader, valid_loader, **param):
    loss_per_epoch = {"train_loss": [], "valid_loss": []}
    num_batch_train = len(train_loader)
    num_batch_valid = len(valid_loader)

    for epoch in range(BestConfig.start_index, Config.train_number_epochs):
        loss_per_batch = {"train_loss": [], "valid_loss": []}
        print("Epoch number: %d" % (epoch + 1))

        with tqdm.tqdm(total=num_batch_train) as pbar_train:
            for i, data in enumerate(train_loader, 0):  # Train for one batch
                img0, img1, label = data
                # img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
                optimizer.zero_grad()
                output1, output2 = model(img0, img1)
                loss = criterion(output1, output2, label)
                loss.backward()
                optimizer.step()
                loss_per_batch["train_loss"].append(loss.item())
                pbar_train.set_description("train loss: {:.4f}".format(np.mean(loss_per_batch['train_loss'])))
                pbar_train.update(1)

        with tqdm.tqdm(total=num_batch_valid) as pbar_valid:
            for i, data in enumerate(valid_loader, 0):  # Evaluation for one batch
                img0, img1, label = data
                # img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
                output1, output2 = model.forward(img0, img1)
                loss = criterion(output1, output2, label)
                loss_per_batch["valid_loss"].append(loss.item())
                pbar_valid.set_description("valid loss: {:.4f}".format(np.mean(loss_per_batch['valid_loss'])))
                pbar_valid.update(1)

        # Question here: -1? average?
        # if loss_per_batch["valid_loss"][-1] < BestConfig.best_loss:
        if np.mean(loss_per_batch["valid_loss"]) < BestConfig.best_loss:
            BestConfig.best_loss = np.mean(loss_per_batch["valid_loss"])
            BestConfig.best_index = epoch

        for key, value in loss_per_batch.items():  # Collect average loss for current epoch
            loss_per_epoch[key].append(np.mean(value))

        save_model(model, optimizer, args.exp_name, epoch, BestConfig.best_index, BestConfig.best_loss)
    return loss_per_epoch['valid_loss'][-1]     # same issue


args = get_args()
rng = np.random.RandomState(seed=2019)

model = create_model(args.model)
criterion = create_loss(args.loss)
lr = args.lr
optimizer = optim.Adam(model.parameters(), lr=lr)
con_epoch = args.con_epoch

train_dataset = datasets.ImageFolder(root=Config.train_dir)
valid_dataset = datasets.ImageFolder(root=Config.valid_dir)
test_dataset = datasets.ImageFolder(root=Config.test_dir)

train_loader, valid_loader, _ = data_loaders(model, train_dataset, valid_dataset, test_dataset)


class BestConfig:
    best_loss = float('inf')
    best_index = 0
    start_index = 0


if (con_epoch != -1):
    BestConfig.best_index, BestConfig.best_loss = load_model(model, optimizer, args.exp_name, con_epoch)
    BestConfig.start_index = con_epoch

train(train_loader, valid_loader)