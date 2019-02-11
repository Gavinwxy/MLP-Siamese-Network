from torch import optim
from torchvision import datasets
import tqdm
import numpy as np

import model
import loss
from utils import save_model, load_model, create_model, create_loss
from dataset import SiameseNetworkDataset, data_loaders
from Config import Config


def evaluate(test_loader, **param):
    num_batch_test = len(test_loader)
    loss_per_batch = []

    with tqdm.tqdm(total=num_batch_test) as pbar_test:
        for i, data in enumerate(test_loader, 0):  # Test for one batch
            img0, img1, label = data
            # img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            output1, output2 = model(img0, img1)
            loss = criterion(output1, output2, label)
            loss.backward()
            loss_per_batch.append(loss.item())
            pbar_test.set_description("test loss: {:.4f}".format(np.mean(loss_per_batch)))
            pbar_test.update(1)

    return np.mean(loss_per_batch)


rng = np.random.RandomState(seed=2019)

# wait to be implement (lack of reading best model and loss from model file)
# call after all the models have trained

model = create_model(args.model)
criterion = create_loss(args.loss)
lr = args.lr
optimizer = optim.Adam(model.parameters(), lr=lr)

train_dataset = datasets.ImageFolder(root=Config.train_dir)
valid_dataset = datasets.ImageFolder(root=Config.valid_dir)
test_dataset = datasets.ImageFolder(root=Config.test_dir)

_, _, test_loader = data_loaders(model, train_dataset, valid_dataset, test_dataset)
evaluate(test_loader)
