import os
import model
import loss
import torch
from torch import optim
from torchvision import transforms, datasets
from torch.autograd import Variable
from dataset import SiameseNetworkDataset
from Config import Config
import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from functools import partial


def train(train_loader, valid_loader, search_times, **param):
    net = param['model']()
    if torch.cuda.is_available():
        net = net.cuda()
    criterion = param['loss_func'](metric=param['metric'])
    optimizer = optim.Adam(net.parameters(), lr=param['lr'])

    loss_per_epoch = {"train_loss": [], "valid_loss": []}
    num_batch_train = len(train_loader)
    num_batch_valid = len(valid_loader)

    best_epoch = 1
    for epoch in range(Config.train_number_epochs):
        loss_per_batch = {"train_loss": [], "valid_loss": []}
        print("Epoch number: %d" % (epoch+1))

        with tqdm.tqdm(total=num_batch_train) as pbar_train:
            for i, data in enumerate(train_loader, 0): # Train for one batch
                img0, img1, label = data
                if torch.cuda.is_available():
                    img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
                optimizer.zero_grad()
                output1, output2 = net(img0, img1)
                loss = criterion(output1, output2, label)
                loss.backward()
                optimizer.step()
                loss_per_batch["train_loss"].append(loss.item())
                pbar_train.set_description("train loss: {:.4f}".format(np.mean(loss_per_batch['train_loss'])))
                pbar_train.update(1)

        with tqdm.tqdm(total=num_batch_valid) as pbar_valid:
            for i, data in enumerate(valid_loader, 0): # Evaluation for one batch
                img0, img1, label = data
                if torch.cuda.is_available():
                    img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
                output1, output2 = net.forward(img0, img1)
                loss = criterion(output1, output2, label)
                loss_per_batch["valid_loss"].append(loss.item())
                pbar_valid.set_description("valid loss: {:.4f}".format(np.mean(loss_per_batch['valid_loss'])))
                pbar_valid.update(1)

        for key, value in loss_per_batch.items(): # Collect average loss for current epoch
            loss_per_epoch[key].append(np.mean(value))

        if loss_per_epoch['valid_loss'][-1] == np.min(loss_per_epoch['valid_loss']):
                best_epoch = epoch + 1                       
                torch.save(net.state_dict(), f=os.path.join(Config.saved_models_dir, 'model' + str(search_times) + '.pth')) # Model saving, Only save the parameters (Recommended)

    return np.min(loss_per_epoch['valid_loss']), best_epoch

def evaluate(test_loader, loop_times, **param):
    net = param['best_net']
    metric = param['metric']

    roc_auc_scores = []
    with tqdm.tqdm(total=loop_times) as pbar_test:
        for _ in range(loop_times):
            y_true = []
            y_score = []
            for i, data in enumerate(test_loader, 0):
                img0, img1, label = data
                output1, output2 = net(img0, img1)
                distance = metric(output1, output2)

                y_true.append(label.item())
                y_score.append(distance.item())
        
            roc_auc_scores.append(roc_auc_score(y_true, y_score))
            pbar_test.set_description("ROC_AUC score: {:.4f}".format(np.mean(roc_auc_scores)))
            pbar_test.update(1)

    return np.mean(roc_auc_scores)

def data_loaders(model, train_dataset, valid_dataset, test_dataset):
    data_transform = transforms.Compose([
        transforms.Resize(model.input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    siamese_train_dataset = SiameseNetworkDataset(imageFolderDataset=train_dataset,
                                                  transform=data_transform,
                                                  should_invert=False)

    siamese_valid_dataset = SiameseNetworkDataset(imageFolderDataset=valid_dataset,
                                                  transform=data_transform,
                                                  should_invert=False)

    siamese_test_dataset = SiameseNetworkDataset(imageFolderDataset=test_dataset,
                                                 transform=data_transform,
                                                 should_invert=False)

    train_loader = torch.utils.data.DataLoader(siamese_train_dataset, batch_size=Config.train_batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(siamese_valid_dataset, batch_size=Config.valid_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(siamese_test_dataset, batch_size=1, shuffle=True)

    return train_loader, valid_loader, test_loader


grid_search = {
    "model": [model.ChopraNet],
    "loss_func": [loss.ContrastiveLoss],
    "metric": [partial(F.pairwise_distance, p=2)],
    "lr": [1e-07, 5e-06, 0.0001, 0.005, 0.1]
}

train_dataset = datasets.ImageFolder(root=Config.train_dir)
valid_dataset = datasets.ImageFolder(root=Config.valid_dir)
test_dataset = datasets.ImageFolder(root=Config.test_dir)

search_times = 1
# Dictionary that stores the best configuration:
best_config = {key:None for key in grid_search.keys()}
best_config['search_best'] = 0
best_config['best_valid_loss'] = np.inf
best_config['best_epoch'] = 1

for model in grid_search['model']:
    for loss_func in grid_search['loss_func']:
        for metric in grid_search['metric']:
            for lr in grid_search['lr']:
                train_loader, valid_loader, _ = data_loaders(model, train_dataset, valid_dataset, test_dataset)
                best_valid_loss, best_epoch = train(train_loader, valid_loader, search_times, model=model, loss_func=loss_func, metric=metric, lr=lr)
                
                if best_valid_loss < best_config['best_valid_loss']:
                    best_config['model'] = model
                    best_config['loss_func'] = loss_func
                    best_config['metric'] = metric
                    best_config['lr'] = lr
                    best_config['search_best'] = search_times
                    best_config['best_valid_loss'] = best_valid_loss
                    best_config['best_epoch'] = best_epoch

                    np.save('best_config.npy', best_config)
                
                print("\nGrid search {} is completed.\n".format(search_times))
                search_times += 1

print("The best model is model {} with best valid loss {:.4f}".format(best_config['search_best'], best_config['best_valid_loss']))

best_net = best_config['model']()
if torch.cuda.is_available():
    best_net = best_net.cuda()
best_net.load_state_dict(torch.load(os.path.join(Config.saved_models_dir, 'model' + str(best_config['search_best']) + '.pth'))) # Instantialize the model before loading the parameters

_, _, test_loader = data_loaders(best_net, train_dataset, valid_dataset, test_dataset)
roc_auc_score = evaluate(test_loader, Config.evaluation_times, best_net=best_net, metric=best_config['metric'])

best_config['roc_auc_score'] = roc_auc_score
np.save('best_config.npy', best_config)
read_dictionary = np.load('best_config.npy').item()
