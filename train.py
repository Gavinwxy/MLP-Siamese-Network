import os
import random
import model
import loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import transforms, datasets
from torch.autograd import Variable
from dataset import SiameseNetworkDataset, TripletDataset
from Config import Config
import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from functools import partial
from copy import deepcopy
import model_resnet 
import model_xception

seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
metric_learning_losses = {'LogisticLoss', 'CosFace', 'ArcFace'}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def train(train_loader, valid_loader, search_times, **param):
    global device

    net = deepcopy(param['model']).to(device)
    if param['loss_func'].__name__ in metric_learning_losses:
        criterion = param['loss_func']()
    else:
        criterion = param['loss_func'](metric=param['metric'])
    optimizer = optim.Adam(net.parameters(), lr=param['lr'], weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, Config.train_number_epochs) 
    loss_per_epoch = {"train_loss": [], "valid_loss": []}
    num_batch_train = len(train_loader)
    num_batch_valid = len(valid_loader)

    best_epoch = 1
    for epoch in range(Config.train_number_epochs):
        loss_per_batch = {"train_loss": [], "valid_loss": []}
        print("Epoch number: %d" % (epoch+1))

        with tqdm.tqdm(total=num_batch_train) as pbar_train:
            for _, data in enumerate(train_loader, 0): # Train for one batch
                if criterion.__class__.__name__ == 'TripletLoss':
                    img0, img1, img2 = data
                    img0, img1, img2 = img0.to(device), img1.to(device), img2.to(device)
                    output1, output2, output3 = net(img0), net(img1), net(img2)
                    loss = criterion(output1, output2, output3)
                elif criterion.__class__.__name__ in metric_learning_losses:
                    img0, img1, label = data
                    img0, img1, label = img0.to(device), img1.to(device), label.to(device)
                    label_flatten = label.view(label.shape[0]).long()
                    if criterion.__class__.__name__ == 'LogisticLoss':
                        output = net.forward_logistic_loss(img0, img1)
                    elif criterion.__class__.__name__ == 'CosFace':
                        output = net.forward_cosine_face(img0, img1, label_flatten)
                    else:
                        output = net.forward_arc_face(img0, img1, label_flatten)
                    loss = criterion(output, label_flatten)
                else:
                    img0, img1, label = data
                    img0, img1, label = img0.to(device), img1.to(device), label.to(device)
                    output1, output2 = net(img0), net(img1)
                    loss = criterion(output1, output2, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                loss_per_batch["train_loss"].append(loss.item())
                pbar_train.set_description("train loss: {:.4f}".format(np.mean(loss_per_batch['train_loss'])))
                pbar_train.update(1)

        with tqdm.tqdm(total=num_batch_valid) as pbar_valid:
            for _, data in enumerate(valid_loader, 0): # Evaluation for one batch
                if criterion.__class__.__name__ == 'TripletLoss':
                    img0, img1, img2 = data
                    img0, img1, img2 = img0.to(device), img1.to(device), img2.to(device)
                    output1, output2, output3 = net(img0), net(img1), net(img2)
                    loss = criterion(output1, output2, output3)
                elif criterion.__class__.__name__ in metric_learning_losses:
                    img0, img1, label = data
                    img0, img1, label = img0.to(device), img1.to(device), label.to(device)
                    label_flatten = label.view(label.shape[0]).long()
                    if criterion.__class__.__name__ == 'LogisticLoss':
                        output = net.forward_logistic_loss(img0, img1)
                    elif criterion.__class__.__name__ == 'CosFace':
                        output = net.forward_cosine_face(img0, img1, label_flatten)
                    else:
                        output = net.forward_arc_face(img0, img1, label_flatten)
                    loss = criterion(output, label_flatten)
                else:
                    img0, img1, label = data
                    img0, img1, label = img0.to(device), img1.to(device), label.to(device)
                    output1, output2 = net(img0), net(img1)
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
    global device

    net = param['best_net']
    loss_func = param['loss_func']
    metric = param['metric']

    roc_auc_scores = []
    with tqdm.tqdm(total=loop_times) as pbar_test:
        for _ in range(loop_times):
            y_true = []
            y_score = []
            for _, data in enumerate(test_loader, 0):
                img0, img1, label = data
                img0, img1, label = img0.to(device), img1.to(device), label.item()

                distance = None
                if loss_func.__name__ in metric_learning_losses:
                    if loss_func.__name__ == 'LogisticLoss':
                        distance = net.forward_logistic_loss(img0, img1)[0, 1].item()
                    if loss_func.__name__ == 'CosFace':
                        distance = net.forward_cosine_face(img0, img1)[0, 1].item()
                    else:
                        distance = net.forward_arc_face(img0, img1)[0, 1].item()
                else:
                    output1, output2 = net(img0), net(img1)
                    distance = metric(output1, output2).item()

                if not np.isnan(distance):
                    y_true.append(label)
                    y_score.append(distance)

            roc_auc_scores.append(roc_auc_score(y_true, y_score))
            pbar_test.set_description("ROC_AUC score: {:.4f}".format(np.mean(roc_auc_scores)))
            pbar_test.update(1)

    return np.mean(roc_auc_scores)

def data_loaders(model, loss_func, train_dataset, valid_dataset, test_dataset):
    data_transform = transforms.Compose([
        transforms.Resize(model.input_size[1:]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    grayscale = model.input_size[0] != 3

    if loss_func.__name__ != 'TripletLoss':
        train_dataset = SiameseNetworkDataset(imageFolderDataset=train_dataset,
                                                       transform=data_transform,
                                                       grayscale=grayscale)

        valid_dataset = SiameseNetworkDataset(imageFolderDataset=valid_dataset,
                                                       transform=data_transform,
                                                       grayscale=grayscale)
    else:
        train_dataset = TripletDataset(imageFolderDataset=train_dataset,
                                                transform=data_transform,
                                                grayscale=grayscale)

        valid_dataset = TripletDataset(imageFolderDataset=valid_dataset,
                                                transform=data_transform,
                                                grayscale=grayscale)

    test_dataset = SiameseNetworkDataset(imageFolderDataset=test_dataset,
                                                  transform=data_transform,
                                                  grayscale=grayscale)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Config.train_batch_size, shuffle=True, num_workers=Config.num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=Config.valid_batch_size, shuffle=True, num_workers=Config.num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=Config.num_workers)

    return train_loader, valid_loader, test_loader


grid_search = {
    "model": [model.DeepID()],
    #"loss_func": [loss.ContrastiveLoss],
    #"loss_func": [loss.TripletLoss],
    #"loss_func": [loss.LogisticLoss],
    #"loss_func": [loss.CosFace],
    "loss_func": [loss.ArcFace],
    "metric": [partial(F.pairwise_distance, p=2)],
    #"lr": [1e-07, 5e-06, 0.0001, 0.005, 0.1]
    "lr": [3e-3]
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
    for metric in grid_search['metric']:
        for lr in grid_search['lr']:
            valid_losses = []
            for loss_func in grid_search['loss_func']:
                train_loader, valid_loader, _ = data_loaders(model, loss_func, train_dataset, valid_dataset, test_dataset)
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

best_net = deepcopy(best_config['model']).to(device)
best_net.load_state_dict(torch.load(os.path.join(Config.saved_models_dir, 'model' + str(best_config['search_best']) + '.pth'))) # Instantialize the model before loading the parameters
best_net.eval()

loss_func = best_config['loss_func']

_, _, test_loader = data_loaders(best_net, loss_func, train_dataset, valid_dataset, test_dataset)
test_roc_auc = evaluate(test_loader, Config.evaluation_times_test, best_net=best_net, loss_func=loss_func, metric=best_config['metric'])

best_config['test_roc_auc'] = test_roc_auc
np.save('best_config.npy', best_config)
read_dictionary = np.load('best_config.npy').item()
