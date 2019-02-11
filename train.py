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


def train(train_loader, valid_loader, search_times, **param):
    #net = type(model)().cuda()
    net = type(param['model'])()
    criterion = param['loss_func']
    optimizer = optim.Adam(net.parameters(), lr=param['lr'])

    iteration_number = 0
    loss_per_epoch = {"train_loss": [], "valid_loss": []}
    num_batch_train = len(train_loader)
    num_batch_valid = len(valid_loader)

    for epoch in range(Config.train_number_epochs):
        loss_per_batch = {"train_loss": [], "valid_loss": []}
        print("Epoch number: %d" % (epoch+1))

        with tqdm.tqdm(total=num_batch_train) as pbar_train:
            for i, data in enumerate(train_loader, 0): # Train for one batch
                img0, img1, label = data
                #img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
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
                #img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
                output1, output2 = net.forward(img0, img1)
                loss = criterion(output1, output2, label)
                loss_per_batch["valid_loss"].append(loss.item())
                pbar_valid.set_description("valid loss: {:.4f}".format(np.mean(loss_per_batch['valid_loss'])))
                pbar_valid.update(1)
        
        for key, value in loss_per_batch.items(): # Collect average loss for current epoch
            loss_per_epoch[key].append(np.mean(value))

    torch.save(net.state_dict(), f=os.path.join(Config.saved_models_dir, 'model' + str(search_times) + 'pth')) # Model saving, Only save the parameters (Recommended)

    return loss_per_epoch['valid_loss'][-1]

def evaluate(test_loader, **param):
    net = param['best_net']
    criterion = param['loss_func']

    num_batch_test = len(test_loader)
    loss_per_batch = []

    with tqdm.tqdm(total=num_batch_test) as pbar_test:
        for i, data in enumerate(test_loader, 0): # Test for one batch
            img0, img1, label = data
            #img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            output1, output2 = net(img0, img1)
            loss = criterion(output1, output2, label)
            loss.backward()
            loss_per_batch.append(loss.item())
            pbar_test.set_description("test loss: {:.4f}".format(np.mean(loss_per_batch)))
            pbar_test.update(1)

    return np.mean(loss_per_batch)

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
    test_loader = torch.utils.data.DataLoader(siamese_test_dataset, batch_size=Config.test_batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader


grid_search = {
    "model": [model.DeepID(), model.ChopraNet()],
    #"loss_func": [loss.ContrastiveLoss()],
    "loss_func": [loss.ChopraLoss()],
    "lr": [0.005]
}

train_dataset = datasets.ImageFolder(root=Config.train_dir)
valid_dataset = datasets.ImageFolder(root=Config.valid_dir)
test_dataset = datasets.ImageFolder(root=Config.test_dir)

search_times = 1
best_config = {key:None for key in grid_search.keys()}
best_config['search_best'] = 0
best_config['best_valid_loss'] = np.inf

for model in grid_search['model']:
    train_loader, valid_loader, _ = data_loaders(model, train_dataset, valid_dataset, test_dataset)

    for loss_func in grid_search['loss_func']:
        for lr in grid_search['lr']:
            final_valid_loss = train(train_loader, valid_loader, search_times, model=model, loss_func=loss_func, lr=lr)
            
            if final_valid_loss < best_config['best_valid_loss']:
                best_config['model'] = model
                best_config['loss_func'] = loss_func
                best_config['lr'] = lr
                best_config['search_best'] = search_times
                best_config['best_valid_loss'] = final_valid_loss

                np.save('best_config.npy', best_config)
            
            print("\nGrid search {} is completed.\n".format(search_times))
            search_times += 1

print("The best model is model {} with final valid loss {:.4f}".format(best_config['search_best'], best_config['best_valid_loss']))

#best_net = type(best_config['model'])().cuda()
best_net = type(best_config['model'])()
best_net.load_state_dict(torch.load(os.path.join(Config.saved_models_dir, 'model' + str(best_config['search_best']) + 'pth'))) # Instantialize the model before loading the parameters

_, _, test_loader = data_loaders(best_net, train_dataset, valid_dataset, test_dataset)
test_loss = evaluate(test_loader, best_net=best_net, loss_func=best_config['loss_func'])

best_config['test_loss'] = test_loss
np.save('best_config.npy', best_config)
read_dictionary = np.load('best_config.npy').item()
