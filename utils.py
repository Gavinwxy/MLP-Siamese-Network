import sys
import os
import torch
import csv
import pandas as pd
from Config import Config

model_dir = "./models/"


class LocalParam:
    def __init__(self, exp_name, model, loss, metric, lr, best_index=0, best_loss=float('inf')):
        self.exp_name = exp_name
        self.model = model
        self.loss = loss
        self.metric = metric
        self.lr = lr
        self.best_index = best_index
        self.best_loss = best_loss


def save_model(model, optimizer, fname, id, best_index, best_loss):
    state = dict()
    state['network'] = model.state_dict()
    state['optimizer'] = optimizer.state_dict()
    state['best_index'] = best_index
    state['best_loss'] = best_loss
    dir = model_dir + fname + "/"
    if not os.path.exists(dir):
        os.mkdir(dir)
    torch.save(state, f=os.path.join(dir, str(id)))


def load_model(model, optimizer, fname, id):
    dir = model_dir + fname + "/"
    state = torch.load(f=os.path.join(dir, str(id)))
    model.load_state_dict(state['network'])
    optimizer.load_state_dict(state['optimizer'])
    return state['best_index'], state['best_loss']


def create_model(net_name, *args, **kwargs):
    pack_meta = __import__('model', globals(), locals(), [net_name])
    net_meta = getattr(pack_meta, net_name)
    net = net_meta(*args, **kwargs)
    return net


def create_loss(loss_name, *args, **kwargs):
    pack_meta = __import__('loss', globals(), locals(), [loss_name])
    loss_meta = getattr(pack_meta, loss_name)
    loss = loss_meta(*args, **kwargs)
    return loss


def log_model(param):
    p = [param.exp_name, param.model, param.loss, param.metric, param.lr, param.best_index, param.best_loss]
    path = os.path.join(model_dir, "best_model.csv")
    csv_file = open(path, "w")
    writer = csv.writer(csv_file)
    writer.writerow(p)
    csv_file.close()


def get_local_param(exp_name):
    path = os.path.join(model_dir, "best_model.csv")
    models = pd.read_csv(path, header=None)
    exp = models[models[0] == exp_name].ix[0].tolist()
    return LocalParam(exp[0], exp[1], exp[2], exp[3], exp[4], exp[5], exp[6])


def del_models(exp_name):
    dir = model_dir + exp_name + "/"
    path = os.path.join(dir, str(Config.train_number_epochs - 1))
    state = torch.load(path)
    index = state['best_index']
    print("Best model index:", index)
    print("delete redundant models...")
    for i in range(0, Config.train_number_epochs - 1):
        if i != index:
            os.remove(os.path.join(dir, str(i)))
    print("deletion completed")
