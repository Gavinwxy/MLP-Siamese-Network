import sys
import os
import torch

model_dir = "./models/"


def save_model(model, optimizer, fname, id, best_index, best_loss):
    state = dict()
    state['network'] = model.state_dict()
    state['optimizer'] = optimizer.state_dict()
    state['best_index'] = best_index
    state['best_loss'] = best_loss
    torch.save(state, f=os.path.join(model_dir, "{}_{}".format(fname, str(id))))


def load_model(model, optimizer, fname, id):
    state = torch.load(f=os.path.join(model_dir, "{}_{}".format(fname, str(id))))
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
