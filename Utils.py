"""Some helper functions for FedAdapt, including:
- get_local_dataloader: split dataset and get respective dataloader.
- get_model: build the model according to location and split layer.
- zero_init: zero initialization.
- fed_avg: FedAvg aggregation.
"""
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

import pickle, struct, socket
from Vgg import *
from Config import *
import collections
import numpy as np

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

np.random.seed(0)
torch.manual_seed(0)

tools = {'cpu': 'stress-ng --cpu 1 --timeout 1500s &',
         'net1': 'netstress -m host &',
         'Wi-Fi': 'sudo tc qdisc add dev ens160 root tbf rate 60mbit latency 50ms burst 1600 &',
         '4G': 'sudo tc qdisc add dev ens160 root tbf rate 20mbit latency 50ms burst 1600 &',
         '3G': 'sudo tc qdisc add dev ens160 root tbf rate 5mbit latency 50ms burst 1600 &'}


def get_local_dataloader(CLIENT_IDEX, cpu_count):
    indices = list(range(N))
    part_tr = indices[int((N / K) * CLIENT_IDEX): int((N / K) * (CLIENT_IDEX + 1))]

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root=dataset_path, train=True, download=False, transform=transform_train)
    subset = Subset(trainset, part_tr)
    trainloader = DataLoader(
        subset, batch_size=B, shuffle=True, num_workers=cpu_count)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, classes


def get_model(location, model_name, layer, device, cfg):
    cfg = cfg.copy()
    net = VGG(location, model_name, layer, cfg)
    net = net.to(device)
    logger.debug(str(net))
    return net


def concat_weights_client(weights, sweights):
    concat_dict = collections.OrderedDict()

    for weight in sweights:
        concat_dict[weight] = []

    for weight in weights:
        concat_dict[weight[0]].append(torch.from_numpy(weight[1]))

    for key in concat_dict:
        if not concat_dict[key]:
            concat_dict[key] = sweights[key]
            continue
            # concat_dict[key] = torch.from_numpy(np.zeros(sweights[key].numel()))
        if concat_dict[key][0].numel() == 1:
            concat_dict[key] = concat_dict[key][0]
            continue
        if torch.cat(concat_dict[key]).size()[0] < sweights[key].numel():
            # length = len(concat_dict[key])
            # concat_dict[key].append(sweights[key][length:])
            concat_dict[key].append(torch.from_numpy(np.zeros((sweights[key].numel()) - torch.cat(concat_dict[key]).size()[0])))
        concat_dict[key] = torch.cat(concat_dict[key])
        if concat_dict[key].size()[0] > sweights[key].numel():
            concat_dict[key] = concat_dict[key][:sweights[key].numel()]
        concat_dict[key] = torch.reshape(concat_dict[key], sweights[key].size())

    return concat_dict


def zero_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.zeros_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            init.zeros_(m.weight)
            init.zeros_(m.bias)
            init.zeros_(m.running_mean)
            init.zeros_(m.running_var)
        elif isinstance(m, nn.Linear):
            init.zeros_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
    return net


def fed_avg(zero_model, w_local_list, totoal_data_size):
    keys = w_local_list[0][0].keys()

    for k in keys:
        for w in w_local_list:
            beta = float(w[1]) / float(totoal_data_size)
            if 'num_batches_tracked' in k:
                zero_model[k] = w[0][k]
            else:
                zero_model[k] += (w[0][k] * beta)

    return zero_model
