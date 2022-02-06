import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import time
import numpy as np
import sys

sys.path.append('../')
import config
import utils
from Communicator import *

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

np.random.seed(0)
torch.manual_seed(0)


class Client(Communicator):
    def __init__(self, index, ip_address, server_addr, server_port, datalen, model_name, split_layer):
        super(Client, self).__init__(index, ip_address, server_addr, server_port, sub_topic='fedserver',
                                     pub_topic='fedadapt', client_num=config.K)
        self.datalen = datalen
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.uninet = utils.get_model('Unit', self.model_name, config.model_len - 1, self.device, config.model_cfg)
        self.sock.connect((server_addr, server_port))
        self.server_addr = server_addr
        self.server_port = server_port

    def initialize(self, split_layer, offload, first, LR):
        if offload or first:
            self.split_layer = split_layer

            logger.debug('Building Model.')
            self.net = utils.get_model('Client', self.model_name, self.split_layer, self.device, config.model_cfg)
            logger.debug(self.net)
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(self.net.parameters(), lr=LR,
                                   momentum=0.9)
        logger.debug('Receiving Global Weights..')
        weights = None
        weights = utils.concat_weights_client(self.recv_msg_udp(self.sock), self.net.state_dict())

        if self.split_layer == (config.model_len - 1):
            self.net.load_state_dict(weights)
        else:
            pweights = utils.split_weights_client(weights, self.net.state_dict())
            self.net.load_state_dict(pweights)
        logger.debug('Initialize Finished')

    def train(self, trainloader):
        # Training start
        s_time_total = time.time()
        time_training_c = 0
        self.net.to(self.device)
        self.net.train()
        if self.split_layer == (config.model_len - 1):  # No offloading training
            for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(trainloader)):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

        e_time_total = time.time()

        msg = ['MSG_TRAINING_TIME_PER_ITERATION', e_time_total - s_time_total]
        self.send_msg_tcp_client(self.sock, (self.server_addr, self.server_port), msg)

        return

    def upload(self):
        msg = ['MSG_LOCAL_WEIGHTS_CLIENT_TO_SERVER', self.net.cpu().state_dict()]
        self.send_msg_udp(self.sock, (self.server_addr, self.server_port), msg)

    def reinitialize(self, split_layers, offload, first, LR):
        self.initialize(split_layers, offload, first, LR)

    def finish(self):
        logger.info(self.packets_sent)
        msg = ['MSG_COMMUNICATION_TIME', config.comm_time]
        self.send_msg_udp(self.sock, (self.server_addr, self.server_port), msg)
