import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

sys.path.append('../')
import Utils
from Communicator import *

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

np.random.seed(0)
torch.manual_seed(0)


class Client(Communicator):
    def __init__(self, index, ip_address, server_addr, server_port, datalen):
        super(Client, self).__init__(index, ip_address, server_addr, server_port, sub_topic='fedserver',
                                     pub_topic='fedbench', client_num=Config.K)
        self.optimizer = None
        self.criterion = None
        self.net = None
        self.split_layer = None
        self.datalen = datalen
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.uninet = Utils.get_model('Unit', Config.model_name, Config.model_len - 1, self.device, Config.model_cfg)

        if Config.COMM == 'TCP':
            logger.info('Connecting to Server.')
            self.sock.connect((server_addr, server_port))
        elif Config.COMM == 'AMQP':
            self.send_msg("1")

    def initialize(self, split_layer, first, LR):
        if first:
            self.split_layer = split_layer

            logger.debug('Building Model.')
            self.net = Utils.get_model('Client', Config.model_name, self.split_layer, self.device, Config.model_cfg)
            logger.debug(self.net)
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(self.net.parameters(), lr=LR,
                                   momentum=0.9)
        logger.debug('Receiving Global Weights..')
        if Config.COMM == 'TCP':
            weights = self.recv_msg(self.sock)[1]
        else:
            weights = self.q.get()[1]

        self.net.load_state_dict(weights)
        logger.debug('Initialize Finished')

    def train(self, trainloader):
        # Training start
        s_time_total = time.time()
        self.net.to(self.device)
        self.net.train()
        if self.split_layer == (Config.model_len - 1):  # No offloading training
            for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(trainloader)):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

        e_time_total = time.time()

        msg = ['MSG_TRAINING_TIME_PER_ITERATION', self.ip, e_time_total - s_time_total]
        if Config.COMM == 'TCP':
            self.snd_msg_tcp(self.sock, msg)
        else:
            self.send_msg(msg)

    def upload(self):
        msg = ['MSG_LOCAL_WEIGHTS_CLIENT_TO_SERVER', self.net.cpu().state_dict()]
        start = time.time()
        if Config.COMM == 'TCP':
            self.snd_msg_tcp(self.sock, msg)
        else:
            self.send_msg(msg)
        Config.comm_time += (time.time() - start)

    def reinitialize(self, split_layers, first, LR):
        self.initialize(split_layers, first, LR)

    def finish(self):
        msg = ['MSG_COMMUNICATION_TIME', Config.comm_time]
        if Config.COMM == 'TCP':
            self.snd_msg_tcp(self.sock, msg)
        else:
            self.send_msg(msg)
            if self.q.get() == 'DONE':
                pass
