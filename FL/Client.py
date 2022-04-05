import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

import Communicator
import Config

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
        """
        Initialises the client

        :param index: client unique ID
        :param ip_address: ip of the client
        :param server_addr: IP address of the server machine or broker
        :param server_port: Port that the server/broker is running on
        :param datalen: Data length for client
        """
        super(Client, self).__init__(index, ip_address, server_addr, server_port, sub_topic='fedserver',
                                     pub_topic='fedbench', client_num=Config.K)
        self.optimizer = None
        self.criterion = None
        self.net = None
        self.split_layer = None
        self.datalen = datalen
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.uninet = Utils.get_model('Unit', Config.model_name, Config.model_len - 1, self.device, Config.model_cfg)

        # Connect to server and ensure server receives messages
        if Config.COMM == 'TCP':
            logger.info('Connecting to Server.')
            self.sock.connect((server_addr, server_port))
        elif Config.COMM == 'AMQP':
            self.send_msg("1")
        elif Config.COMM == 'UDP':
            self.tcp_sock.connect((server_addr, server_port))
            self.send_msg_udp(self.sock, self.tcp_sock, (server_addr, server_port + 1), b'')
            self.server_addr = server_addr
            self.server_port = server_port
            self.computation_time = ['MSG_TRAINING_TIME_PER_ITERATION', self.ip]

    def initialize(self, split_layer, first, LR):
        """

        :param split_layer: client split layer value
        :param first: Indicates the first round
        :param LR: Learning rate value
        """
        # retrieve and build initial client model if it is the first round
        if first:
            self.split_layer = split_layer

            logger.debug('Building Model.')
            self.net = Utils.get_model('Client', Config.model_name, self.split_layer, self.device, Config.model_cfg)
            logger.debug(self.net)
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(self.net.parameters(), lr=LR,
                                   momentum=0.9)

        # retrieve model weights from server
        logger.debug('Receiving Global Weights..')
        if Config.COMM == 'TCP':
            weights = self.recv_msg(self.sock)[1]
        elif Config.COMM == 'UDP':
            # fill in missing values
            weights = Utils.concat_weights_client(self.recv_msg_udp(self.sock), self.net.state_dict())
        else:
            weights = self.q.get()[1]

        # load model from new weights
        self.net.load_state_dict(weights)
        logger.debug('Initialize Finished')

    def train(self, trainloader):
        """

        :param trainloader: Shows how much of the model needs to be trained
        """
        # Training model
        s_time_total = time.time()
        self.net.to(self.device)
        self.net.train()
        if self.split_layer == (Config.model_len - 1):
            for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(trainloader)):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

        e_time_total = time.time()

        # send training time to server
        msg = ['MSG_TRAINING_TIME_PER_ITERATION', self.ip, e_time_total - s_time_total]
        if Config.COMM == 'TCP':
            self.snd_msg_tcp(self.sock, msg)
        elif Config.COMM == 'UDP':
            self.computation_time.append(time.time() - s_time_total)
        else:
            self.send_msg(msg)

    def upload(self):
        # send newly trained weights to server
        msg = ['MSG_LOCAL_WEIGHTS_CLIENT_TO_SERVER', self.net.cpu().state_dict()]
        if Config.COMM == 'TCP':
            self.snd_msg_tcp(self.sock, msg)
        elif Config.COMM == 'UDP':
            self.send_msg_udp(self.sock, self.tcp_sock, (self.server_addr, self.server_port + 1), msg)
        else:
            self.send_msg(msg)

    def reinitialize(self, split_layers, first, LR):
        """
        Calls initialize for rounds 1+

        :param split_layers: client split layer value
        :param first: Indicates the first round
        :param LR: Learning rate value
        """
        self.initialize(split_layers, first, LR)

    def finish(self):
        logger.info(self.packets_sent)
        # Send finishing message to ensure safe deletion
        if Config.COMM == 'UDP':
            self.send_msg_tcp_client(self.tcp_sock, self.computation_time)
        elif Config.COMM == 'TCP':
            pass
        else:
            if self.q.get() == 'DONE':
                pass
