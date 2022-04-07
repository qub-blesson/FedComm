import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import tqdm
from torchvision.transforms import transforms

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

        self.build_optimize_model(split_layer, first, LR)
        weights = self.retrieve_model_weights()
        self.load_new_model(weights)

    def train(self, trainloader):
        """

        :param trainloader: Shows how much of the model needs to be trained
        """
        s_time_total, e_time_total = self.train_model(trainloader)
        self.send_training_time_to_server(s_time_total, e_time_total)

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
        # Send finishing message to ensure safe deletion
        if Config.COMM == 'UDP':
            logger.info(self.packets_sent)
            self.send_msg_tcp_client(self.tcp_sock, self.computation_time)
        elif Config.COMM == 'TCP':
            pass
        else:
            if self.q.get() == 'DONE':
                pass

    def retrieve_model_weights(self):
        # retrieve model weights from server
        logger.debug('Receiving Global Weights..')
        if Config.COMM == 'TCP':
            return self.recv_msg(self.sock)[1]
        elif Config.COMM == 'UDP':
            # fill in missing values
            return Utils.concat_weights_client(self.recv_msg_udp(self.sock), self.net.state_dict())
        else:
            return self.q.get()[1]

    def build_optimize_model(self, split_layer, first, LR):
        # retrieve and build initial client model if it is the first round
        if first:
            self.split_layer = split_layer

            logger.debug('Building Model.')
            self.net = Utils.get_model('Client', Config.model_name, self.split_layer, self.device, Config.model_cfg)
            logger.debug(self.net)
            self.criterion = nn.CrossEntropyLoss()

        if self.net is not None:
            self.optimizer = optim.SGD(self.net.parameters(), lr=LR,
                                       momentum=0.9)

    def load_new_model(self, weights):
        # load model from new weights
        self.net.load_state_dict(weights)
        logger.debug('Initialize Finished')

    def train_model(self, trainloader):
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

        return s_time_total, e_time_total

    def send_training_time_to_server(self, s_time_total, e_time_total):
        # send training time to server
        msg = ['MSG_TRAINING_TIME_PER_ITERATION', self.ip, e_time_total - s_time_total]
        if Config.COMM == 'TCP':
            self.snd_msg_tcp(self.sock, msg)
        elif Config.COMM == 'UDP':
            self.computation_time.append(time.time() - s_time_total)
        else:
            self.send_msg(msg)

    # testing purposes
    def test(self):
        """
        test the new model and get the accuracy

        :return: Accuracy from testing
        """
        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
             ])
        testset = torchvision.datasets.CIFAR10(root=Config.dataset_path, train=False, download=False,
                                               transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(testloader)):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100. * correct / total
        logger.info('Test Accuracy: {}'.format(acc))

        # Save checkpoint.
        torch.save(self.net.state_dict(), './' + Config.model_name + '.pth')

        return acc
