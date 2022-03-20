import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import threading
import tqdm
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.append('../')
from Communicator import *
import utils
import config

np.random.seed(0)
torch.manual_seed(0)


class Server(Communicator):
    def __init__(self, index, ip_address, server_port, model_name):
        super(Server, self).__init__(index, ip_address, ip_address, server_port, pub_topic="fedserver",
                                     sub_topic='fedadapt', client_num=config.K)
        self.threads = None
        self.criterion = None
        self.optimizers = None
        self.nets = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.port = server_port
        self.model_name = model_name

        if config.COMM == 'TCP':
            self.sock.bind((self.ip, self.port))
            self.client_socks = {}

            while len(self.client_socks) < config.K:
                self.sock.listen(5)
                logger.info("Waiting Incoming Connections.")
                (client_sock, (ip, port)) = self.sock.accept()
                logger.info('Got connection from ' + str(ip))
                self.client_socks[str(ip)] = client_sock
        elif config.COMM == 'MQTT' or config.COMM == 'AMQP':
            connections = 0
            while connections < config.K:
                connections += int(self.q.get())

            logger.info("Clients have connected to MQTT Server")

        self.uninet = utils.get_model('Unit', self.model_name, config.model_len - 1, self.device, config.model_cfg)

        self.transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
             ])
        self.testset = torchvision.datasets.CIFAR10(root=config.dataset_path, train=False, download=False,
                                                    transform=self.transform_test)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=100, shuffle=False, num_workers=4)

    def initialize(self, first, LR):
        if first:
            self.nets = {}
            self.optimizers = {}
            for i in range(len(config.split_layer)):
                client_ip = config.CLIENTS_LIST[i]
                self.nets[client_ip] = utils.get_model('Server', self.model_name, config.split_layer[i], self.device,
                                                       config.model_cfg)
            self.criterion = nn.CrossEntropyLoss()

        msg = ['MSG_INITIAL_GLOBAL_WEIGHTS_SERVER_TO_CLIENT', self.uninet.state_dict()]
        if config.COMM == 'TCP':
            for i in self.client_socks:
                self.snd_msg_tcp(self.client_socks[i], msg)
        elif config.COMM == 'MQTT' or config.COMM == 'AMQP':
            self.send_msg(msg)

    def train(self, thread_number, client_ips):
        # Training start

        ttpi = {}  # Training time per iteration
        if config.COMM == 'TCP':
            for s in self.client_socks:
                msg = self.recv_msg(self.client_socks[s], 'MSG_TRAINING_TIME_PER_ITERATION')
                ttpi[msg[1]] = msg[2]
        elif config.COMM == 'MQTT' or config.COMM == 'AMQP':
            connections = 0
            while connections != config.K:
                msg = self.q.get()
                while msg[0] != 'MSG_TRAINING_TIME_PER_ITERATION':
                    self.q.put(msg)
                    msg = self.q.get()
                connections += 1
                ttpi[msg[1]] = msg[2]
        return ttpi

    def aggregate(self, client_ips):
        w_local_list = []
        msg = None

        for i in range(len(client_ips)):
            if config.COMM == 'TCP':
                msg = self.recv_msg(self.client_socks[client_ips[i]], 'MSG_LOCAL_WEIGHTS_CLIENT_TO_SERVER')
            elif config.COMM == 'MQTT' or config.COMM == 'AMQP':
                while msg is None:
                    msg = self.q.get()
            w_local = (msg[1], config.N / config.K)
            w_local_list.append(w_local)

        zero_model = utils.zero_init(self.uninet).state_dict()
        aggregated_model = utils.fed_avg(zero_model, w_local_list, config.N)
        self.uninet.load_state_dict(aggregated_model)

    def test(self, r):
        self.uninet.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(self.testloader)):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.uninet(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100. * correct / total
        logger.info('Test Accuracy: {}'.format(acc))

        # Save checkpoint.
        torch.save(self.uninet.state_dict(), './' + config.model_name + '.pth')

        return acc

    def reinitialize(self, first, LR):
        self.initialize(first, LR)

    def finish(self, client_ips):
        msg = []
        if config.COMM == 'TCP':
            for i in range(len(client_ips)):
                msg.append(self.recv_msg(self.client_socks[client_ips[i]], 'MSG_COMMUNICATION_TIME')[1])
        elif config.COMM == 'MQTT' or config.COMM == 'AMQP':
            connections = 0
            while connections != config.K:
                msg.append(self.q.get()[1])
                connections += 1
            self.send_msg(['DONE'])
        return max(msg)
