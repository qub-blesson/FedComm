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

# set log level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys

# include all files to path
sys.path.append('../')
from Communicator import *
import Utils
import Config

# set seed
np.random.seed(0)
torch.manual_seed(0)


# server class
class Server(Communicator):
    def __init__(self, index, ip_address, server_port, test=False):
        """
        Initialise server

        :param index: unique ID for server
        :param ip_address: Server address
        :param server_port: Port to host server on
        """
        super(Server, self).__init__(index, ip_address, ip_address, server_port, pub_topic="fedserver",
                                     sub_topic='fedbench', client_num=Config.K)
        # init variables
        self.testloader = None
        self.testset = None
        self.transform_test = None
        self.uninet = None
        self.client_socks = None
        self.client_ip = None
        self.criterion = None
        self.optimizers = None
        self.nets = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.port = server_port

        if not test:
            self.connect_devices()
            self.get_model()

    def initialize(self, first):
        """
        Initialise federated learning process for server

        :param first: Indicates first initial round
        """
        self.get_empty_client_model(first)
        self.send_initial_model_weights()

    def train(self):
        """
        Get training time per client

        :return: training time per iteration
        """

        # Retrieve TTPI per client
        ttpi = {}  # Training time per iteration
        if Config.COMM == 'TCP':
            for s in self.client_socks:
                msg = self.recv_msg(self.client_socks[s], 'MSG_TRAINING_TIME_PER_ITERATION')
                ttpi[msg[1]] = msg[2]
        elif Config.COMM == 'UDP':
            return
        else:
            connections = 0
            while connections != Config.K:
                msg = self.q.get()
                while msg[0] != 'MSG_TRAINING_TIME_PER_ITERATION':
                    self.q.put(msg)
                    msg = self.q.get()
                connections += 1
                ttpi[msg[1]] = msg[2]
        return ttpi

    def aggregate(self, client_ips):
        """
        combines weights into a single model to send out to clients next round

        :param client_ips: ip of all clients in FL process
        """
        w_local_list = []
        if Config.COMM == 'UDP':
            msg = self.recv_msg_udp_agg(self.sock, 'MSG_LOCAL_WEIGHTS_CLIENT_TO_SERVER')
            for weight in msg:
                weights = Utils.concat_weights_client(msg[weight], self.uninet.state_dict())
                w_local = (weights, Config.N / Config.K)
                w_local_list.append(w_local)

        for i in range(len(client_ips)):
            msg = None
            if Config.COMM == 'TCP':
                msg = self.recv_msg(self.client_socks[client_ips[i]], 'MSG_LOCAL_WEIGHTS_CLIENT_TO_SERVER')
            elif Config.COMM != 'UDP':
                while msg is None:
                    msg = self.q.get()
            if Config.COMM != 'UDP':
                w_local = (msg[1], Config.N / Config.K)
                w_local_list.append(w_local)
        zero_model = Utils.zero_init(self.uninet).state_dict()
        # average the model weights
        aggregated_model = Utils.fed_avg(zero_model, w_local_list, Config.N)

        # load model with new aggregated weights
        self.uninet.load_state_dict(aggregated_model)

    def test(self):
        """
        test the new model and get the accuracy

        :return: Accuracy from testing
        """
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
        torch.save(self.uninet.state_dict(), './' + Config.model_name + '.pth')

        return acc

    def reinitialize(self, first):
        """
        Calls initialize for rounds 1+

        :param first: Indicate initial starting round
        """
        self.initialize(first)

    def finish(self):
        """
        Finish FL process from server

        :return: training time per iteration for UDP
        """
        if Config.COMM == 'UDP':
            while len(self.udp_ttpi) < 4:
                pass
            logger.info(self.packets_received)
            return self.udp_ttpi
        self.send_msg(['DONE'])

    def connect_devices(self):
        # wait for incoming connections via various application layer protocols
        logger.info("Waiting For Incoming Connections.")
        if Config.COMM == 'TCP':
            self.sock.bind((self.ip, self.port))
            self.client_socks = {}

            while len(self.client_socks) < Config.K:
                self.sock.listen(5)
                (client_sock, (ip, port)) = self.sock.accept()
                logger.info('Got connection from ' + str(ip))
                self.client_socks[str(ip)] = client_sock
        elif Config.COMM == 'MQTT' or Config.COMM == 'AMQP':
            connections = 0
            while connections < Config.K:
                connections += int(self.q.get())

            logger.info("Clients have connected")
        elif Config.COMM == 'UDP':
            self.sock.bind((self.ip, self.port + 1))
            self.tcp_sock.bind((self.ip, self.port))
            self.client_socks = {}
            self.client_ip = {}

            while len(self.client_socks) < Config.K:
                self.tcp_sock.listen(5)
                (client_sock, (ip, port)) = self.tcp_sock.accept()
                self.client_socks[str(ip)] = client_sock

            self.thread = threading.Thread(target=self.recv_end, args=[self.client_socks])
            self.thread.start()

            while len(self.client_ip) < Config.K:
                msg = self.init_recv_msg_udp(self.sock)
                logger.info('Got connection from ' + str(msg[0]))
                self.client_ip[str(msg[0])] = (msg[0], msg[1])

    def get_model(self):
        # get initial model for server
        self.uninet = Utils.get_model('Unit', Config.model_name, Config.model_len - 1, self.device, Config.model_cfg)

        self.transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
             ])
        self.testset = torchvision.datasets.CIFAR10(root=Config.dataset_path, train=False, download=False,
                                                    transform=self.transform_test)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=100, shuffle=False, num_workers=4)

    def get_empty_client_model(self, first):
        if first:
            self.nets = {}
            # TODO: perhaps remove optimizers
            self.optimizers = {}
            for i in range(len(Config.split_layer)):
                client_ip = Config.CLIENTS_LIST[i]
                self.nets[client_ip] = Utils.get_model('Server', Config.model_name, Config.split_layer[i], self.device,
                                                       Config.model_cfg)
            self.criterion = nn.CrossEntropyLoss()

    def send_initial_model_weights(self):
        # send initial weights to all clients
        msg = ['MSG_INITIAL_GLOBAL_WEIGHTS_SERVER_TO_CLIENT', self.uninet.state_dict()]
        if Config.COMM == 'TCP':
            for i in self.client_socks:
                self.snd_msg_tcp(self.client_socks[i], msg)
        elif Config.COMM == 'UDP':
            for i in self.client_ip:
                self.send_msg_udp(self.sock, self.client_socks[i], self.client_ip[i], msg)
        else:
            self.send_msg(msg)
