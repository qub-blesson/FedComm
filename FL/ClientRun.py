import signal

import torch
import socket
import time
import multiprocessing
import os
import argparse
from FL.Client import Client
import Config
import Utils
import logging
import sys

# set logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# add all files to path
sys.path.append('../')


class ClientRun:
    def __init__(self, communicator, model, stress, limiter, monitor, test=False):
        # global variables
        self.stress = stress
        self.limiter = limiter
        self.monitor = monitor
        self.model = model
        self.communicator = communicator

        self.index = None
        self.ip_address = None
        self.split_layer = None
        self.LR = None
        self.datalen = None
        self.trainloader = None
        self.client = None
        self.first = None

        if not test:
            self.set_args()
            self.init_client_values()
            self.apply_stress()
            self.monitor_network()
            self.limit_network_bandwidth()
            self.init_client()
            self.start_FL_process()
            self.finish_client()

    def set_args(self):
        if self.model != '':
            Config.model_name = self.model
        Config.COMM = self.communicator

        if Config.model_name == 'VGG5':
            Config.split_layer = [6] * Config.K
            Config.model_len = 7

    def init_client_values(self):
        # set init values to send to client
        try:
            self.ip_address = Config.HOST2IP[socket.gethostname()]
            self.index = Config.CLIENTS_CONFIG[self.ip_address]
            self.split_layer = Config.split_layer[self.index]
        except KeyError as e:
            print("hostname not in list")
        self.datalen = Config.N / Config.K
        self.LR = Config.LR

    # could not test on Windows machine
    def apply_stress(self):
        # apply stress
        # os.system has been tested - we can expect it too work
        if self.stress is not None:
            if self.stress == 'cpu':  # or int(ip_address[:-1]) % 2 == 0:
                os.system('sudo test')
                os.system(Utils.tools[self.stress])
            elif self.stress == 'net':
                if self.index % 2 == 0:
                    os.system('netstress -m host -n 9999 &')
                else:
                    os.system('netstress -s slave -h '+Config.CLIENTS_LIST[self.index+1]+'-n 9999 &')
                os.system(signal.SIGINT.__str__())

    # could not test on Windows machine
    def monitor_network(self):
        if self.monitor is not None:
            # os.system has been tested - we can expect it too work
            os.system('sudo test')
            os.system('sudo tshark -q -z io,stat,30,"COUNT(tcp.analysis.retransmission) tcp.analysis.retransmission" &')

    # could not test on Windows machine
    def limit_network_bandwidth(self):
        # apply network limit
        if self.limiter is not None:
            # os.system has been tested - we can expect it too work
            os.system('sudo tc qdisc del dev ens160 root')
            os.system(Utils.tools[self.limiter])

    def init_client(self):
        logger.info('Preparing Client')
        # call client - sets up client for FL process
        self.client = Client(self.index, self.ip_address, Config.SERVER_ADDR, Config.SERVER_PORT, self.datalen)

        # initialise client
        self.first = True  # First initialization control
        self.client.initialize(self.split_layer, self.first, self.LR)
        self.first = False

        # get number of cores
        logger.info('Preparing Data.')
        cpu_count = multiprocessing.cpu_count()
        self.trainloader, classes = Utils.get_local_dataloader(self.index, cpu_count)

    def start_FL_process(self):
        if self.client is None:
            pass
        # start training process
        for r in range(Config.R):
            # logs for user
            logger.info('====================================>')
            logger.info('ROUND: {} START'.format(r))

            self.client.train(self.trainloader)
            logger.info('ROUND: {} END'.format(r))

            logger.info('==> Waiting for aggregration')
            self.client.upload()

            logger.info('==> Reinitialization for Round : {:}'.format(r + 1))
            s_time_rebuild = time.time()

            # decrease learning rate from round 50
            if r > 49:
                self.LR = Config.LR * 0.1

            # reinitialise for next round
            self.client.reinitialize(Config.split_layer[self.index], self.first, self.LR)
            e_time_rebuild = time.time()
            logger.info('Rebuild time: ' + str(e_time_rebuild - s_time_rebuild))
            logger.info('==> Reinitialization Finish')

        logger.info('FL Training')

    def finish_client(self):
        # finish client
        self.client.finish()
        if self.monitor is not None:
            os.system('sudo pkill tshark')
        if Config.COMM == 'UDP':
            while True:
                pass
