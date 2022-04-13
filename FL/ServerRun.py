import time
import torch
import pickle
import argparse
from FL.Server import Server
import Config
import logging
import sys
import numpy as np

# set logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# add all files to path
sys.path.append('../')


class ServerRun:
    def __init__(self, communicator, model, stress, limiter, test=False):
        # global variables
        self.stress = stress
        self.limiter = limiter
        self.communicator = communicator
        self.model = model
        self.results = ''
        self.server = None
        self.first = True
        self.res = {}
        self.state = None
        self.comp_time = 0

        if not test:
            self.update_model()
            self.create_results_file()
            self.init_server()
            self.start_FL_process()
            self.finish_server()

    def update_model(self):
        if self.model != '':
            Config.model_name = self.model
        Config.COMM = self.communicator

        # update VGG5 config if selected
        if Config.model_name == 'VGG5':
            Config.split_layer = [6] * Config.K
            Config.model_len = 7

    def create_results_file(self):
        # name results file
        self.results = '../results/FedBench_' + Config.COMM + '_' + self.limiter + '_' + self.stress + '_'\
                  + Config.model_name + '.pkl'
        # initialize results
        self.res = {'training_time': [], 'test_acc_record': [], 'communication_time': []}

    def init_server(self):
        # make server based on application layer protocol selected
        if self.communicator == 'TCP' or self.communicator == 'UDP':
            self.server = Server(0, Config.SERVER_ADDR, Config.SERVER_PORT)
        else:
            self.server = Server(Config.K, Config.SERVER_ADDR, Config.SERVER_PORT)
        # initialize server
        self.server.initialize(self.first)

        # set first to false
        self.first = False

    def start_FL_process(self):
        # start FL process on a per-round basis
        for r in range(Config.R):
            logger.info('====================================>')
            logger.info('==> Round {:} Start'.format(r))

            # time training and aggregation process
            s_time = time.time()
            self.state = self.server.train()
            self.server.aggregate(Config.CLIENTS_LIST)
            e_time = time.time()

            # Recording each round training time and test accuracy
            training_time = e_time - s_time
            self.res['training_time'].append(training_time)
            if self.communicator != 'UDP':
                self.comp_time = 0
                for key in self.state:
                    self.comp_time += self.state[key]
                self.comp_time /= Config.K
                self.res['communication_time'].append(training_time - self.comp_time)
            test_acc = self.server.test()
            self.res['test_acc_record'].append(test_acc)

            # write results to pickle output file
            with open(self.results, 'wb') as f:
                pickle.dump(self.res, f)

            # log for user
            logger.info('Round Finish')
            if self.communicator != 'UDP':
                logger.info('==> Round Training Computation Time: {:}'.format(self.comp_time))
                logger.info('==> Round Training Communication Time: {:}'.format(training_time - self.comp_time))

            logger.info('==> Reinitialization for Round : {:}'.format(r + 1))

            # re-initialise for new round
            self.server.reinitialize(self.first)
            logger.info('==> Reinitialization Finish')

    def finish_server(self):
        # server finish
        state = self.server.finish()

        if self.communicator == 'UDP':
            comp_time = np.array([0] * Config.R)

            for i in state:
                comp_time = np.add(comp_time, state[i])
            comp_time /= Config.K
            for i in range(Config.R):
                self.res['communication_time'].append(self.res['training_time'][i] - comp_time[i])
            with open(self.results, 'wb') as f:
                pickle.dump(self.res, f)
