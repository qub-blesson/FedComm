import time
import torch
import pickle
import argparse
from Server import Server
import Config
import logging
import sys
import numpy as np

# set logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# add all files to path
sys.path.append('../')

# set arg parse
parser = argparse.ArgumentParser()
parser.add_argument('--communicator', help='Communication protocol', default='TCP')
parser.add_argument('--model', help='Model type: VGG5, VGG8, VGG18', default='VGG8')
parser.add_argument('--stress', help='Tool used to apply stress: cpu, net', default='')
parser.add_argument('--limiter', help='Tool used to limit network: 3G, 4G, Wi-Fi', default='')
parser.add_argument('--rounds', help='Number of training rounds', type=int, default=5)
args = parser.parse_args()
# store parameters based on input
stress = args.stress
limiter = args.limiter
Config.R = args.rounds

# set communication protocol
communicator = args.communicator
if args.model != '':
    Config.model_name = args.model
Config.COMM = communicator

# update VGG5 config if selected
if Config.model_name == 'VGG5':
    Config.split_layer = [6] * Config.K
    Config.model_len = 7

# name results file
results = '../results/FedBench_'+Config.COMM+'_'+limiter+'_'+stress+'_'+Config.model_name+'.pkl'

# First initialization control
first = True

# make server based on application layer protocol selected
server = None
if communicator == 'TCP' or communicator == 'UDP':
    server = Server(0, Config.SERVER_ADDR, Config.SERVER_PORT)
else:
    server = Server(Config.K, Config.SERVER_ADDR, Config.SERVER_PORT)
# initialize server
server.initialize(first)

# set first to false
first = False

# initialize results
res = {'training_time': [], 'test_acc_record': [], 'communication_time': []}

# start FL process on a per-round basis
for r in range(Config.R):
    logger.info('====================================>')
    logger.info('==> Round {:} Start'.format(r))

    # time training and aggregation process
    s_time = time.time()
    state = server.train()
    server.aggregate(Config.CLIENTS_LIST)
    e_time = time.time()

    # Recording each round training time and test accuracy
    training_time = e_time - s_time
    res['training_time'].append(training_time)
    if communicator != 'UDP':
        comp_time = 0
        for key in state:
            comp_time += state[key]
        comp_time /= Config.K
        res['communication_time'].append(training_time - comp_time)
    test_acc = server.test()
    res['test_acc_record'].append(test_acc)

    # write results to pickle output file
    with open(results, 'wb') as f:
        pickle.dump(res, f)

    # log for user
    logger.info('Round Finish')
    if communicator != 'UDP':
        logger.info('==> Round Training Computation Time: {:}'.format(comp_time))
        logger.info('==> Round Training Communication Time: {:}'.format(training_time - comp_time))

    logger.info('==> Reinitialization for Round : {:}'.format(r + 1))

    # re-initialise for new round
    server.reinitialize(first)
    logger.info('==> Reinitialization Finish')
# server finish
state = server.finish()

if communicator == 'UDP':
    comp_time = np.array([0] * Config.R)

    for i in state:
        comp_time = np.add(comp_time, state[i])
    comp_time /= Config.K
    for i in range(Config.R):
        res['communication_time'].append(res['training_time'][i] - comp_time[i])
    with open(results, 'wb') as f:
        pickle.dump(res, f)
