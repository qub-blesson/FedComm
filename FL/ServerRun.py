import time
import torch
import pickle
import argparse
from Server import Server
import Config
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append('../')

parser = argparse.ArgumentParser()
parser.add_argument('--communicator', help='Communication protocol', default='TCP')
parser.add_argument('--model', help='Model type: VGG5, VGG8, VGG18', default='VGG8')
parser.add_argument('--stress', help='Tool used to apply stress: cpu, net', default='')
parser.add_argument('--limiter', help='Tool used to limit network: 3G, 4G, Wi-Fi', default='')
parser.add_argument('--rounds', help='Number of training rounds', type=int, default=5)
args = parser.parse_args()
stress = args.stress
limiter = args.limiter
config.R = args.rounds

communicator = args.communicator
if args.model != '':
    config.model_name = args.model
config.COMM = communicator

if config.model_name == 'VGG5':
    config.split_layer = [6] * config.K
    config.model_len = 7

results = '../results/FedBench_'+config.COMM+'_'+limiter+'_'+stress+'_'+config.model_name+'.pkl'

first = True  # First initialization control

server = None
if communicator == 'TCP':
    server = Server(0, config.SERVER_ADDR, config.SERVER_PORT)
else:
    server = Server(config.K, config.SERVER_ADDR, config.SERVER_PORT)
server.initialize(first)
first = False

res = {'training_time': [], 'test_acc_record': [], 'communication_time': []}

for r in range(config.R):
    logger.info('====================================>')
    logger.info('==> Round {:} Start'.format(r))

    s_time = time.time()
    state = server.train()
    server.aggregate(config.CLIENTS_LIST)
    e_time = time.time()

    # Recording each round training time and test accuracy
    training_time = e_time - s_time
    res['training_time'].append(training_time)
    comp_time = 0
    for key in state:
        comp_time += state[key]
    comp_time /= config.K
    res['communication_time'].append(training_time - comp_time)
    test_acc = server.test()
    res['test_acc_record'].append(test_acc)

    with open(results, 'wb') as f:
        pickle.dump(res, f)

    logger.info('Round Finish')
    logger.info('==> Round Training Computation Time: {:}'.format(comp_time))
    logger.info('==> Round Training Communication Time: {:}'.format(training_time - comp_time))

    logger.info('==> Reinitialization for Round : {:}'.format(r + 1))

    server.reinitialize(first)
    logger.info('==> Reinitialization Finish')
comm_time = server.finish(config.CLIENTS_LIST)
