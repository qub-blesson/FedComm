import time
import torch
import pickle
import argparse
from Server import Server
import config
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append('../')

parser = argparse.ArgumentParser()
parser.add_argument('--communicator', help='Communication protocol', default='TCP')
parser.add_argument('--model', help='Model type', default='VGG8')
args = parser.parse_args()

LR = config.LR
communicator = args.communicator
config.model_name = args.model
config.COMM = communicator

if config.model_name == 'VGG5':
    config.split_layer = [6, 6, 6, 6]
    config.model_len = 7

first = True  # First initialization control

logger.info('Preparing Sever.')

server = None
if communicator == 'TCP':
    server = Server(0, config.SERVER_ADDR, config.SERVER_PORT)
else:
    server = Server(config.K, config.SERVER_ADDR, config.SERVER_PORT)
server.initialize(first, LR)
first = False

logger.info('Classic FL Training')

res = {'training_time': [], 'test_acc_record': [], 'communication_time': []}

for r in range(config.R):
    logger.info('====================================>')
    logger.info('==> Round {:} Start'.format(r))

    s_time = time.time()
    state = server.train(thread_number=config.K, client_ips=config.CLIENTS_LIST)
    server.aggregate(config.CLIENTS_LIST)
    e_time = time.time()

    # Recording each round training time and test accuracy
    training_time = e_time - s_time
    res['training_time'].append(training_time)
    comp_time = 0
    for key in state:
        comp_time += state[key]
    comp_time /= 4
    res['communication_time'].append(training_time - comp_time)
    test_acc = server.test(r)
    res['test_acc_record'].append(test_acc)

    with open('../results/FedAdapt_res.pkl', 'wb') as f:
        pickle.dump(res, f)

    logger.info('Round Finish')
    logger.info('==> Round Training Computation Time: {:}'.format(comp_time))
    logger.info('==> Round Training Communication Time: {:}'.format(training_time - comp_time))

    logger.info('==> Reinitialization for Round : {:}'.format(r + 1))

    if r > 49:
        LR = config.LR * 0.1

    server.reinitialize(first, LR)
    logger.info('==> Reinitialization Finish')
comm_time = server.finish(config.CLIENTS_LIST)
