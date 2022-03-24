import torch
import socket
import time
import multiprocessing
import os
import argparse
from Client import Client
import config
import utils
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
sys.path.append('../')
parser = argparse.ArgumentParser()
parser.add_argument('--communicator', help='Communication protocol', default='TCP')
parser.add_argument('--model', help='Model type', default='VGG8')
args = parser.parse_args()
if args.model is not None:
    config.model_name = args.model
config.COMM = args.communicator

if config.model_name == 'VGG5':
    config.split_layer = [6, 6, 6, 6]
    config.model_len = 7

ip_address = config.HOST2IP[socket.gethostname()]
index = config.CLIENTS_CONFIG[ip_address]
datalen = config.N / config.K
split_layer = config.split_layer[index]
LR = config.LR

logger.info('Preparing Client')
client = Client(index, ip_address, config.SERVER_ADDR, config.SERVER_PORT, datalen, split_layer)

first = True  # First initialization control
client.initialize(split_layer, first, LR)
first = False

logger.info('Preparing Data.')
cpu_count = multiprocessing.cpu_count()
trainloader, classes = utils.get_local_dataloader(index, cpu_count)

logger.info('Classic FL Training')

for r in range(config.R):
    logger.info('====================================>')
    logger.info('ROUND: {} START'.format(r))

    client.train(trainloader)
    logger.info('ROUND: {} END'.format(r))

    logger.info('==> Waiting for aggregration')
    client.upload()

    logger.info('==> Reinitialization for Round : {:}'.format(r + 1))
    s_time_rebuild = time.time()

    if r > 49:
        LR = config.LR * 0.1

    client.reinitialize(config.split_layer[index], first, LR)
    e_time_rebuild = time.time()
    logger.info('Rebuild time: ' + str(e_time_rebuild - s_time_rebuild))
    logger.info('==> Reinitialization Finish')
client.finish()
