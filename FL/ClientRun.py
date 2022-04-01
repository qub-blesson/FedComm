import torch
import socket
import time
import multiprocessing
import os
import argparse
from Client import Client
import Config
import Utils
import logging
import sys

# set logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# add all files to path
sys.path.append('../')
# set arg parse
parser = argparse.ArgumentParser()
parser.add_argument('--communicator', help='Communication protocol', default='TCP')
parser.add_argument('--model', help='Model type', default='VGG8')
parser.add_argument('--stress', help='Tool used to limit network or apply stress: cpu, net', default=None)
parser.add_argument('--limiter', help='Tool used to limit network or apply stress: 3G, 4G, Wi-Fi', default=None)
parser.add_argument('--rounds', help='Number of training rounds', type=int, default=5)
args = parser.parse_args()
# set parameters based on input
stress = args.stress
limiter = args.limiter

Config.R = args.rounds

if args.model != '':
    Config.model_name = args.model
Config.COMM = args.communicator

if Config.model_name == 'VGG5':
    Config.split_layer = [6] * Config.K
    Config.model_len = 7

# set init values to send to client
ip_address = Config.HOST2IP[socket.gethostname()]
index = Config.CLIENTS_CONFIG[ip_address]
datalen = Config.N / Config.K
split_layer = Config.split_layer[index]
LR = Config.LR

# apply stress
if stress is not None:
    if stress == 'cpu':# or int(ip_address[:-1]) % 2 == 0:
        os.system('sudo test')
        os.system(Utils.tools[stress])
    else:
        #host =
        # TODO: Fix netstress for my project - rethink my options
        os.system('netstress -m host %s &', )

# apply network limit
if limiter is not None:
    os.system('sudo tc qdisc del dev ens160 root')
    os.system(Utils.tools[limiter])

logger.info('Preparing Client')
# call client - sets up client for FL process
client = Client(index, ip_address, Config.SERVER_ADDR, Config.SERVER_PORT, datalen)

# initialise client
first = True  # First initialization control
client.initialize(split_layer, first, LR)
first = False

# get number of cores
logger.info('Preparing Data.')
cpu_count = multiprocessing.cpu_count()
trainloader, classes = Utils.get_local_dataloader(index, cpu_count)

logger.info('FL Training')

# start training process
for r in range(Config.R):
    # logs for user
    logger.info('====================================>')
    logger.info('ROUND: {} START'.format(r))

    client.train(trainloader)
    logger.info('ROUND: {} END'.format(r))

    logger.info('==> Waiting for aggregration')
    client.upload()

    logger.info('==> Reinitialization for Round : {:}'.format(r + 1))
    s_time_rebuild = time.time()

    # decrease learning rate from round 50 TODO: Possibly remove
    if r > 49:
        LR = Config.LR * 0.1

    # reinitialise for next round
    client.reinitialize(Config.split_layer[index], first, LR)
    e_time_rebuild = time.time()
    logger.info('Rebuild time: ' + str(e_time_rebuild - s_time_rebuild))
    logger.info('==> Reinitialization Finish')
# finish client
client.finish()
