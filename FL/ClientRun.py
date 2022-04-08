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

# global variables
stress = None
limiter = None
monitor = None
model = None
communicator = None
index = None
ip_address = None
split_layer = None
LR = None
datalen = None
trainloader = None
client = None
first = None


def parse_args():
    global stress, limiter, monitor, model, communicator
    # set arg parse
    parser = argparse.ArgumentParser()
    parser.add_argument('--communicator', help='Communication protocol', default='TCP')
    parser.add_argument('--model', help='Model type', default='VGG8')
    parser.add_argument('--stress', help='Tool used to limit network or apply stress: cpu, net', default=None)
    parser.add_argument('--limiter', help='Tool used to limit network or apply stress: 3G, 4G, Wi-Fi', default=None)
    parser.add_argument('--rounds', help='Number of training rounds', type=int, default=5)
    parser.add_argument('--monitor', help='Monitors packet loss rate', default=None)
    args = parser.parse_args()
    # set parameters based on input
    stress = args.stress
    limiter = args.limiter
    monitor = args.monitor
    Config.R = args.rounds
    model = args.model
    communicator = args.communicator


def set_args():
    if model != '':
        Config.model_name = model
    Config.COMM = communicator

    if Config.model_name == 'VGG5':
        Config.split_layer = [6] * Config.K
        Config.model_len = 7


def init_client_values():
    global ip_address, index, datalen, split_layer, LR
    # set init values to send to client
    ip_address = Config.HOST2IP[socket.gethostname()]
    index = Config.CLIENTS_CONFIG[ip_address]
    datalen = Config.N / Config.K
    split_layer = Config.split_layer[index]
    LR = Config.LR


def apply_stress():
    # apply stress
    if stress is not None:
        if stress == 'cpu':  # or int(ip_address[:-1]) % 2 == 0:
            os.system('sudo test')
            os.system(Utils.tools[stress])
        else:
            # host =
            # TODO: Fix netstress for my project - rethink my options
            os.system('netstress -m host %s &', )


def monitor_network():
    if monitor is not None:
        os.system('sudo test')
        os.system('sudo tshark -q -z io,stat,30,"COUNT(tcp.analysis.retransmission) tcp.analysis.retransmission" &')


def limit_network_bandwidth():
    # apply network limit
    if limiter is not None:
        os.system('sudo tc qdisc del dev ens160 root')
        os.system(Utils.tools[limiter])


def init_client():
    global trainloader, client, first
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


def start_FL_process():
    if client is None:
        pass
    global LR
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

        # decrease learning rate from round 50
        if r > 49:
            LR = Config.LR * 0.1

        # reinitialise for next round
        client.reinitialize(Config.split_layer[index], first, LR)
        e_time_rebuild = time.time()
        logger.info('Rebuild time: ' + str(e_time_rebuild - s_time_rebuild))
        logger.info('==> Reinitialization Finish')


logger.info('FL Training')


def finish_client():
    # finish client
    client.finish()
    if monitor is not None:
        os.system('sudo pkill tshark')
    if Config.COMM == 'UDP':
        while True:
            pass


if __name__ == '__main__':
    parse_args()
    set_args()
    init_client_values()
    apply_stress()
    monitor_network()
    limit_network_bandwidth()
    init_client()
    start_FL_process()
    finish_client()
