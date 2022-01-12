import time
import torch
import pickle
import argparse

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys

sys.path.append('../')
from Server import Server
import config
import utils
import PPO

parser = argparse.ArgumentParser()
parser.add_argument('--offload', help='FedAdapt or classic FL mode', type=utils.str2bool, default=False)
parser.add_argument('--communicator', help='Communication protocol', default='TCP')
args = parser.parse_args()

LR = config.LR
offload = args.offload
communicator = args.communicator
config.COMM = communicator
first = True  # First initializaiton control

logger.info('Preparing Sever.')

if communicator == 'TCP':
    server = Server(0, config.SERVER_ADDR, config.SERVER_PORT, 'VGG8')
elif communicator == 'MQTT' or communicator == 'AMQP':
    server = Server(config.K, config.SERVER_ADDR, config.SERVER_PORT, 'VGG8')
server.initialize(config.split_layer, offload, first, LR)
first = False

state_dim = 2 * config.G
action_dim = config.G

if offload:
    # Initialize trained RL agent
    agent = PPO.PPO(state_dim, action_dim, config.action_std, config.rl_lr, config.rl_betas, config.rl_gamma,
                    config.K_epochs, config.eps_clip)
    agent.policy.load_state_dict(torch.load('./PPO_FedAdapt.pth'))

if offload:
    logger.info('FedAdapt Training')
else:
    logger.info('Classic FL Training')

res = {}
res['trianing_time'], res['test_acc_record'], res['bandwidth_record'], res['communication_time'] = [], [], [], []

for r in range(config.R):
    logger.info('====================================>')
    logger.info('==> Round {:} Start'.format(r))

    s_time = time.time()
    state, bandwidth = server.train(thread_number=config.K, client_ips=config.CLIENTS_LIST)
    aggregrated_model = server.aggregate(config.CLIENTS_LIST)
    e_time = time.time()

    # Recording each round training time, bandwidth and test accuracy
    trianing_time = e_time - s_time
    res['trianing_time'].append(trianing_time)
    res['bandwidth_record'].append(bandwidth)

    test_acc = server.test(r)
    res['test_acc_record'].append(test_acc)

    with open('../results/FedAdapt_res.pkl', 'wb') as f:
        pickle.dump(res, f)

    logger.info('Round Finish')
    logger.info('==> Round Training Time: {:}'.format(trianing_time))

    logger.info('==> Reinitialization for Round : {:}'.format(r + 1))
    if offload:
        split_layers = server.adaptive_offload(agent, state)
    else:
        split_layers = config.split_layer

    if r > 49:
        LR = config.LR * 0.1

    server.reinitialize(split_layers, offload, first, LR)
    logger.info('==> Reinitialization Finish')
comm_time = server.finish(config.CLIENTS_LIST)
res['communication_time'].append(comm_time)
with open('../results/FedAdapt_res.pkl', 'wb') as f:
    pickle.dump(res, f)
