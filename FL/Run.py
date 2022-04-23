import argparse
import socket
import sys
import logging

# add all files to path
sys.path.append('../')
import Config
from FL.ClientRun import ClientRun
from FL.ServerRun import ServerRun
import Utils

# set logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    # set arg parse
    parser = argparse.ArgumentParser()
    parser.add_argument('--communicator', help='Select communication protocol: TCP, UDP, ZMTP, AMQP, MQTT',
                        default='TCP')
    parser.add_argument('--model', help='Model type', default='VGG8')
    parser.add_argument('--stress', help='Tool used to limit network or apply stress: CPU, NET, ALL', default=None)
    parser.add_argument('--limiter', help='Tool used to limit network or apply stress: 3G, 4G, Wi-Fi', default=None)
    parser.add_argument('--rounds', help='Number of training rounds', type=int, default=5)
    parser.add_argument('--monitor', help='Monitors packet loss rate: True', default=None)
    parser.add_argument('--target', help='Server', default='127.0.0.1')
    args = parser.parse_args()
    # set parameters based on input
    stress = args.stress
    limiter = args.limiter
    monitor = args.monitor
    Config.R = args.rounds
    model = args.model
    communicator = args.communicator.upper()
    target = args.target

    if stress is not None:
        stress = stress.upper()
    if limiter is not None:
        limiter = limiter.upper()

    # Error handling
    if Config.R < 1:
        logger.error("Number of rounds not valid, setting to 5")
        Config.R = 5

    if communicator not in Utils.available_communicators:
        logger.error("Communicator does not exist, setting to TCP")
        communicator = 'TCP'

    if stress not in Utils.available_stress:
        logger.error("Stressor does not exist, not applying stress")
        stress = None

    if limiter not in Utils.available_limiter:
        logger.error("Limiter does not exist, not applying limiter")
        limiter = None

    if model not in Utils.available_models:
        logger.error("Model type does not exist, using VGG8 model")
        model = 'VGG8'

    return target, communicator, model, stress, limiter, monitor


if __name__ == '__main__':
    target, communicator, model, stress, limiter, monitor = parse_args()
    if target == socket.gethostbyname(socket.gethostname()) or socket.gethostname() not in Config.HOST2IP:
        if stress is None:
            stress = ''
        if limiter is None:
            limiter = ''
        ServerRun(communicator, model, stress, limiter)
    else:
        ClientRun(communicator, model, stress, limiter, monitor)
