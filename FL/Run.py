import argparse
import socket
import sys

# add all files to path
sys.path.append('../')
import Config
from FL.ClientRun import ClientRun
from FL.ServerRun import ServerRun


def parse_args():
    # set arg parse
    parser = argparse.ArgumentParser()
    parser.add_argument('--communicator', help='Communication protocol', default='TCP')
    parser.add_argument('--model', help='Model type', default='VGG8')
    parser.add_argument('--stress', help='Tool used to limit network or apply stress: cpu, net', default=None)
    parser.add_argument('--limiter', help='Tool used to limit network or apply stress: 3G, 4G, Wi-Fi', default=None)
    parser.add_argument('--rounds', help='Number of training rounds', type=int, default=5)
    parser.add_argument('--monitor', help='Monitors packet loss rate', default=None)
    parser.add_argument('--target', help='Server', default='127.0.0.1')
    args = parser.parse_args()
    # set parameters based on input
    stress = args.stress
    limiter = args.limiter
    monitor = args.monitor
    Config.R = args.rounds
    model = args.model
    communicator = args.communicator
    target = args.target

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
