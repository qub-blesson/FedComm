# Communicator Object

import pickle
import struct
import socket

import logging
import sys

import pika
import threading
from queue import Queue
import paho.mqtt.client as mqtt

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Communicator(object):
    def __init__(self, index, ip_address, host='0.0.0.0', port=1883, pub_topic='fedadapt', sub_topic='fedadapt',
                 client_num=0, user="server", password="password"):
        # all types
        self.ip = ip_address
        self.client = None
        self.sock = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

    # UDP Functionality
    def send_msg_udp(self, sock, address, msg):
        msg_pickle = pickle.dumps(msg)
        sys.getsizeof(msg_pickle)
        sock.sendto(msg_pickle, address)
        #logger.debug(msg[0] + 'sent to' + str(sock.getpeername()[0]) + ':' + str(sock.getpeername()[1]))

    def recv_msg_udp(self, sock, expect_msg_type=None):
        (msg, ip) = sock.recvfrom(65535)
        msg = pickle.loads(msg)
        #logger.debug(msg[0] + 'received from' + str(sock.getpeername()[0]) + ':' + str(sock.getpeername()[1]))

        if expect_msg_type is not None:
            if msg[0] == 'Finish':
                return msg
            elif msg[0] != expect_msg_type:
                raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
        return msg
