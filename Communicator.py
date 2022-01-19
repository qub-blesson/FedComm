# Communicator Object

import pickle
import struct
import socket

import logging
import sys
import torch

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
        self.chunk = 500
        self.sock = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

    # UDP Functionality
    def send_msg_udp(self, sock, address, msg):
        msg_pickle = pickle.dumps(msg)
        #messages = [msg_pickle[i:i + self.chunk] for i in range(0, len(msg_pickle), self.chunk)]
        messages = torch.split(self.chunk)
        logger.info(sys.getsizeof(messages[0]))
        for message in messages:
            sock.sendto(message, address)
        sock.sendto(pickle.dumps("END"), address)

    def recv_msg_udp(self, sock, expect_msg_type=None):
        buffer = []
        read_next = True
        try:
            while read_next:
                msg, ip = sock.recvfrom(65535)
                msg = pickle.loads(msg)
                buffer.append(msg)
        except:
            socket.error

        if expect_msg_type is not None:
            if msg[0] == 'Finish':
                return msg
            elif msg[0] != expect_msg_type:
                raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
        return msg
