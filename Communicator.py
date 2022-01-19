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

from itertools import islice

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
        messages = torch.split(self.chunk)

        for message in messages:
            message = pickle.dumps(message)
            sock.sendto(message, address)
        sock.sendto(pickle.dumps("END"), address)

    def recv_msg_udp(self, sock, expect_msg_type=None):
        buffer = []
        read_next = True
        try:
            while read_next:
                msg, ip = sock.recvfrom(65536)
                logger.info(msg)
                msg = pickle.loads(msg)
                if msg == "END":
                    break
                buffer.append(msg)
        except:
            socket.error

        if expect_msg_type is not None:
            if msg[0] == 'Finish':
                return msg
            elif msg[0] != expect_msg_type:
                raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
        logger.info(buffer)

        return buffer
