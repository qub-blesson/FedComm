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
        self.chunk = 507
        self.sock = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.message_count = 0

    # UDP Functionality
    def send_msg_udp(self, sock, address, msg):
        messageSplit = pickle.dumps(msg)

        for i in range(0, len(messageSplit), self.chunk):
            sock.sendto(messageSplit[i:i + self.chunk], address)
        sock.sendto(b"END", address)

    def recv_msg_udp(self, sock, expect_msg_type=None):
        buffer = bytearray()
        read_next = True
        msg = None
        try:
            while read_next:
                msg, ip = sock.recvfrom(4096)
                if msg == b"END":
                    break
                buffer.extend(msg)
        except Exception:
            pass

        buffer = pickle.loads(buffer)
        if expect_msg_type is not None:
            if msg[0] == 'Finish':
                return msg
            elif msg[0] != expect_msg_type:
                raise Exception("Expected " + expect_msg_type + " but received " + msg[0])

        return buffer

    def init_recv_msg_udp(self, sock):
        buffer = bytearray()
        read_next = True
        ip = None
        try:
            while read_next:
                (msg, ip) = sock.recvfrom(4096)
                if msg == b"END":
                    break
        except Exception:
            pass

        return ip
