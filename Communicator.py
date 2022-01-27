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

    # UDP Functionality
    # TODO: Check if sending works
    def send_msg_udp(self, sock, address, msg):
        if msg == b'':
            sock.sendto(b'', address)
        else:
            if msg[0] == 'MSG_INITIAL_GLOBAL_WEIGHTS_SERVER_TO_CLIENT' or \
                    msg[0] == 'MSG_LOCAL_WEIGHTS_CLIENT_TO_SERVER':
                send_buffers = [param for param in msg[1].values()]
                for i in range(len(send_buffers)):
                    key = list(msg[1].items())[i][0]
                    if len(send_buffers[i].size()) > 0:
                        flat_params = torch.cat([tensor.view(-1) for tensor in send_buffers[i]])
                        msg_split = flat_params.split(self.chunk)
                        for message in msg_split:
                            message = [key, message.detatch().numpy()]
                            self.pickle_send_udp(message, address, sock)
                    else:
                        message = [key, send_buffers[i].detatch().numpy()]
                        self.pickle_send_udp(message, address, sock)
        sock.sendto(b"END", address)

    def pickle_send_udp(self, message, address, sock):
        msg_pickle = pickle.dumps(message)
        sock.sendto(struct.pack(">I", len(msg_pickle)), address)
        sock.sendto(msg_pickle, address)

    def recv_msg_udp(self, sock, expect_msg_type=None):
        buffer = bytearray()
        read_next = True
        msg = None
        try:
            while read_next:
                msg_len = struct.unpack(">I", sock.recv(100))[0]
                msg = sock.recvfrom(msg_len)[0]
                if msg == b"END":
                    break
                buffer.extend(msg)
        except Exception:
            pass

        buffer = pickle.loads(buffer)
        #if expect_msg_type is not None:
            #if msg[0] == 'Finish':
                #return msg
            #elif msg[0] != expect_msg_type:
                #raise Exception("Expected " + expect_msg_type + " but received " + msg[0])

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
