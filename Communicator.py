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
        self.MAX_BUFFER_SIZE = 65536
        self.sock = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

    # UDP Functionality
    def send_msg_udp(self, sock, address, msg):
        if msg == b'':
            sock.sendto(b'', address)
        else:
            self.handle_weights(sock, address, msg)
        sock.sendto(b"END", address)

    def handle_weights(self, sock, address, msg):
        send_buffers = [param for param in msg[1].values()]
        for i in range(len(send_buffers)):
            key = list(msg[1].items())[i][0]
            if len(send_buffers[i].size()) > 0:
                flat_params = torch.cat([tensor.view(-1) for tensor in send_buffers[i]])
                msg_split = flat_params.split(self.chunk)
                for message in msg_split:
                    message = [key, message.detach().numpy()]
                    self.pickle_send_udp(message, address, sock)
            else:
                message = [key, send_buffers[i].detach().numpy()]
                self.pickle_send_udp(message, address, sock)

    def pickle_send_udp(self, message, address, sock):
        msg_pickle = pickle.dumps(message)
        sock.sendto(msg_pickle, address)

    def recv_msg_udp(self, sock, expect_msg_type=None):
        buffer = []
        read_next = True
        address = None
        count = 0

        try:
            while read_next:
                msg, address = sock.recvfrom(self.MAX_BUFFER_SIZE)
                if msg == b"END":
                    break
                try:
                    buffer.append(pickle.loads(msg))
                except:
                    continue

        except Exception:
            pass

        #if expect_msg_type is not None:
            #if msg[0] == 'Finish':
                #return msg
            #elif msg[0] != expect_msg_type:
                #raise Exception("Expected " + expect_msg_type + " but received " + msg[0])

        return buffer

    def recv_msg_udp_agg(self, sock, expect_msg_type=None):
        agg_dict = {'192.168.101.116': [], '192.168.101.217': [], '192.168.101.218': [], '192.168.101.219': []}
        read_next = True
        count = 0

        while read_next:
            try :
                msg, address = sock.recvfrom(4096)
            except:
                break
            if msg == b"END":
                count += 1
                sock.settimeout(2)
                if count == 4:
                    break
                continue
            try:
                agg_dict[address[0]].append(pickle.loads(msg))
            except:
                continue

        sock.settimeout(None)
        return agg_dict

    def init_recv_msg_udp(self, sock):
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
