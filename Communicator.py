# Communicator Object

import pickle
import struct
import socket

import logging
import sys
import numpy as np
import torch

import pika
import threading
from queue import Queue
import paho.mqtt.client as mqtt

from itertools import islice

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
end_msg = False


class Communicator(object):
    def __init__(self, index, ip_address, host='0.0.0.0', port=1883, pub_topic='fedadapt', sub_topic='fedadapt',
                 client_num=0, user="server", password="password"):
        # all types
        self.ip = ip_address
        self.client = None
        self.chunk = 500
        self.MAX_BUFFER_SIZE = 65536
        self.sock = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.packets_sent = 0
        self.packets_received = 0
        self.tcp_sock = socket.socket()
        self.ttpi = {}

    # UDP Functionality
    def send_msg_udp(self, sock, tcp_sock, address, msg):
        if msg == b'':
            sock.sendto(b'', address)
            self.packets_sent += 1
            sock.sendto(b"END", address)
        else:
            self.handle_weights(sock, address, msg)
            tcp_sock.sendall(struct.pack(">I", len(b'END')))
            tcp_sock.sendall(b'END')
            self.packets_sent += 2

    def recv_end(self, socks):
        global end_msg
        read_next = True
        while read_next:
            for s in socks:
                msg_len = struct.unpack(">I", socks[s].recv(4))[0]
                msg = socks[s].recv(msg_len, socket.MSG_WAITALL)
                if msg != b'end':
                    self.ttpi[msg[1]] = np.array(msg[2:])
            end_msg = True

    def send_msg_tcp_client(self, sock, msg):
        msg_pickle = pickle.dumps(msg)
        sock.sendall(struct.pack(">I", len(msg_pickle)))
        sock.sendall(msg_pickle)

    def recv_msg_tcp(self, sock, expect_msg_type=None):
        msg_len = struct.unpack(">I", sock.recv(4))[0]
        msg = sock.recv(msg_len, socket.MSG_WAITALL)
        msg = pickle.loads(msg)
        logger.debug(msg[0] + 'received from' + str(sock.getpeername()[0]) + ':' + str(sock.getpeername()[1]))

        if expect_msg_type is not None:
            if msg[0] == 'Finish':
                return msg
            elif msg[0] != expect_msg_type:
                raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
        return msg

    def handle_weights(self, sock, address, msg):
        send_buffers = [param for param in msg[1].values()]
        for i in range(len(send_buffers)):
            key = list(msg[1].items())[i][0]
            if len(send_buffers[i].size()) > 0:
                flat_params = torch.cat([tensor.view(-1) for tensor in send_buffers[i]])
                msg_split = flat_params.split(self.chunk)
                self.packets_sent += len(msg_split)
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

        try:
            while read_next:
                msg, address = sock.recvfrom(self.MAX_BUFFER_SIZE)
                sock.settimeout(2)
                try:
                    buffer.append(pickle.loads(msg))
                except:
                    continue

        except Exception:
            pass

        # if expect_msg_type is not None:
        # if msg[0] == 'Finish':
        # return msg
        # elif msg[0] != expect_msg_type:
        # raise Exception("Expected " + expect_msg_type + " but received " + msg[0])

        sock.settimeout(None)
        return buffer

    def recv_msg_udp_agg(self, sock, expect_msg_type=None):
        agg_dict = {'192.168.101.116': [], '192.168.101.217': [], '192.168.101.218': [], '192.168.101.219': []}
        read_next = True
        global end_msg

        while read_next:
            try:
                msg, address = sock.recvfrom(4096)
                sock.settimeout(5)
            except:
                pass
            if msg is not None:
                agg_dict[address[0]].append(pickle.loads(msg))
            if end_msg:
                end_msg = False
                break

        for key in agg_dict:
            self.packets_received += len(agg_dict[key])
        sock.settimeout(None)
        return agg_dict

    def init_recv_msg_udp(self, sock):
        read_next = True
        ip = None
        try:
            while read_next:
                (msg, ip) = sock.recvfrom(4096)
                self.packets_received += 1
                if msg == b"END":
                    break
        except Exception:
            pass

        return ip
