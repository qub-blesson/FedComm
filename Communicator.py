# Communicator Object

import pickle
from queue import Queue

import logging
import zmq
import time

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Communicator(object):
    def __init__(self, index, ip_address, host, port, pub_topic='fedadapt', sub_topic='fedadapt', client_num=0):
        self.q = Queue()
        self.index = index
        self.host = host
        self.port = port
        # set 0MQ context
        self.context = zmq.Context()
        # init pub socket
        self.pub_socket = None
        self.sub_socket = None
        self.client_to_server()
        self.server_to_client()

    # subscribe to server from clients
    def client_to_server(self):
        if self.index == config.K:
            self.pub_socket = self.context.socket(zmq.PUB)
            self.pub_socket.bind("tcp://*:%s" % self.port)
        else:
            self.sub_socket = self.context.socket(zmq.SUB)
            status = -1
            while status != 0:
                status = self.sub_socket.connect("tcp://" + self.host + ":" + str(self.port))
            self.sub_socket.subscribe(b'')

    # server subscribes to all clients
    def server_to_client(self):
        if self.index == config.K:
            self.sub_socket = self.context.socket(zmq.SUB)
            for i in config.CLIENTS_LIST:
                status = -1
                while status != 0:
                    status = self.sub_socket.connect("tcp://"+i+":"+str(self.port))
        else:
            self.pub_socket = self.context.socket(zmq.PUB)
            self.pub_socket.bind("tcp://*:%s" % self.port)

    def send_msg(self, msg):
        msg_pickle = pickle.dumps(msg)
        self.pub_socket.send(msg_pickle)

    # equivalent to recv_msg
    def recv_msg(self):
        # load message
        msg = self.sub_socket.recv()
        msg = pickle.loads(msg)
        return msg
