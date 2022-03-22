# Communicator Object

import pickle
import threading
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
        self.ip = ip_address
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
        self.thread = threading.Thread(target=self.recv_msg)
        self.thread.start()

    # subscribe to server from clients
    def client_to_server(self):
        if self.index == config.K:
            self.pub_socket = self.context.socket(zmq.PUB)
            self.pub_socket.bind("tcp://*:%s" % self.port)
            time.sleep(30)
        else:
            self.sub_socket = self.context.socket(zmq.SUB)
            self.sub_socket.connect("tcp://" + self.host + ":" + str(self.port))
            self.sub_socket.subscribe(b'')

    # server subscribes to all clients
    def server_to_client(self):
        if self.index == config.K:
            self.sub_socket = self.context.socket(zmq.SUB)
            for i in config.CLIENTS_LIST:
                self.sub_socket.connect("tcp://"+i+":"+str(self.port))
                self.sub_socket.subscribe(b'')
        else:
            self.pub_socket = self.context.socket(zmq.PUB)
            self.pub_socket.bind("tcp://*:%s" % self.port)

    def send_msg(self, msg):
        msg_pickle = pickle.dumps(msg)
        self.pub_socket.send(msg_pickle)

    def recv_msg(self):
        # load message
        while True:
            self.q.put(pickle.loads(self.sub_socket.recv()))
