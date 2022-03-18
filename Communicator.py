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
        # set 0MQ context
        self.context = zmq.Context()
        # open publish sockets for all devices
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.bind("tcp://*:%s" % port)
        self.sub_socket = self.context.socket(zmq.SUB)
        time.sleep(15)
        # subscribe to server from clients
        if index != config.K:
            self.sub_socket.connect("tcp://"+host+":"+str(port))
        else:
            # server subscribes to all clients
            for i in config.CLIENTS_LIST:
                self.sub_socket.connect("tcp://"+i+":"+str(port))
        self.sub_socket.subscribe(b'')

    def send_msg(self, msg):
        msg_pickle = pickle.dumps(msg)
        self.pub_socket.send(msg_pickle)

    # equivalent to recv_msg
    def recv_msg(self):
        # load message
        msg = self.sub_socket.recv()
        msg = pickle.loads(msg)
        return msg
