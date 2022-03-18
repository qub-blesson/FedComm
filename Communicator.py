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
        self.client_id = index
        self.ip = ip_address
        self.sub_topic = sub_topic
        self.pub_topic = pub_topic
        # set 0MQ context
        self.context = zmq.Context()
        # open publish sockets for all devices
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.bind("tcp://*:%s" % port)
        self.sub_socket = self.context.socket(zmq.SUB)
        time.sleep(5)
        # subscribe to server from clients
        if index != config.K:
            self.sub_socket.connect("tcp://"+host+":"+str(port))
            self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, sub_topic)
        else:
            # server subscribes to all clients
            for i in config.CLIENTS_LIST:
                self.sub_socket.connect("tcp://"+i+":"+str(port))
                self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, sub_topic)

    def send_msg(self, msg):
        msg_pickle = pickle.dumps(msg)
        self.pub_socket.send(self.pub_topic, msg_pickle)

    def on_disconnect(self, client, userdata, rc):
        logging.info("Client %d Disconnect result code: " + str(rc), self.client_id)

    # equivalent to recv_msg
    def on_message(self, client, userdata, message):
        # load message and put into queue
        msg = pickle.loads(message.payload)
        self.q.put(msg)

    def on_subscribe(self, client, userdata, mid, granted_qos):
        print("Subscribe message id: " + str(mid))
