# Communicator Object

import pickle
import struct
import socket
from queue import Queue

import logging
import paho.mqtt.client as mqtt
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Communicator(object):
    connections = 0
    def __init__(self, index, ip_address, host, port, pub_topic='fedadapt', sub_topic='fedadapt', client_num=0):
        self.client_id = index
        self.ip = ip_address
        self.sub_topic = sub_topic
        self.pub_topic = pub_topic
        self.client_num = client_num
        # create client
        self.client = mqtt.Client(str(self.client_id))
        self.q = Queue()
        # assign functionality
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message
        self.client.on_subscribe = self.on_subscribe
        # establish connection to host
        self.client.connect(host, port)
        # start communication
        self.client.loop_start()

    # standard socket communication
    def send_msg(self, msg):
        if self.client_id == 0:
            # server
            # logging.info("topic = %s" % str(self.pub_topic))
            self.client.publish(self.pub_topic, msg)
            logging.info("sent")
        else:
            # client
            self.client.publish(self.pub_topic, msg)

    def recv_msg(self, sock, expect_msg_type=None):
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

    # MQTT functionality below
    def on_connect(self, client, userdata, flags, rc):
        logging.info("Connected flags" + str(flags) + "result code " + str(rc))
        self.client.subscribe(self.sub_topic)

    def on_disconnect(self, client, userdata, rc):
        logging.info("Disconnect result code: " + str(rc))

    def __del__(self):
        self.client.loop_stop()
        self.client.disconnect()

    # equivalent to recv_msg
    def on_message(self, client, userdata, message):
        msg = str(message.payload.decode("utf-8"))
        self.client.publish("Message Received")
        self.q.put(message)

    def on_subscribe(self, client, userdata, mid, granted_qos):
        print("Subscribe message id: " + str(mid))
