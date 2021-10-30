# Communicator Object

import pickle
import struct
import socket

import logging
import paho.mqtt.client as mqtt
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Communicator(object):
    def __init__(self, index, ip_address, host, port, topic='fedadapt', client_num=0):
        self.client_id = index
        self.ip = ip_address
        self.topic = topic
        self.client_num = client_num
        # create client
        self.client = mqtt.Client(str(self.client_id))
        # assign functionality
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message
        self.client.on_subscribe = self.on_subscribe
        # establish connection to host
        self.client.connect(host, port)
        # start communication
        self.client.loop_start()

    # TODO: modify send/recv msg functions to support mqtt

    # standard socket communication
    def send_msg(self, sock, msg):
        msg_pickle = pickle.dumps(msg)
        sock.sendall(struct.pack(">I", len(msg_pickle)))
        sock.sendall(msg_pickle)
        logger.debug(msg[0] + 'sent to' + str(sock.getpeername()[0]) + ':' + str(sock.getpeername()[1]))

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

    def on_disconnect(self, client, userdata, rc):
        logging.info("Disconnect result code: " + str(rc))

    def __del__(self):
        self.client.loop_stop()
        self.client.disconnect()

    def on_message(self, client, userdata, message):
        msg = str(message.payload.decode("utf-8"))
        self.client.publish("Message Received")

    def on_subscribe(self, client, userdata, mid, granted_qos):
        print("Subscribe message id: " + str(mid))
