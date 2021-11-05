# Communicator Object

import pickle
from queue import Queue

import logging
import paho.mqtt.client as mqtt

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Communicator(object):
    def __init__(self, index, ip_address, host, port, pub_topic='fedadapt', sub_topic='fedadapt', client_num=0):
        self.client_id = index
        self.ip = ip_address
        self.sub_topic = sub_topic
        self.pub_topic = pub_topic
        self.client_num = client_num
        # create client
        self.client = mqtt.Client(str(self.client_id))
        # create message queue
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

    def send_msg(self, msg):
        msg_pickle = pickle.dumps(msg)
        if self.client_id == config.K:
            # server
            self.client.publish(self.pub_topic, msg_pickle)
        else:
            # client
            self.client.publish(self.pub_topic, msg_pickle)

    # MQTT functionality below
    def on_connect(self, client, userdata, flags, rc):
        logger.info('Connecting to MQTT Server.')
        self.client.subscribe(self.sub_topic)
        if self.client_id != config.K:
            self.send_msg("1")

    def on_disconnect(self, client, userdata, rc):
        logging.info("Client %d Disconnect result code: " + str(rc), self.client_id)

    def __del__(self):
        self.client.loop_stop()
        self.client.disconnect()

    # equivalent to recv_msg
    def on_message(self, client, userdata, message):
        # load message and put into queue
        msg = pickle.loads(message.payload)
        self.q.put(msg)

    def on_subscribe(self, client, userdata, mid, granted_qos):
        print("Subscribe message id: " + str(mid))
