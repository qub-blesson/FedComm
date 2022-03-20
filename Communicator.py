# Communicator Object

import pickle
import struct
import socket

import logging
import pika
import threading
from queue import Queue
import paho.mqtt.client as mqtt

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Communicator(object):
    def __init__(self, index, ip_address, host='0.0.0.0', port=1883, pub_topic='fedadapt', sub_topic='fedadapt',
                 client_num=0, user="server", password="password"):
        # all types
        self.ip = ip_address
        self.client = None
        if config.COMM == 'TCP':
            self.sock = socket.socket()
        elif config.COMM == 'MQTT':
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
            self.client.on_message = self.on_message_MQTT
            self.client.on_subscribe = self.on_subscribe
            # establish connection to host
            self.client.connect(host, port)
            # start communication
            self.client.loop_start()
        elif config.COMM == 'AMQP':
            self.sub_channel = None
            self.client_id = index
            self.user = user
            self.host = host
            self.port = port
            # Create connection to AMQP server
            self.credentials = pika.PlainCredentials(user, password)
            self.connection = pika.BlockingConnection(pika.ConnectionParameters(
                host=host, port=port, credentials=self.credentials, heartbeat=600, blocked_connection_timeout=300))
            self.pub_topic = pub_topic
            self.sub_topic = sub_topic
            # Create pub channel
            self.pub_channel = self.connection.channel()
            # Create exchange
            self.pub_channel.exchange_declare(exchange=self.pub_topic, exchange_type='fanout')
            self.count = 0
            # Create Q
            self.q = Queue()
            self.thread = threading.Thread(target=self.recv_msg_amqp)
            self.thread.start()

    # TCP Functionality
    def snd_msg_tcp(self, sock, msg):
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
    def send_msg(self, msg):
        msg_pickle = pickle.dumps(msg)
        if config.COMM == 'MQTT':
            self.client.publish(self.pub_topic, msg_pickle)
        elif config.COMM == 'AMQP':
            self.pub_channel.basic_publish(exchange=self.pub_topic, routing_key='', body=msg_pickle)

    def on_connect(self, client, userdata, flags, rc):
        logger.info('Connecting to MQTT Server.')
        self.client.subscribe(self.sub_topic)
        if self.client_id != config.K:
            self.send_msg("1")

    def on_disconnect(self, client, userdata, rc):
        logging.info("Client %d Disconnect result code: " + str(rc), self.client_id)

    def __del__(self):
        if self.client is not None:
            self.client.loop_stop()
            self.client.disconnect()

    # equivalent to recv_msg
    def on_message_MQTT(self, client, userdata, message):
        # load message and put into queue
        msg = pickle.loads(message.payload)
        self.q.put(msg)

    def on_subscribe(self, client, userdata, mid, granted_qos):
        print("Subscribe message id: " + str(mid))

    # AMQP functionality below
    def recv_msg_amqp(self):
        # create new consumer connection
        credentials = pika.PlainCredentials(self.user, 'password')
        connection = pika.BlockingConnection(pika.ConnectionParameters(
            host=self.host, port=self.port, credentials=credentials))
        # create exchange
        self.sub_channel = connection.channel()
        self.sub_channel.exchange_declare(exchange=self.sub_topic, exchange_type='fanout')
        # Create queue for receiving messages
        res = self.sub_channel.queue_declare(queue='')
        q = res.method.queue
        self.sub_channel.queue_bind(exchange=self.sub_topic, queue=q)
        self.sub_channel.basic_consume(queue=q, on_message_callback=self.on_message_amqp, auto_ack=True)
        self.sub_channel.start_consuming()

    def on_message_amqp(self, ch, method, properties, body):
        # load message and put into queue
        msg = pickle.loads(body)
        self.q.put(msg)
        if msg[0] == 'DONE':
            self.clean()
        elif msg[0] == 'MSG_COMMUNICATION_TIME':
            self.count += 1
            if self.count == 4:
                self.clean()

    def clean(self):
        self.sub_channel.stop_consuming()
