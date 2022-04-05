# Communicator Object

import pickle
import struct
import socket
import time
import logging
import pika
import threading
from queue import Queue
import paho.mqtt.client as mqtt
import zmq
import torch

import Config

# set log level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Communicator between client/server
class Communicator(object):
    def __init__(self, index, ip_address, host='0.0.0.0', port=1883, pub_topic='fedadapt', sub_topic='fedadapt',
                 client_num=0, user="server", password="password"):
        """

        :param index: Unique ID
        :param ip_address: IP address of calling client/server
        :param host: Host server IP
        :param port: Server port number
        :param pub_topic: publish topic for MQTT, AMQP, ZMTP
        :param sub_topic: subscribe topic for MQTT, AMQP, ZMTP
        :param client_num: Client number
        :param user: Username for AMQP
        :param password: Password for AMQP
        """
        # all types
        self.ip = ip_address
        self.client = None
        self.q = Queue()
        self.host = host
        self.port = port
        self.client_id = index
        self.index = None

        # TCP specific setup
        if Config.COMM == 'TCP':
            self.sock = socket.socket()
        # MQTT specific setup
        elif Config.COMM == 'MQTT':
            # set topics and client number
            self.sub_topic = sub_topic
            self.pub_topic = pub_topic
            self.client_num = client_num
            # create client
            self.client = mqtt.Client(str(self.client_id))
            # assign functionality
            self.client.on_connect = self.on_connect
            self.client.on_disconnect = self.on_disconnect
            self.client.on_message = self.on_message_MQTT
            self.client.on_subscribe = self.on_subscribe
            # establish connection to host
            self.client.connect(host, port)
            # start communication
            self.client.loop_start()
        # AMQP specific setup
        elif Config.COMM == 'AMQP':
            self.sub_channel = None
            self.user = user
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
            # Create receive message thread
            self.thread = threading.Thread(target=self.recv_msg_amqp)
            self.thread.start()
        # ZMTP specific setup
        elif Config.COMM == '0MQ':
            # create ZMTP context
            self.context = zmq.Context()
            self.pub_socket = None
            self.sub_socket = None
            # Connect clients/server
            self.client_to_server()
            self.server_to_client()
            # Create and start receive thread
            self.thread = threading.Thread(target=self.recv_msg_0mq)
            self.thread.start()
        elif Config.COMM == 'UDP':
            self.chunk = 512
            self.MAX_BUFFER_SIZE = 65536
            self.sock = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
            self.packets_sent = 0
            self.packets_received = 0
            self.tcp_sock = socket.socket()
            self.udp_ttpi = {}

    """ TCP Functionality """
    def snd_msg_tcp(self, sock, msg):
        """
        :param sock: TCP Socket of client
        :param msg: Message to send to client
        """
        msg_pickle = pickle.dumps(msg)
        sock.sendall(struct.pack(">I", len(msg_pickle)))
        sock.sendall(msg_pickle)
        logger.debug(msg[0] + 'sent to' + str(sock.getpeername()[0]) + ':' + str(sock.getpeername()[1]))

    def recv_msg(self, sock, expect_msg_type=None):
        """
        Recieve message via TCP socket

        :param sock: Client socket
        :param expect_msg_type: expected message ID / type
        :return: message received
        """
        msg_len = struct.unpack(">I", sock.recv(4))[0]
        msg = sock.recv(msg_len, socket.MSG_WAITALL)
        msg = pickle.loads(msg)
        logger.debug(msg[0] + 'received from' + str(sock.getpeername()[0]) + ':' + str(sock.getpeername()[1]))

        # error handling
        if expect_msg_type is not None:
            if msg[0] == 'Finish':
                return msg
            elif msg[0] != expect_msg_type:
                raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
        return msg

    """ General functionality """
    def send_msg(self, msg):
        """
        Send model params via MQTT, AMQP or ZMTP

        :param msg: Message to send (model)
        """
        msg_pickle = pickle.dumps(msg)
        if Config.COMM == 'MQTT':
            self.client.publish(self.pub_topic, msg_pickle)
        elif Config.COMM == 'AMQP':
            self.pub_channel.basic_publish(exchange=self.pub_topic, routing_key='', body=msg_pickle)
        elif Config.COMM == '0MQ':
            self.pub_socket.send(msg_pickle)

    """ MQTT functionality """
    def on_connect(self, client, userdata, flags, rc):
        """
        Logs that device has connected to MQTT broker

        :param client: Required, not needed
        :param userdata: Required, not needed
        :param flags: Required, not needed
        :param rc: Required, not needed
        """
        logger.info('Connecting to MQTT Server.')
        self.client.subscribe(self.sub_topic)
        # Send test message
        if self.client_id != Config.K:
            self.send_msg("1")

    def on_disconnect(self, client, userdata, rc):
        """
        Log that user has disconnected

        :param client: Required, not needed
        :param userdata: Required, not needed
        :param rc: Return code
        """
        logging.info("Client %d Disconnect result code: " + str(rc), self.client_id)

    def __del__(self):
        if self.client is not None:
            self.client.loop_stop()
            self.client.disconnect()

    # equivalent to recv_msg
    def on_message_MQTT(self, client, userdata, message):
        """
        Automatic handling of messages sent through the broker

        :param client: Required, not needed
        :param userdata: Required, not needed
        :param message: Message received
        """
        # load message and put into queue
        msg = pickle.loads(message.payload)
        self.q.put(msg)

    def on_subscribe(self, client, userdata, mid, granted_qos):
        """

        :param client: Required, not needed
        :param userdata: Required, not needed
        :param mid: Message ID
        :param granted_qos: Required, not needed
        """
        print("Subscribe message id: " + str(mid))

    """ AMQP functionality """
    def recv_msg_amqp(self):
        """
        Receive message threaded loop
        """
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
        # bind queue and start listening
        self.sub_channel.queue_bind(exchange=self.sub_topic, queue=q)
        self.sub_channel.basic_consume(queue=q, on_message_callback=self.on_message_amqp, auto_ack=True)
        self.sub_channel.start_consuming()

    def on_message_amqp(self, ch, method, properties, body):
        """
        Handling of received packets via AMQP

        :param ch: Required, not needed
        :param method: Required, not needed
        :param properties: Required, not needed
        :param body: message content
        """
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
        """
        Clean up AMQP, stops listening for messages
        """
        self.sub_channel.stop_consuming()

    """ 0MQ Functionality """
    def client_to_server(self):
        """
        subscribe to server from clients
        """
        if self.client_id == Config.K:
            self.pub_socket = self.context.socket(zmq.PUB)
            self.pub_socket.bind("tcp://*:%s" % self.port)
            time.sleep(30)
        else:
            self.sub_socket = self.context.socket(zmq.SUB)
            self.sub_socket.connect("tcp://" + self.host + ":" + str(self.port))
            self.sub_socket.subscribe(b'')

    def server_to_client(self):
        """
        server subscribes to all clients
        """
        if self.client_id == Config.K:
            self.sub_socket = self.context.socket(zmq.SUB)
            for i in Config.CLIENTS_LIST:
                self.sub_socket.connect("tcp://" + i + ":" + str(self.port))
                self.sub_socket.subscribe(b'')
        else:
            self.pub_socket = self.context.socket(zmq.PUB)
            self.pub_socket.bind("tcp://*:%s" % self.port)

    def recv_msg_0mq(self):
        """
        Receive and unpickle message sent via ZeroMQ (ZMTP)
        """
        # load message
        while True:
            self.q.put(pickle.loads(self.sub_socket.recv()))

    """ UDP Functionality """
    def send_msg_udp(self, sock, tcp_sock, address, msg):
        if msg == b'':
            sock.sendto(b'', address)
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
                if msg != b'END':
                    msg = pickle.loads(msg)
                    self.udp_ttpi[msg[1]] = np.array(msg[2:])
            end_msg = True

    def send_msg_tcp_client(self, sock, msg):
        msg_pickle = pickle.dumps(msg)
        sock.sendall(struct.pack(">I", len(msg_pickle)))
        sock.sendall(msg_pickle)

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

        sock.settimeout(None)
        return buffer

    def recv_msg_udp_agg(self, sock, expect_msg_type=None):
        agg_dict = {'192.168.101.116': [], '192.168.101.217': [], '192.168.101.218': [], '192.168.101.219': []}
        read_next = True
        global end_msg

        while read_next:
            try:
                msg, address = sock.recvfrom(4096)
                sock.settimeout(2)
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
                if msg == b"END":
                    break
        except Exception:
            pass

        return ip