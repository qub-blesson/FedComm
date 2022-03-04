# Communicator Object

import pickle
from queue import Queue
import json

import logging
from coapthon.client.helperclient import HelperClient
from coapthon.messages.request import Request
from coapthon.messages.response import Response
from coapthon import defines
from coapthon.resources.resource import Resource
from coapthon.server.coap import CoAP

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

q = []

class Communicator(object):
    def __init__(self, index, ip_address, host, port, pub_topic='fedadapt', sub_topic='fedadapt', client_num=0):
        self.client_id = index
        self.ip = ip_address
        self.path = 'encoding'
        self.payload = 'text/plain'
        self.host = host
        self.port = port

        if index != config.K:
            self.client = HelperClient(server=(host, port))
        else:
            self.server = CoAPServer(host, port)

    def send_msg(self, payload):
        response = self.client.get('test/')
        print(response)

    def server_listen(self):
        try:
            self.server.listen(100)
        except KeyboardInterrupt:
            print("Server shutting down")
            self.server.close()


class CoAPServer(CoAP):
    def __init__(self, host, port):
        CoAP.__init__(self, (host, port))
        self.add_resource('test/', Test())


class Test(Resource):
    def __init__(self, name="Test", coap_server=None):
        super(Test, self).__init__(name, coap_server, visible=True, observable=True, allow_children=True)

    def render_GET(self, request):
        self.content_type = "text/plain"
        self.payload = [param.detach().numpy() for param in q[0]]

        return self
