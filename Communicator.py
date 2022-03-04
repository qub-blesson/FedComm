# Communicator Object

import pickle
from queue import Queue

import logging
from coapthon.client.helperclient import HelperClient
from coapthon.messages.request import Request
from coapthon import defines
from coapthon.resources.resource import Resource
from coapthon.server.coap import CoAP

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
            server = CoAPServer(host, port)
            try:
                server.listen(100)
            except KeyboardInterrupt:
                print("Server shutting down")
                server.close()

    def send_msg(self, payload):
        payload = pickle.dumps(payload)
        response = self.client.get('test')
        #request = Request()
        #request.code = defines.Codes.GET.number
        #request.type = defines.Types['CON']
        #request.destination = (self.host, self.port)
        #request.uri_path = self.path
        #request.content_type = defines.Content_types["application/xml"]
        #request.payload = payload
        #response = self.client.send_request(request)
        #print(response)

class CoAPServer(CoAP):
    def __init__(self, host, port):
        CoAP.__init__(self, (host, port))