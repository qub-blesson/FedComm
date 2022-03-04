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
        self.client = HelperClient(server=(host, port))

        if index != config.K:
            request = Request()
            request.code = defines.Codes.GET.number
            request.type = defines.Types['NON']
            request.destination = (host, port)
            request.uri_path = self.path
            request.content_type = defines.Content_types["application/xml"]
            request.payload = 'GIVE DATA'
            response = self.client.send_request(request)
            print(response)
        else:
            server = CoAPServer(host, port)
            try:
                server.listen(100)
            except KeyboardInterrupt:
                print("Server shutting down")
                server.close()

class CoAPServer(CoAP):
    def __init__(self, host, port):
        CoAP.__init__(self, (host, port))