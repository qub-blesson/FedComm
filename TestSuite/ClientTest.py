import collections
import socketserver
import unittest
import sys

import gevent.server
import numpy as np
import torch
import torchvision
import paho

import Config
import Utils
import Vgg
from FL.Client import Client

sys.path.append('../')


class BasicServer(gevent.server.StreamServer):
    def handle(self, socket, address):
        print('test')


class ClientTest(unittest.TestCase):
    def test_build_optimize_model__model_received(self):
        Config.COMM = ''
        self.client = Client(0, '', Config.SERVER_ADDR, Config.SERVER_PORT, Config.N)
        self.split_layer = Config.split_layer
        self.assertIsNone(self.client.net)
        self.client.build_optimize_model(6, True, Config.LR)
        self.assertIsNotNone(self.client.net)

    def test_build_optimize_model__correct_model_received_VGG5(self):
        Config.COMM = ''
        Config.model_name = 'VGG5'
        Config.split_layer = [6] * Config.K
        Config.model_len = 7
        self.client = Client(0, '', Config.SERVER_ADDR, Config.SERVER_PORT, Config.N)
        self.client.build_optimize_model(6, True, Config.LR)
        expected = 25
        self.assertEqual(len(self.client.net.state_dict()), expected)

    def test_build_optimize_model__correct_model_received_VGG8(self):
        Config.COMM = ''
        Config.model_name = 'VGG8'
        Config.split_layer = [9] * Config.K
        Config.model_len = 10
        self.client = Client(0, '', Config.SERVER_ADDR, Config.SERVER_PORT, Config.N)
        self.client.build_optimize_model(9, True, Config.LR)
        expected = 46
        self.assertEqual(len(self.client.net.state_dict()), expected)

    def test_build_optimize_model__correct_model_type_VGG5(self):
        Config.COMM = ''
        Config.model_name = 'VGG5'
        Config.split_layer = [6] * Config.K
        Config.model_len = 7
        self.client = Client(0, '', Config.SERVER_ADDR, Config.SERVER_PORT, Config.N)
        self.client.build_optimize_model(6, True, Config.LR)
        self.assertEqual(type(self.client.net.state_dict()), collections.OrderedDict)

    def test_build_optimize_model__correct_model_type_VGG8(self):
        Config.COMM = ''
        Config.model_name = 'VGG8'
        Config.split_layer = [9] * Config.K
        Config.model_len = 10
        self.client = Client(0, '', Config.SERVER_ADDR, Config.SERVER_PORT, Config.N)
        self.client.build_optimize_model(9, True, Config.LR)
        self.assertEqual(type(self.client.net.state_dict()), collections.OrderedDict)

    def test_build_optimize_model__no_model_built_VGG5(self):
        Config.COMM = ''
        Config.model_name = 'VGG5'
        Config.split_layer = [6] * Config.K
        Config.model_len = 7
        self.client = Client(0, '', Config.SERVER_ADDR, Config.SERVER_PORT, Config.N)
        self.client.build_optimize_model(6, False, Config.LR)
        self.assertIsNone(self.client.net)

    def test_build_optimize_model__no_model_built_VGG8(self):
        Config.COMM = ''
        Config.model_name = 'VGG8'
        Config.split_layer = [9] * Config.K
        Config.model_len = 10
        self.client = Client(0, '', Config.SERVER_ADDR, Config.SERVER_PORT, Config.N)
        self.client.build_optimize_model(9, False, Config.LR)
        self.assertIsNone(self.client.net)

    def test_build_optimize_model__model_optimised_VGG8(self):
        Config.COMM = ''
        Config.model_name = 'VGG8'
        Config.split_layer = [9] * Config.K
        Config.model_len = 10
        self.client = Client(0, '', Config.SERVER_ADDR, Config.SERVER_PORT, Config.N)
        self.client.net = Utils.get_model('Client', Config.model_name, 6, 'cpu', Config.model_cfg)
        self.client.build_optimize_model(9, False, 0.02)
        self.assertEqual(type(self.client.optimizer), torch.optim.SGD)

    def test_build_optimize_model__model_optimised_VGG5(self):
        Config.COMM = ''
        Config.model_name = 'VGG5'
        Config.split_layer = [6] * Config.K
        Config.model_len = 7
        self.client = Client(0, '', Config.SERVER_ADDR, Config.SERVER_PORT, Config.N)
        self.client.net = Utils.get_model('Client', Config.model_name, 6, 'cpu', Config.model_cfg)
        self.client.build_optimize_model(6, False, Config.LR)
        self.assertEqual(type(self.client.optimizer), torch.optim.SGD)

    def test_build_optimize_model__LR_VGG8(self):
        Config.COMM = ''
        Config.model_name = 'VGG8'
        Config.split_layer = [9] * Config.K
        Config.model_len = 10
        self.client = Client(0, '', Config.SERVER_ADDR, Config.SERVER_PORT, Config.N)
        self.client.net = Utils.get_model('Client', Config.model_name, 6, 'cpu', Config.model_cfg)
        self.client.build_optimize_model(9, False, 0.02)
        self.assertEqual(self.client.optimizer.param_groups[0]['lr'], 0.02)

    def test_build_optimize_model__LR_VGG5(self):
        Config.COMM = ''
        Config.model_name = 'VGG5'
        Config.split_layer = [6] * Config.K
        Config.model_len = 7
        self.client = Client(0, '', Config.SERVER_ADDR, Config.SERVER_PORT, Config.N)
        self.client.net = Utils.get_model('Client', Config.model_name, 6, 'cpu', Config.model_cfg)
        self.client.build_optimize_model(6, False, Config.LR)
        self.assertEqual(type(self.client.optimizer), torch.optim.SGD)
        self.assertEqual(self.client.optimizer.param_groups[0]['lr'], Config.LR)

    def test_load_new_model__VGG5(self):
        Config.COMM = ''
        Config.model_name = 'VGG5'
        Config.split_layer = [6] * Config.K
        Config.model_len = 7
        self.client = Client(0, '', Config.SERVER_ADDR, Config.SERVER_PORT, Config.N)
        self.client.net = Utils.get_model('Client', Config.model_name, 6, 'cpu', Config.model_cfg)
        self.net = Utils.get_model('Client', Config.model_name, 6, 'cpu', Config.model_cfg)
        self.client.load_new_model(self.net.state_dict())
        self.assertIsNotNone(self.client.net)

    def test_load_new_model__VGG8(self):
        Config.COMM = ''
        Config.model_name = 'VGG8'
        Config.split_layer = [9] * Config.K
        Config.model_len = 10
        self.client = Client(0, '', Config.SERVER_ADDR, Config.SERVER_PORT, Config.N)
        self.client.net = Utils.get_model('Client', Config.model_name, 9, 'cpu', Config.model_cfg)
        self.net = Utils.get_model('Client', Config.model_name, 9, 'cpu', Config.model_cfg)
        self.client.load_new_model(self.net.state_dict())
        self.assertIsNotNone(self.client.net)

    def test_train_model__VGG5(self):
        Config.COMM = ''
        self.client = Client(0, '', Config.SERVER_ADDR, Config.SERVER_PORT, Config.N)
        trainloader, classes = Utils.get_local_dataloader(0, 1)
        Config.model_name = 'VGG5'
        Config.model_len = 7
        self.client.net = Utils.get_model('Client', Config.model_name, 6, 'cpu', Config.model_cfg)
        self.client.build_optimize_model(6, True, Config.LR)
        s, e = self.client.train_model(trainloader)
        expected = 5  # seconds
        self.assertGreaterEqual(e - s, expected)

    def test_train_model__Accuracy_VGG5(self):
        Config.COMM = ''
        self.client = Client(0, '', Config.SERVER_ADDR, Config.SERVER_PORT, Config.N)
        trainloader, classes = Utils.get_local_dataloader(0, 1)
        Config.model_name = 'VGG5'
        Config.model_len = 7
        self.client.net = Utils.get_model('Client', Config.model_name, 6, 'cpu', Config.model_cfg)
        self.client.build_optimize_model(6, True, Config.LR)
        acc_before_train = self.client.test()
        self.client.train_model(trainloader)
        acc_after_train = self.client.test()
        self.assertGreater(acc_after_train, acc_before_train)

    def test_train_model__VGG8(self):
        Config.COMM = ''
        self.client = Client(0, '', Config.SERVER_ADDR, Config.SERVER_PORT, Config.N)
        trainloader, classes = Utils.get_local_dataloader(0, 1)
        Config.model_name = 'VGG8'
        Config.model_len = 10
        self.client.net = Utils.get_model('Client', Config.model_name, 9, 'cpu', Config.model_cfg)
        self.client.build_optimize_model(9, True, Config.LR)
        s, e = self.client.train_model(trainloader)
        expected = 10  # seconds
        self.assertGreaterEqual(e - s, expected)

    def test_train_model__Accuracy_VGG8(self):
        Config.COMM = ''
        self.client = Client(0, '', Config.SERVER_ADDR, Config.SERVER_PORT, Config.N)
        trainloader, classes = Utils.get_local_dataloader(0, 1)
        Config.model_name = 'VGG8'
        Config.model_len = 10
        self.client.net = Utils.get_model('Client', Config.model_name, 9, 'cpu', Config.model_cfg)
        self.client.build_optimize_model(9, True, Config.LR)
        acc_before_train = self.client.test()
        self.client.train_model(trainloader)
        acc_after_train = self.client.test()
        self.assertGreater(acc_after_train, acc_before_train)

    """
    def test_send_training_time_to_server(self):
        Config.COMM = ''
        self.client = Client(0, '', Config.SERVER_ADDR, Config.SERVER_PORT, Config.N)
        server = BasicServer(('127.0.0.1', 0))
        server.start()
        self.client.sock = gevent.socket.create_connection(('127.0.0.1', server.server_port))
        response = self.client.sock.makefile().read()
        print(response)
        server.stop()
    """
