import unittest
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
from FL.Server import Server

sys.path.append('../')


class MyTestCase(unittest.TestCase):
    def test_get_model(self):
        Config.COMM = ''
        self.server = Server(0, '', '', test=True)
        self.assertIsNone(self.server.uninet)
        self.server.get_model()
        self.assertIsNotNone(self.server.uninet)

    def test_get_model__VGG5(self):
        Config.COMM = ''
        Config.model_name = 'VGG5'
        Config.model_len = 7
        self.server = Server(0, '', '', test=True)
        self.server.get_model()
        expected = 25
        self.assertEqual(len(self.server.uninet.state_dict()), expected)

    def test_get_model__VGG8(self):
        Config.COMM = ''
        Config.model_name = 'VGG8'
        Config.model_len = 10
        self.server = Server(0, '', '', test=True)
        self.server.get_model()
        expected = 46
        self.assertEqual(len(self.server.uninet.state_dict()), expected)

    def test_get_client_model__min_1(self):
        Config.COMM = ''
        self.server = Server(0, '', '', test=True)
        self.assertIsNone(self.server.nets)
        self.server.get_empty_client_model(True)
        self.assertGreaterEqual(len(self.server.nets), 1)

    def test_get_client_model__VGG5(self):
        Config.COMM = ''
        Config.model_name = 'VGG5'
        Config.split_layer = [6] * Config.K
        self.server = Server(0, '', '', test=True)
        self.assertIsNone(self.server.nets)
        self.server.get_empty_client_model(True)
        expected = 0
        for c in self.server.nets:
            self.assertEqual(len(self.server.nets[c].state_dict()), expected)

    def test_get_client_model__VGG8(self):
        Config.COMM = ''
        Config.model_name = 'VGG8'
        Config.split_layer = [9] * Config.K
        self.server = Server(0, '', '', test=True)
        self.assertIsNone(self.server.nets)
        self.server.get_empty_client_model(True)
        expected = 0
        for c in self.server.nets:
            self.assertEqual(len(self.server.nets[c].state_dict()), expected)

    def test_get_client_model__4(self):
        Config.COMM = ''
        Config.model_name = 'VGG8'
        Config.split_layer = [9] * Config.K
        self.server = Server(0, '', '', test=True)
        self.assertIsNone(self.server.nets)
        self.server.get_empty_client_model(True)
        expected = 4
        self.assertEqual(len(self.server.nets), expected)

    def test_get_client_model__8(self):
        Config.COMM = ''
        Config.model_name = 'VGG8'
        Config.split_layer = [9] * Config.K * 2
        Config.CLIENTS_LIST += ['1', '2', '3', '4']
        self.server = Server(0, '', '', test=True)
        self.assertIsNone(self.server.nets)
        self.server.get_empty_client_model(True)
        expected = 8
        self.assertEqual(len(self.server.nets), expected)
