import unittest
import sys

sys.path.append('../')
from FL.ClientRun import *
import Config


class ClientRunTest(unittest.TestCase):
    def test_set_args__VGG5(self):
        self.client = ClientRun(None, None, None, None, None, True)
        self.client.model = 'VGG5'
        self.client.communicator = 'MQTT'
        self.assertNotEqual(Config.model_name, self.client.model)
        VGG5_model_length = 7
        self.assertNotEqual(Config.model_len, VGG5_model_length)
        self.client.set_args()
        self.assertEqual(Config.model_name, self.client.model)
        self.assertEqual(Config.model_len, VGG5_model_length)

    def test_init_client_values(self):
        self.client = ClientRun(None, None, None, None, None, True)
        self.assertIsNone(self.client.datalen)
        self.assertIsNone(self.client.LR)
        self.client.init_client_values()
        datalength = 12500
        self.assertEqual(self.client.datalen, datalength)
        LR = 0.01
        self.assertEqual(self.client.LR, LR)
