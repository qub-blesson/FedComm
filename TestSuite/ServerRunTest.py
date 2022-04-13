import unittest
import time
import torch
import pickle
import argparse
from FL.Server import Server
import Config
import logging
import sys
import numpy as np
from FL.ServerRun import ServerRun

# set logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# add all files to path
sys.path.append('../')


class ServerRunTest(unittest.TestCase):
    def test_update_model_VGG5(self):
        self.server = ServerRun(None, None, None, None, True)
        self.server.model = 'VGG5'
        self.server.communicator = 'MQTT'
        self.assertNotEqual(Config.model_name, self.server.model)
        VGG5_model_length = 7
        self.assertNotEqual(Config.model_len, VGG5_model_length)
        self.server.update_model()
        self.assertEqual(Config.model_name, self.server.model)
        self.assertEqual(Config.model_len, VGG5_model_length)

    def test_create_results_file(self):
        self.server = ServerRun(None, None, 'stress', 'lim', True)
        self.assertEqual(self.server.results, '')
        self.assertDictEqual(self.server.res, {})
        self.server.create_results_file()
        self.assertEqual(self.server.results,
                         '../results/FedBench_TCP_lim_stress_VGG8.pkl')

