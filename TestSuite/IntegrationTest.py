import argparse
import socket
import sys

# add all files to path
import unittest

sys.path.append('../')
import Config
from FL.ClientRun import ClientRun
from FL.ServerRun import ServerRun


class IntegrationTest(unittest.TestCase):
    def test_TCP_1round_vgg5(self):
        target = '192.168.101.120'
        stress = None
        limiter = None
        communicator = 'TCP'
        model = 'VGG5'
        monitor = None
        Config.R = 1
        if target == socket.gethostbyname(socket.gethostname()) or socket.gethostname() not in Config.HOST2IP:
            if stress is None:
                stress = ''
            if limiter is None:
                limiter = ''
            ServerRun(communicator, model, stress, limiter)
        else:
            ClientRun(communicator, model, stress, limiter, monitor)

        self.assertTrue(False)


if __name__ == '__main__':
    intTest = IntegrationTest()
    intTest.test_TCP_1round_vgg5()
