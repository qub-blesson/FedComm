import argparse
import pickle
import socket
import sys

# add all files to path
import time
import unittest

sys.path.append('../')
import Config
from FL.ClientRun import ClientRun
from FL.ServerRun import ServerRun

accuracy_vgg5 = [44.48, 53.7, 56.33, 62.91, 64.48, 65.7, 67.46, 68.4, 69.9, 70.89, 70.27, 71.72, 71.97, 71.75, 73.31,
                 73.56, 74.03, 74.44, 75.01, 73.47, 74.66, 75.48, 75.48, 76.38, 76.55, 75.61, 76.69, 76.98, 77.31,
                 77.26, 76.8, 77.47, 77.87, 78.27, 78.39, 78.11, 78.09, 78.15, 78.2, 79.08, 78.95, 78.96, 79.43, 79.22,
                 78.78, 78.02, 78.95, 80.05, 79.86, 79.65, 80.04, 80.76, 80.76, 80.85, 80.7, 81.0, 81.04, 80.97, 81.0,
                 81.04, 81.04, 80.93, 81.03, 80.96, 81.01, 81.07, 81.19, 81.26, 81.15, 81.25, 81.05, 81.23, 81.36,
                 81.23, 81.18, 81.24, 81.21, 81.3, 81.43, 81.25, 81.33, 81.35, 81.4, 81.31, 81.38, 81.26, 81.41, 81.39,
                 81.32, 81.53, 81.41, 81.25, 81.56, 81.27, 81.57, 81.5, 81.5, 81.61, 81.42, 81.51]


accuracy_vgg8 = [44.81, 55.32, 62.64, 64.01, 69.36]


class IntegrationTest(unittest.TestCase):
    def test_TCP_1round_vgg5(self):
        #time.sleep(90)
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
            ServerRun(communicator, model, stress, limiter, integrated_test=True)
            output = []
            with open('../results/output.pkl', 'rb') as f:
                output.append(pickle.load(f))

            self.assertEqual(len(output[0]['communication_time']), Config.R)  # length should be the same as the round
            j = 0
            for i in output[0]['test_acc_record']:
                self.assertAlmostEqual(i, accuracy_vgg5[j], delta=0)
                j += 1

        else:
            ClientRun(communicator, model, stress, limiter, monitor)

    def test_TCP_1round_vgg8(self):
        #time.sleep(90)
        target = '192.168.101.120'
        stress = None
        limiter = None
        communicator = 'TCP'
        model = 'VGG8'
        monitor = None
        Config.R = 1
        if target == socket.gethostbyname(socket.gethostname()) or socket.gethostname() not in Config.HOST2IP:
            if stress is None:
                stress = ''
            if limiter is None:
                limiter = ''
            ServerRun(communicator, model, stress, limiter, integrated_test=True)
            output = []
            with open('../results/output.pkl', 'rb') as f:
                output.append(pickle.load(f))

            self.assertEqual(len(output[0]['communication_time']), Config.R)  # length should be the same as the round
            j = 0
            for i in output[0]['test_acc_record']:
                self.assertAlmostEqual(i, accuracy_vgg8[j], delta=0)
                j += 1

        else:
            ClientRun(communicator, model, stress, limiter, monitor)

    def test_TCP_5round_vgg5(self):
        #time.sleep(90)
        target = '192.168.101.120'
        stress = None
        limiter = None
        communicator = 'TCP'
        model = 'VGG5'
        monitor = None
        Config.R = 5
        if target == socket.gethostbyname(socket.gethostname()) or socket.gethostname() not in Config.HOST2IP:
            if stress is None:
                stress = ''
            if limiter is None:
                limiter = ''
            ServerRun(communicator, model, stress, limiter, integrated_test=True)
            output = []
            with open('../results/output.pkl', 'rb') as f:
                output.append(pickle.load(f))

            self.assertEqual(len(output[0]['communication_time']), Config.R)  # length should be the same as the round
            j = 0
            for i in output[0]['test_acc_record']:
                self.assertAlmostEqual(i, accuracy_vgg5[j], delta=0)
                j += 1

        else:
            ClientRun(communicator, model, stress, limiter, monitor)

    def test_TCP_5round_vgg8(self):
        #time.sleep(90)
        target = '192.168.101.120'
        stress = None
        limiter = None
        communicator = 'TCP'
        model = 'VGG8'
        monitor = None
        Config.R = 5
        if target == socket.gethostbyname(socket.gethostname()) or socket.gethostname() not in Config.HOST2IP:
            if stress is None:
                stress = ''
            if limiter is None:
                limiter = ''
            ServerRun(communicator, model, stress, limiter, integrated_test=True)
            output = []
            with open('../results/output.pkl', 'rb') as f:
                output.append(pickle.load(f))

            self.assertEqual(len(output[0]['communication_time']), Config.R)  # length should be the same as the round
            j = 0
            for i in output[0]['test_acc_record']:
                self.assertAlmostEqual(i, accuracy_vgg8[j], delta=0)
                j += 1

        else:
            ClientRun(communicator, model, stress, limiter, monitor)

    def test_MQTT_5round_vgg8(self):
        #time.sleep(90)
        target = '192.168.101.120'
        stress = None
        limiter = None
        communicator = 'MQTT'
        model = 'VGG8'
        monitor = None
        Config.R = 5
        if target == socket.gethostbyname(socket.gethostname()) or socket.gethostname() not in Config.HOST2IP:
            if stress is None:
                stress = ''
            if limiter is None:
                limiter = ''
            ServerRun(communicator, model, stress, limiter, integrated_test=True)
            output = []
            with open('../results/output.pkl', 'rb') as f:
                output.append(pickle.load(f))

            self.assertEqual(len(output[0]['communication_time']), Config.R)  # length should be the same as the round
            j = 0
            for i in output[0]['test_acc_record']:
                self.assertAlmostEqual(i, accuracy_vgg8[j], delta=3)
                j += 1

        else:
            ClientRun(communicator, model, stress, limiter, monitor)

    def test_MQTT_5round_vgg5(self):
        #time.sleep(90)
        target = '192.168.101.120'
        stress = None
        limiter = None
        communicator = 'MQTT'
        model = 'VGG5'
        monitor = None
        Config.R = 5
        if target == socket.gethostbyname(socket.gethostname()) or socket.gethostname() not in Config.HOST2IP:
            if stress is None:
                stress = ''
            if limiter is None:
                limiter = ''
            ServerRun(communicator, model, stress, limiter, integrated_test=True)
            output = []
            with open('../results/output.pkl', 'rb') as f:
                output.append(pickle.load(f))

            self.assertEqual(len(output[0]['communication_time']), Config.R)  # length should be the same as the round
            j = 0
            for i in output[0]['test_acc_record']:
                self.assertAlmostEqual(i, accuracy_vgg8[j], delta=3)
                j += 1

        else:
            ClientRun(communicator, model, stress, limiter, monitor)

    def test_MQTT_1round_vgg8(self):
        #time.sleep(90)
        target = '192.168.101.120'
        stress = None
        limiter = None
        communicator = 'MQTT'
        model = 'VGG8'
        monitor = None
        Config.R = 1
        if target == socket.gethostbyname(socket.gethostname()) or socket.gethostname() not in Config.HOST2IP:
            if stress is None:
                stress = ''
            if limiter is None:
                limiter = ''
            ServerRun(communicator, model, stress, limiter, integrated_test=True)
            output = []
            with open('../results/output.pkl', 'rb') as f:
                output.append(pickle.load(f))

            self.assertEqual(len(output[0]['communication_time']), Config.R)  # length should be the same as the round
            j = 0
            for i in output[0]['test_acc_record']:
                self.assertAlmostEqual(i, accuracy_vgg8[j], delta=3)
                j += 1

        else:
            ClientRun(communicator, model, stress, limiter, monitor)

    def test_MQTT_1round_vgg5(self):
        #time.sleep(90)
        target = '192.168.101.120'
        stress = None
        limiter = None
        communicator = 'MQTT'
        model = 'VGG5'
        monitor = None
        Config.R = 1
        if target == socket.gethostbyname(socket.gethostname()) or socket.gethostname() not in Config.HOST2IP:
            if stress is None:
                stress = ''
            if limiter is None:
                limiter = ''
            ServerRun(communicator, model, stress, limiter, integrated_test=True)
            output = []
            with open('../results/output.pkl', 'rb') as f:
                output.append(pickle.load(f))

            self.assertEqual(len(output[0]['communication_time']), Config.R)  # length should be the same as the round
            j = 0
            for i in output[0]['test_acc_record']:
                self.assertAlmostEqual(i, accuracy_vgg8[j], delta=3)
                j += 1

        else:
            ClientRun(communicator, model, stress, limiter, monitor)

    def test_AMQP_5round_vgg8(self):
        #time.sleep(90)
        target = '192.168.101.120'
        stress = None
        limiter = None
        communicator = 'AMQP'
        model = 'VGG8'
        monitor = None
        Config.R = 5
        if target == socket.gethostbyname(socket.gethostname()) or socket.gethostname() not in Config.HOST2IP:
            if stress is None:
                stress = ''
            if limiter is None:
                limiter = ''
            ServerRun(communicator, model, stress, limiter, integrated_test=True)
            output = []
            with open('../results/output.pkl', 'rb') as f:
                output.append(pickle.load(f))

            self.assertEqual(len(output[0]['communication_time']), Config.R)  # length should be the same as the round
            j = 0
            for i in output[0]['test_acc_record']:
                self.assertAlmostEqual(i, accuracy_vgg8[j], delta=3)
                j += 1

        else:
            ClientRun(communicator, model, stress, limiter, monitor)

    def test_AMQP_5round_vgg5(self):
        #time.sleep(90)
        target = '192.168.101.120'
        stress = None
        limiter = None
        communicator = 'AMQP'
        model = 'VGG5'
        monitor = None
        Config.R = 5
        if target == socket.gethostbyname(socket.gethostname()) or socket.gethostname() not in Config.HOST2IP:
            if stress is None:
                stress = ''
            if limiter is None:
                limiter = ''
            ServerRun(communicator, model, stress, limiter, integrated_test=True)
            output = []
            with open('../results/output.pkl', 'rb') as f:
                output.append(pickle.load(f))

            self.assertEqual(len(output[0]['communication_time']), Config.R)  # length should be the same as the round
            j = 0
            for i in output[0]['test_acc_record']:
                self.assertAlmostEqual(i, accuracy_vgg8[j], delta=3)
                j += 1

        else:
            ClientRun(communicator, model, stress, limiter, monitor)

    def test_AMQP_1round_vgg8(self):
        #time.sleep(90)
        target = '192.168.101.120'
        stress = None
        limiter = None
        communicator = 'AMQP'
        model = 'VGG8'
        monitor = None
        Config.R = 1
        if target == socket.gethostbyname(socket.gethostname()) or socket.gethostname() not in Config.HOST2IP:
            if stress is None:
                stress = ''
            if limiter is None:
                limiter = ''
            ServerRun(communicator, model, stress, limiter, integrated_test=True)
            output = []
            with open('../results/output.pkl', 'rb') as f:
                output.append(pickle.load(f))

            self.assertEqual(len(output[0]['communication_time']), Config.R)  # length should be the same as the round
            j = 0
            for i in output[0]['test_acc_record']:
                self.assertAlmostEqual(i, accuracy_vgg8[j], delta=3)
                j += 1

        else:
            ClientRun(communicator, model, stress, limiter, monitor)

    def test_AMQP_1round_vgg5(self):
        #time.sleep(90)
        target = '192.168.101.120'
        stress = None
        limiter = None
        communicator = 'AMQP'
        model = 'VGG8'
        monitor = None
        Config.R = 1
        if target == socket.gethostbyname(socket.gethostname()) or socket.gethostname() not in Config.HOST2IP:
            if stress is None:
                stress = ''
            if limiter is None:
                limiter = ''
            ServerRun(communicator, model, stress, limiter, integrated_test=True)
            output = []
            with open('../results/output.pkl', 'rb') as f:
                output.append(pickle.load(f))

            self.assertEqual(len(output[0]['communication_time']), Config.R)  # length should be the same as the round
            j = 0
            for i in output[0]['test_acc_record']:
                self.assertAlmostEqual(i, accuracy_vgg8[j], delta=3)
                j += 1

        else:
            ClientRun(communicator, model, stress, limiter, monitor)