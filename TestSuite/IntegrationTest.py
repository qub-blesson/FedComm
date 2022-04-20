import os
import unittest

from cffi.setuptools_ext import execfile

import FL.Run as Run


class IntegrationTest(unittest.TestCase):
    def test_system(self):
        os.system('python3 ../FL.Run --communicator TCP --model VGG5 --limiter Wi-Fi --rounds 1')
        self.assertTrue(False)
