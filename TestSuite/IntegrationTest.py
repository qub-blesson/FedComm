import os
import unittest

from cffi.setuptools_ext import execfile

import FL.Run as Run


class IntegrationTest(unittest.TestCase):
    def test_system(self):
        """os.system('python3 ../FL.Run')"""


