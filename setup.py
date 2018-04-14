#!/bin/env python
# Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
# See the LICENSE file for licensing terms (BSD-style).

from __future__ import print_function

import glob
import os
import os.path
import sys
import time
import traceback
import urllib
from distutils.core import setup  # , Extension, Command

assert sys.version_info[0] == 2 and sys.version_info[1] >= 7,\
    "requires Python version 2.7 or later, but not Python 3.x"


scripts = """
ocrorot-train
""".split()

setup(
    name='ocrorot',
    version='v0.0',
    author="Thomas Breuel",
    description="Page rotation detection/correction.",
    packages=["ocrorot"],
    scripts=scripts,
)
