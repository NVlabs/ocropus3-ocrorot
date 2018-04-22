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

remote = "https://storage.googleapis.com/tmb-models/"
models = ["rot-000003456-020897.pt", "logskew-000015808-000132.pt"]
for model in models:
    if not os.path.exists(model):
        print("downloading", remote+model)
        urllib.URLopener().retrieve(remote+model, model)

scripts = """
ocrorot-train
ocroskew-train
""".split()

setup(
    name='ocrorot',
    version='v0.0',
    author="Thomas Breuel",
    description="Page rotation detection/correction.",
    packages=["ocrorot"],
    scripts=scripts,
    data_files = [("share/ocrorot", models)]
)
