# Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
# See the LICENSE file for licensing terms (BSD-style).

import torch
import pytorch_fft.fft.autograd as afft
from torch import nn
from torch.autograd import Variable


class Spectrum(nn.Module):
    def __init__(self, nonlin="logplus1"):
        nn.Module.__init__(self)
        self.fft2d = afft.Fft2d()
        self.nonlin = nonlin

    def forward(self, x):
        y = Variable(torch.zeros(x.size()).cuda())
        #print type(x), type(y)
        #print type(x.data), type(y.data)
        re, im = self.fft2d(x, y)
        #print type(re), type(im)
        #print type(re.data), type(im.data)
        mag = (re**2 + im**2)
        if self.nonlin == None:
            return mag
        elif self.nonlin == "logplus1":
            return torch.log(1.0 + mag)
        elif self.nonlin == "sqrt":
            return torch.mag ** .5
        else:
            raise Exception("{}: unknown nonlinearity".format(self.nonlin))

    def __repr__(self):
        return "Spectrum"
