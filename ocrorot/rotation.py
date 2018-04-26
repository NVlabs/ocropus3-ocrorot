# Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
# See the LICENSE file for licensing terms (BSD-style).

from random import randint

import numpy as np
import torch
import scipy.ndimage as ndi
from numpy import cos, sin
from torch.autograd import Variable


def get_patch(image, center, scale=1.0, alpha=0.0, shape=(256, 256)):
    yx = np.array(center, 'f')
    hw = np.array(shape, 'f')
    m = np.array([[cos(alpha), -sin(alpha)],
                  [sin(alpha), cos(alpha)]], 'f')/scale
    offset = yx - np.dot(m, hw/2.0)
    return ndi.affine_transform(image, m, offset=offset, output_shape=shape, order=1)


def get_patches(image, npatches=64, shape=(256, 256), ntrials=1024, minmean=0.02, maxmean=0.08, ralpha=5.0):
    patches = []
    h, w = image.shape
    for i in range(ntrials):
        if len(patches) >= npatches:
            break
        y, x = randint(0, h-1), randint(0, w-1)
        patch = get_patch(image, (y, x), shape=shape)
        if np.mean(patch) < minmean:
            continue
        if np.mean(patch) > maxmean:
            continue
        patches.append(patch)
    return patches


class RotationEstimator(object):
    def __init__(self, mname):
        self.model = torch.load(mname)
        self.model.eval()
        self.patchsize = (256, 256)
        self.check = True

    def predict(self, patch):
        input = torch.FloatTensor(patch).cuda()[None, None, :, :]
        output = self.model.forward(Variable(input, volatile=True))
        return np.array(output.data.cpu(), 'f')[0]

    def predictions(self, binimage):
        if self.check:
            assert binimage.ndim == 2
            assert np.amin(binimage) < 0.05
            assert np.amax(binimage) > 0.95
            middle = np.sum(np.minimum(binimage > 0.2, binimage < 0.8))
            allowed = 0.2 * np.prod(binimage.shape)
            assert middle < allowed, (middle, allowed)
        self.patches = get_patches(binimage)
        self.preds = [self.predict(p) for p in self.patches]
        return self.preds

    def prediction(self, binimage):
        predictions = self.predictions(binimage)
        self.pred = np.mean(np.array(predictions), 0)
        return self.pred

    def rotation(self, binimage):
        self.rot = np.argmax(self.prediction(binimage)) * 90.0
        return self.rot

    def bad_patches(self):
        predicted = np.argmax(self.prediction)
        selector = [np.argmax(p) != predicted for p in self.preds]
        patches = [p for i, p in enumerate(self.patches) if selector[i]]
        return patches


class SkewEstimator(RotationEstimator):
    def __init__(self, mname, arange=3.0):
        RotationEstimator.__init__(self, mname)
        self.arange = arange
        self.abuckets, = self.predict(np.zeros(self.patchsize, 'f')).shape

    def rotation(self, binimage):
        raise Exception("bad method")

    def skew(self, binimage):
        bucket = np.argmax(self.prediction(binimage))
        self.angle = (bucket * 2.0 * self.arange / self.abuckets) - self.arange
        return self.angle

    def bad_patches(self, tol=4):
        predicted = np.argmax(self.prediction)
        selector = [abs(np.argmax(p)-predicted) > tol for p in self.preds]
        patches = [p for i, p in enumerate(self.patches) if selector[i]]
        return patches
