#!/usr/bin/python

import random as pyr
import argparse

import numpy as np
import scipy.ndimage as ndi
from pylab import *
from scipy import ndimage as ndi
from dlinputs import gopen, utils
from matplotlib import *

rc("image", cmap="gray")


rc("image", cmap="gray")
ion()
parser = argparse.ArgumentParser("degrade pages for binarization")
parser.add_argument("--display", action="store_true")
parser.add_argument("input")
parser.add_argument("output")
args = parser.parse_args()


def sigmoid(x):
    return 1/(1+exp(-x))


def random_trs(translation=0.05, rotation=2.0, scale=0.1, aniso=0.1):
    if not isinstance(translation, (tuple, list)):
        translation = (-translation, translation)
    if not isinstance(rotation, (tuple, list)):
        rotation = (-rotation, rotation)
    if not isinstance(scale, (tuple, list)):
        scale = (-scale, scale)
    if not isinstance(aniso, (tuple, list)):
        aniso = (-aniso, aniso)
    dx = pyr.uniform(*translation)
    dy = pyr.uniform(*translation)
    alpha = pyr.uniform(*rotation)
    alpha = alpha * pi / 180.0
    scale = 10**pyr.uniform(*scale)
    aniso = 10**pyr.uniform(*aniso)
    c = cos(alpha)
    s = sin(alpha)
    # print "\t", (dx, dy), alpha, scale, aniso
    sm = np.array([[scale / aniso, 0], [0, scale * aniso]], 'f')
    m = np.array([[c, -s], [s, c]], 'f')
    m = np.dot(sm, m)

    def f(image, order=1):
        w, h = image.shape
        c = np.array([w, h]) / 2.0
        d = c - np.dot(m, c) + np.array([dx * w, dy * h])
        return ndi.affine_transform(image, m, offset=d, order=order, mode="nearest", output=dtype("f"))

    return f, dict(translation=(dx, dy), alpha=alpha, scale=scale, aniso=aniso)


def make_at_scale(shape, scale):
    h, w = shape
    h0, w0 = int(h/scale+1), int(w/scale+1)
    data = rand(h0, w0)
    result = ndi.zoom(data, scale)
    return result[:h, :w]


def make_random(shape, lohi, scales, weights=None):
    if weights is None:
        weights = [1.0] * len(scales)
    result = make_at_scale(shape, scales[0]) * weights[0]
    for s, w in zip(scales, weights):
        result += make_at_scale(shape, s) * w
    lo, hi = lohi
    result -= amin(result)
    result /= amax(result)
    result *= (hi-lo)
    result += lo
    return result


def make_all_random(page):
    blur = 3 * rand()
    sep = 0.1+0.2*rand()
    while 1:
        scales = add.accumulate(rand(4))
        scales = 10**scales
        if scales[-1] < 500:
            break
    weights = rand(4)
    mask = ndi.gaussian_filter(page, blur)
    mask /= amax(mask)
    bg = make_random(page.shape, (0.0, 0.5-sep), scales, weights)
    fg = make_random(page.shape, (0.5+sep, 1.0), scales, weights)
    degraded = mask * bg + (1.0-mask) * fg
    lo, hi = 0.2 * rand(), 0.8 + 0.2*rand()
    params = dict(blur=blur, sep=sep, clip=(lo, hi),
                  scales=scales, weights=weights)
    clipped = clip(degraded, lo, hi)
    clipped -= amin(clipped)
    clipped /= amax(clipped)
    shifted = clipped
    return array(shifted, 'f'), params


data = gopen.open_source(args.input)
sink = gopen.open_sink(args.output)

for sample in data:
    print
    utils.print_sample(sample)
    page = sample["framed.png"]
    trs, _ = random_trs()
    page = trs(page)
    degraded, params = make_all_random(page)
    if args.display:
        suptitle(repr(params))
        subplot(121)
        imshow(degraded, vmin=0, vmax=1)
        subplot(122)
        imshow(degraded[1500:2000, 1500:2000], vmin=0, vmax=1)
        ginput(1, 0.001)
    output = {
        "__key__": sample["__key__"],
        "gray.png": degraded,
        "bin.png": page
    }
    utils.print_sample(output)
    sink.write(output)
sink.close()
