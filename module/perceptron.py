#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

#predict
def predict(wvec, xmat):
    out = np.dot(xmat, wvec)
    res = np.ones((xmat.shape[0], 1))
    negative = np.where(out < 0)
    res[negative] = -1
    return [res, out]

#train
def train(wvec, xmat, label):
    [res, out] = predict(wvec,xmat)
    c = 0.5
    tmp_vec = out * label
    negative = np.where(tmp_vec < 0)
    if len(negative[0]) > 0:
        for i in negative[0]:
            wvec += c * label[i] * xmat[i]
