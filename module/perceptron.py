#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

#predict
def predict(wvec, xmat):
    out = np.dot(xmat, wvec)
    res = np.ones(xmat.shape[0])
    negative = np.where(out < 0)
    res[negative] = -1
    return [res, out]

#train
def train(wvec, xmat, label, cof):
    [res, out] = predict(wvec,xmat)
    tmp_vec = out * label
    negative = np.where(tmp_vec < 0)
    if len(negative[0]) > 0:
        for i in negative[0]:
            wvec += cof * label[i] * xmat[i]

if __name__ == '__main__':
    num_train = 500
    cof = 0.1
    x1 = np.random.randint(0,10,(10,2))
    x2 = np.random.randint(10,20,(10,2))
    x = np.r_[x1,x2]
    x = np.c_[x, np.ones(20)]

    label = np.ones(20)
    label[10:] = -1
    w = np.random.random(3)
    for i in range(num_train):
        train(w, x, label, cof)

    res, out = predict(w, x)
    print res;
