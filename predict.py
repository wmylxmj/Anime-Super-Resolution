# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 20:28:04 2019

@author: wmy
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from keras import backend as K
from keras.losses import mean_absolute_error, mean_squared_error
from keras.models import load_model
from keras.optimizers import Adam
import random
import os
from model import wdsr_a, wdsr_b
from utils import DataLoader

model = wdsr_b(scale=4, num_res_blocks=32)
model.load_weights('./weights/wdsr-b-32-x4.h5')

data_loader = DataLoader(scale=4)

def predict(model, fp, sp):
    lr = Image.open(fp)
    lr = np.asarray(lr)
    x = np.array([lr])
    y = model.predict(x)
    y = np.clip(y, 0, 255)
    y = y.astype('uint8')
    sr = Image.fromarray(y[0])
    sr.save(sp)
    pass

def resize(fp, sp, scale=4):
    lr = Image.open(fp)
    lr = lr.resize((scale*lr.size[0], scale*lr.size[1]))
    lr.save(sp)
    pass

def downsampling(fp, sp):
    hr = data_loader.imread(fp)
    lr = data_loader.downsampling(hr)
    lr.save(sp)
    pass

def copy(fp, sp):
    lr = Image.open(fp)
    lr.save(sp)
    pass

def predict_testset(setpath='datasets/test'):
    files = data_loader.search(setpath)
    for index, file in enumerate(files):
        copy(fp=file, sp='outputs/lr_' + str(index+1) + '.jpg')
        predict(model, fp=file, sp='outputs/sr_' + str(index+1) + '.jpg')
        pass
    pass

predict_testset()

