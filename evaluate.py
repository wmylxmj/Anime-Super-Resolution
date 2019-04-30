# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 21:24:36 2019

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

def evaluate_test(model, setpath='datasets/train', difficulty='easy', name='evaluate'):
    images = data_loader.search(setpath)
    image = random.choice(images)
    hr = data_loader.imread(image)
    resize = (hr.size[0]//data_loader.scale, hr.size[1]//data_loader.scale)
    hidden_scale = random.uniform(1, 3)
    radius = random.uniform(1, 3)
    if difficulty=='easy':
        hidden_scale = random.uniform(1, 1.5)
        radius = random.uniform(1, 1.5)
        pass
    elif difficulty=='normal':
        hidden_scale = random.uniform(1.5, 2)
        radius = random.uniform(1.5, 2)
        pass
    elif difficulty=='hard':
        hidden_scale = random.uniform(2, 2.5)
        radius = random.uniform(2, 2.5)
        pass
    elif difficulty=='lunatic':
        hidden_scale = random.uniform(2.5, 3)
        radius = random.uniform(2.5, 3)
        pass
    else:
        raise ValueError("unknown difficulty")
    hidden_resize = (int(resize[0]/hidden_scale), int(resize[1]/hidden_scale))
    lr = data_loader.gaussianblur(hr, radius)
    lr = lr.resize(hidden_resize)
    lr = lr.resize(resize)
    lr_resize = lr.resize(hr.size)
    lr =  np.asarray(lr)
    sr = model.predict(np.array([lr]))[0]
    sr = np.clip(sr, 0, 255)
    sr = sr.astype('uint8')
    lr = Image.fromarray(lr)
    sr = Image.fromarray(sr)
    lr_resize.save("images/" + name + "_lr.jpg")
    sr.save("images/" + name + "_sr.jpg")
    hr.save("images/" + name + "_hr.jpg")
    pass

evaluate_test(model, difficulty='easy', name='easy')
evaluate_test(model, difficulty='normal', name='normal')
evaluate_test(model, difficulty='hard', name='hard')
evaluate_test(model, difficulty='lunatic', name='lunatic')

