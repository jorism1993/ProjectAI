# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 09:48:48 2018

@author: Joris
"""
import keras 

model = keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_tensor=None, input_shape=(200,200,3), pooling=None)

model.summary()
