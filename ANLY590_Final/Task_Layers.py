#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 13:49:18 2019

@author: geoff
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt



# Numbers of units in Hidden Layers of Task Specific Layers
t1_l1_units = 32
t1_l2_units = 16
t1_l3_units = 4

t2_l1_units = 32
t2_l2_units = 16
t2_l3_units = 4



# Setup for testing / feeding into Neural Network

from_Bi_LSTM = tf.placeholder(tf.float32, shape = (None, n_features))
task_1_y = tf.placeholder(tf.float32, shape = (None, 1))
task_2_y = tf.placeholder(tf.float32, shape = (None, 1))

# Task 1 Layers

task_1_layer1 = tf.layers.dense(inputs = from_Bi_LSTM, 
                                      units = t1_l1_units, 
                                      activation = tf.nn.relu, 
                                      use_bias = True,
                                      kernel_initializer = tf.initializers.glorot_normal,
                                      name = 'Task_1_Layer_1')
task_1_layer2 = tf.layers.dense(inputs = task_1_layer1, 
                                      units = t1_l2_units, 
                                      activation = tf.nn.relu, 
                                      use_bias = True,
                                      kernel_initializer = tf.initializers.glorot_normal,
                                      name = 'Task_1_Layer_2')
task_1_layer3 = tf.layers.dense(inputs = task_1_layer2, 
                                      units = t1_l3_units, 
                                      activation = tf.nn.relu, 
                                      use_bias = True,
                                      kernel_initializer = tf.initializers.glorot_normal,
                                      name = 'Task_1_Layer_3')
task_1_output = tf.layers.dense(inputs = task_1_layer3, 
                                      units = 1, 
                                      activation = tf.nn.sigmoid,
                                      use_bias = True,
                                      kernel_initializer = tf.initializers.glorot_normal,
                                      name = 'Task_1_Output')

# Task 2 Layers

task_2_layer1 = tf.layers.dense(inputs = from_Bi_LSTM, 
                                      units = t2_l1_units, 
                                      activation = tf.nn.relu, 
                                      use_bias = True,
                                      kernel_initializer = tf.initializers.glorot_normal,
                                      name = 'Task_2_Layer_1')
task_2_layer2 = tf.layers.dense(inputs = task_2_layer1, 
                                      units = t2_l2_units, 
                                      activation = tf.nn.relu, 
                                      use_bias = True,
                                      kernel_initializer = tf.initializers.glorot_normal,
                                      name = 'Task_2_Layer_2')
task_2_layer3 = tf.layers.dense(inputs = task_2_layer2, 
                                      units = t2_l3_units, 
                                      activation = tf.nn.relu, 
                                      use_bias = True,
                                      kernel_initializer = tf.initializers.glorot_normal,
                                      name = 'Task_2_Layer_3')
task_2_output = tf.layers.dense(inputs = task_2_layer3, 
                                      units = 1, 
                                      activation = tf.nn.sigmoid,
                                      use_bias = True,
                                      kernel_initializer = tf.initializers.glorot_normal,
                                      name = 'Task_2_Output')

# Calculate Loss
Y1_Loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = task_1_y, logits = task_1_output)
Y2_Loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = task_2_y, logits = task_2_output)
Joint_Loss = Y1_Loss + Y2_Loss


# optimisers
Optimiser = tf.train.AdamOptimizer().minimize(Joint_Loss)
Y1_op = tf.train.AdamOptimizer().minimize(Y1_Loss)
Y2_op = tf.train.AdamOptimizer().minimize(Y2_Loss)


with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    _, Joint_Loss = session.run([Optimiser, Joint_Loss], feed_dict = 
                    {
                      X: ,
                      Y1: ,
                      Y2: 
                      })
    print(Joint_Loss)



















'''

task_1_layer1 = keras.layers.Dense(32, activation = 'relu') (from_Bi_LSTM)
task_1_layer2 = keras.layers.Dense(16, activation = 'relu') (task_1_layer1)
task_1_layer3 = keras.layers.Dense(4, activation = 'relu') (task_1_layer2)
task_1_output = keras.layers.Dense(2, activation = 'sigmoid') (task_1_layer3)

task_2_layer1 = keras.layers.Dense(32, activation = 'relu') (from_Bi_LSTM)
task_2_layer2 = keras.layers.Dense(16, activation = 'relu') (task_2_layer1)
task_2_layer3 = keras.layers.Dense(4, activation = 'relu') (task_2_layer2)
task_2_output = keras.layers.Dense(2, activation = 'sigmoid') (task_2_layer3)
















        self.observation = tf.placeholder(tf.float32, shape=(None, n_features))
        self.reward = tf.placeholder(tf.float32, shape=(None, 1))
        
        # Layers Construction
        self.layer1 = tf.layers.dense(inputs = self.observation, 
                                      units = n_hidden1, 
                                      activation = tf.nn.relu, 
                                      use_bias = True,
                                      kernel_initializer = tf.initializers.glorot_normal,
                                      name = 'CriticHidden1')
        self.layer2 = tf.layers.dense(inputs = self.layer1, 
                                      units = n_hidden2, 
                                      activation = tf.nn.relu, 
                                      use_bias = True,
                                      kernel_initializer = tf.initializers.glorot_normal,
                                      name = 'CriticHidden2')
        self.layer3 = tf.layers.dense(inputs = self.layer2, 
                                      units = 1,
                                      name = 'ValueEstimate')
            
        # Optimizer
        self.opt = tf.train.AdamOptimizer(learning_rate)
        self.loss = tf.reduce_mean(tf.square(self.reward - self.layer3))
        self.train = self.opt.minimize(self.loss)
        
'''