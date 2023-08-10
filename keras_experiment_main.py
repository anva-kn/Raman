#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 02:33:27 2020

@author: akula
"""


import numpy as np
import pandas as pd
import shelve
# import keras_experiment
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

filename='shelve_save_data.out'

my_shelf = shelve.open(filename)
for key in my_shelf:
    globals()[key]=my_shelf[key]
my_shelf.close()

one_spectrum = data_15000[0][0]
print(data_15000.shape, f_sup.shape)

whole_data = np.zeros((20000, 1600))
print(whole_data.shape)
whole_data[:2500] = np.copy(data_15000[0]/np.max(data_15000[0]))
whole_data[2500:5000] = np.copy(data_15000[1]/np.max(data_15000[1]))
whole_data[5000:7500] = np.copy(data_15000[2]/np.max(data_15000[2]))
whole_data[7500:10000] = np.copy(data_15000[3]/np.max(data_15000[3]))
whole_data[10000:12500] = np.copy(data_1500[0]/np.max(data_1500[0]))
whole_data[12500:15000] = np.copy(data_1500[1]/np.max(data_1500[1]))
whole_data[15000:17500] = np.copy(data_1500[2]/np.max(data_1500[2]))
whole_data[17500:20000] = np.copy(data_1500[3]/np.max(data_1500[3]))
labels = np.zeros((20000, 2))
labels[0:10000,0] = 1
labels[10000:20000,1] = 1

idx = np.random.permutation(len(labels))
# plt.plot(f_sup, whole_data[1299], '-')#, label='one spectrum')
# plt.legend()
# plt.show()
X = np.copy(whole_data)[idx]
y = np.copy(labels)[idx]
x_train, y_train = X[:16000], y[:16000]
x_test, y_test = X[16000:], y[16000:]
x_train_another, y_train_another = np.copy(X), np.copy(y)

# print(y)
# print(X.shape, y.shape)
# checksumbitch = np.where(whole_data == 0)
# indeces = np.unique(checksumbitch[0])

model = keras.Sequential(
    [
        keras.Input(shape=(1600,)),
        layers.Dense(400, activation="relu"),
        layers.Dense(100, activation="relu"),
        layers.Dense(2, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=['accuracy'],
)

history = model.fit(x_train_another, y_train_another, validation_split=0.2, epochs=30, batch_size=256)

# Evaluate the model on the test data using `evaluate`
# print("Evaluate on test data")
# results = model.evaluate(x_test, y_test, batch_size=128)
# print("test loss, test acc:", results)

np.mean(model.predict(x_test).argmax(axis=-1) == y_test.argmax(axis=-1))

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.figure(2)
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()