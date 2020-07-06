import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

x_train = np.random.rand(100000, 50)
y_train = x_train.take(axis=1, indices=-1) * np.pi
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)

model = Sequential()
model.add(Dense(16, activation='relu', input_shape=x_train.shape[1:]))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.compile(Adam(amsgrad=True), 'mse')
print('TF VERSION: {}'.format(tf.__version__))
t = time.time()
model.fit(x_train, y_train, batch_size=32, validation_data=(x_test, y_test), epochs=20)
print('time to completion: {} seconds'.format(round(time.time() - t, 4)))
