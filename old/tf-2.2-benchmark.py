import time
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

x_train = np.random.rand(100000, 50)
y_train = x_train.take(axis=1, indices=-1) * np.pi

model = Sequential()
model.add(Dense(16, activation='relu', input_shape=x_train.shape[1:]))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.compile(Adam(amsgrad=True), 'mse')
t = time.time()
model.fit(x_train, y_train, batch_size=32, validation_split=0.2, epochs=20)
print('time to completion: {} seconds'.format(round(time.time() - t, 4)))
